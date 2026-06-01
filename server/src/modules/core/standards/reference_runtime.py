from __future__ import annotations

import gc
import threading
import time
from contextlib import suppress
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Literal

import numpy as np
import structlog
import torch
from app.config import settings
from app.observability import log_event
from modules.core.standards.reference_constants import (
    LIGHTGLUE_TORCH_WEIGHTS_PATH,
    SUPERPOINT_REALTIME_MAX_KEYPOINTS,
    SUPERPOINT_TORCH_WEIGHTS_PATH,
)

logger = structlog.get_logger(__name__)

AlignmentDeviceKind = Literal["cuda", "cpu"]


@dataclass(slots=True)
class TorchReferenceRuntime:
    device_kind: AlignmentDeviceKind
    device: torch.device
    extractor: torch.nn.Module
    matcher: torch.nn.Module
    lock: threading.RLock

    @property
    def label(self) -> str:
        return f"torch_{self.device_kind}"


@lru_cache(maxsize=2)
def get_reference_runtime(max_keypoints: int | None = None) -> TorchReferenceRuntime:
    requested_device = settings.ALIGNMENT_DEVICE

    candidates: list[AlignmentDeviceKind]
    if requested_device == "cuda":
        candidates = ["cuda"]
    elif requested_device == "cpu":
        candidates = ["cpu"]
    else:
        candidates = ["cuda", "cpu"]

    errors: list[str] = []

    for device_kind in candidates:
        try:
            runtime = _build_runtime(
                device_kind=device_kind,
                max_keypoints=max_keypoints,
            )
            _warmup_runtime(runtime)
            log_event(
                logger,
                "info",
                "inspection.alignment.runtime.ready",
                device=runtime.label,
                max_keypoints=max_keypoints,
            )
            return runtime
        except Exception as exc:
            errors.append(f"{device_kind}: {exc}")
            _release_torch_memory()
            log_event(
                logger,
                "warning",
                "inspection.alignment.runtime.failed",
                device=device_kind,
                max_keypoints=max_keypoints,
                error_type=type(exc).__name__,
                exception=exc,
            )

            if _should_abort_runtime_fallback(exc=exc, device_kind=device_kind):
                raise

            if requested_device in {"cuda", "cpu"}:
                break

    raise RuntimeError(
        "Не удалось инициализировать PyTorch alignment runtime: " + "; ".join(errors)
    )


def compute_superpoint_features(
    image_tensor: torch.Tensor,
    *,
    max_keypoints: int | None,
) -> dict[str, torch.Tensor]:
    runtime = get_reference_runtime(max_keypoints=max_keypoints)

    with runtime.lock, torch.inference_mode():
        tensor = image_tensor.to(runtime.device, non_blocking=False)
        features = _extract(runtime.extractor, tensor)

    return _detach_feature_dict(features)


def match_feature_arrays(
    *,
    reference_keypoints: np.ndarray,
    reference_descriptors: np.ndarray,
    reference_size: tuple[int, int],
    frame_keypoints: np.ndarray,
    frame_descriptors: np.ndarray,
    frame_size: tuple[int, int],
    max_keypoints: int | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    runtime = get_reference_runtime(max_keypoints=max_keypoints)

    ref_features = _features_to_device(
        keypoints=reference_keypoints,
        descriptors=reference_descriptors,
        image_size=reference_size,
        device=runtime.device,
    )
    frame_features = _features_to_device(
        keypoints=frame_keypoints,
        descriptors=frame_descriptors,
        image_size=frame_size,
        device=runtime.device,
    )

    with runtime.lock, torch.inference_mode():
        output = runtime.matcher(
            {
                "image0": ref_features,
                "image1": frame_features,
            }
        )

    pairs = _extract_match_pairs(output)
    if pairs is None or pairs.size == 0:
        return None

    scores = _extract_match_scores(output)
    if scores is not None and scores.shape[0] == pairs.shape[0]:
        order = np.argsort(-scores)
        pairs = pairs[order]

    max_pairs = 512
    if pairs.shape[0] > max_pairs:
        pairs = pairs[:max_pairs]

    ref_indices = pairs[:, 0]
    frame_indices = pairs[:, 1]

    if ref_indices.size == 0:
        return None

    ref_points = reference_keypoints[ref_indices].astype(np.float32, copy=False)
    frame_points = frame_keypoints[frame_indices].astype(np.float32, copy=False)

    return ref_points, frame_points


def warmup_reference_matching() -> None:
    started_at = time.perf_counter()

    try:
        runtime = get_reference_runtime(
            max_keypoints=SUPERPOINT_REALTIME_MAX_KEYPOINTS
        )
    except Exception as exc:
        log_event(
            logger,
            "warning",
            "inspection.alignment.runtime.warmup_failed",
            error_type=type(exc).__name__,
            exception=exc,
        )
        return

    done_at = time.perf_counter()
    log_event(
        logger,
        "info",
        "inspection.alignment.runtime.warmup_finished",
        device=runtime.label,
        warmup_ms=round((done_at - started_at) * 1000, 1),
    )


def clear_reference_runtime_cache() -> None:
    get_reference_runtime.cache_clear()
    _release_torch_memory()


def _build_runtime(
    *,
    device_kind: AlignmentDeviceKind,
    max_keypoints: int | None,
) -> TorchReferenceRuntime:
    from lightglue import LightGlue, SuperPoint

    device = _resolve_torch_device(device_kind)

    extractor = SuperPoint(max_num_keypoints=max_keypoints).eval()
    extractor.load_state_dict(
        torch.load(SUPERPOINT_TORCH_WEIGHTS_PATH, map_location="cpu")
    )

    matcher = LightGlue(
        features=None,
        input_dim=256,
        descriptor_dim=256,
        weights=None,
    ).eval()
    matcher_state = _load_lightglue_state_dict(
        LIGHTGLUE_TORCH_WEIGHTS_PATH,
        n_layers=matcher.conf.n_layers,
    )
    missing, unexpected = matcher.load_state_dict(matcher_state, strict=False)

    significant_missing = [key for key in missing if key != "confidence_thresholds"]
    if significant_missing:
        raise RuntimeError(
            "LightGlue weights несовместимы. Missing keys: "
            + ", ".join(significant_missing[:20])
        )

    if unexpected:
        raise RuntimeError(
            "LightGlue weights содержат неожиданные ключи: "
            + ", ".join(unexpected[:20])
        )

    extractor = extractor.to(device)
    matcher = matcher.to(device)

    return TorchReferenceRuntime(
        device_kind=device_kind,
        device=device,
        extractor=extractor,
        matcher=matcher,
        lock=threading.RLock(),
    )


def _resolve_torch_device(device_kind: AlignmentDeviceKind) -> torch.device:
    if device_kind == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("torch.cuda недоступен")
        return torch.device("cuda:0")

    return torch.device("cpu")


def _warmup_runtime(runtime: TorchReferenceRuntime) -> None:
    dummy = torch.rand(
        1,
        1,
        128,
        128,
        dtype=torch.float32,
        device=runtime.device,
    )

    with runtime.lock, torch.inference_mode():
        features = _extract(runtime.extractor, dummy)

        if _feature_count(features) >= 4:
            runtime.matcher(
                {
                    "image0": features,
                    "image1": features,
                }
            )


def _extract(
    extractor: torch.nn.Module,
    image: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if hasattr(extractor, "extract"):
        result = extractor.extract(image)
    else:
        result = extractor({"image": image})

    if not isinstance(result, dict):
        raise RuntimeError("SuperPoint вернул неожиданный результат")

    return result


def _detach_feature_dict(features: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    result: dict[str, torch.Tensor] = {}

    for key, value in features.items():
        if torch.is_tensor(value):
            result[key] = value.detach().cpu()
        else:
            result[key] = value

    return result


def _feature_count(features: dict[str, torch.Tensor]) -> int:
    keypoints = features.get("keypoints")
    if keypoints is None:
        return 0

    if keypoints.ndim == 3:
        return int(keypoints.shape[1])

    return int(keypoints.shape[0])


def _features_to_device(
    *,
    keypoints: np.ndarray,
    descriptors: np.ndarray,
    image_size: tuple[int, int],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    width, height = image_size

    kpts = torch.from_numpy(np.asarray(keypoints, dtype=np.float32)).to(device)
    desc = torch.from_numpy(np.asarray(descriptors, dtype=np.float32)).to(device)

    if kpts.ndim == 2:
        kpts = kpts.unsqueeze(0)

    if desc.ndim == 2:
        desc = desc.unsqueeze(0)

    image_size_tensor = torch.tensor(
        [[float(width), float(height)]],
        dtype=torch.float32,
        device=device,
    )

    return {
        "keypoints": kpts,
        "descriptors": desc,
        "image_size": image_size_tensor,
    }


def _load_lightglue_state_dict(
    weights_path,
    *,
    n_layers: int,
) -> dict[str, torch.Tensor]:
    raw = torch.load(weights_path, map_location="cpu")

    if (
        isinstance(raw, dict)
        and "state_dict" in raw
        and isinstance(raw["state_dict"], dict)
    ):
        raw = raw["state_dict"]

    if not isinstance(raw, dict):
        raise RuntimeError(f"Некорректный формат LightGlue weights: {weights_path}")

    state_dict = {str(key): value for key, value in raw.items()}

    for index in range(n_layers):
        state_dict = {
            key.replace(f"self_attn.{index}", f"transformers.{index}.self_attn"): value
            for key, value in state_dict.items()
        }
        state_dict = {
            key.replace(
                f"cross_attn.{index}", f"transformers.{index}.cross_attn"
            ): value
            for key, value in state_dict.items()
        }

    state_dict = {
        key: value
        for key, value in state_dict.items()
        if not (key.startswith("log_assignment.") and key.endswith(".r"))
    }

    return state_dict


def _extract_match_pairs(output: dict[str, Any]) -> np.ndarray | None:
    if "matches" in output:
        matches = _to_numpy_cpu(output["matches"]).astype(np.int64, copy=False)

        if matches.ndim == 3:
            matches = matches[0]

        if matches.ndim == 2 and matches.shape[1] == 2 and matches.size > 0:
            valid = (matches[:, 0] >= 0) & (matches[:, 1] >= 0)
            return matches[valid]

    if "matches0" in output:
        matches0 = _to_numpy_cpu(output["matches0"]).astype(np.int64, copy=False)

        if matches0.ndim == 2:
            matches0 = matches0[0]

        ref_indices = np.arange(matches0.shape[0], dtype=np.int64)
        valid = matches0 >= 0

        if not np.any(valid):
            return None

        return np.stack([ref_indices[valid], matches0[valid]], axis=1)

    return None


def _extract_match_scores(output: dict[str, Any]) -> np.ndarray | None:
    for key in ("scores", "matching_scores", "matching_scores0"):
        if key not in output:
            continue

        scores = _to_numpy_cpu(output[key]).astype(np.float32, copy=False)

        if scores.ndim == 2:
            scores = scores[0]

        return scores.reshape(-1)

    return None


def _to_numpy_cpu(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().cpu().numpy()

    if isinstance(value, list | tuple):
        return np.asarray(
            [
                item.detach().cpu().numpy() if torch.is_tensor(item) else item
                for item in value
            ]
        )

    return np.asarray(value)


def _release_torch_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.empty_cache()


def _should_abort_runtime_fallback(
    *,
    exc: Exception,
    device_kind: AlignmentDeviceKind,
) -> bool:
    return device_kind == "cuda" and isinstance(exc, torch.OutOfMemoryError)
