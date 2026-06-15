from __future__ import annotations

import gc
import threading
import time
from collections import OrderedDict
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import structlog
import torch
from app.config import settings
from app.observability import log_event
from modules.yolo.inspection.domain.types import YoloDetection
from ultralytics import YOLO

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class _TorchYoloModel:
    model: YOLO
    class_names: dict[int, str]
    lock: threading.RLock
    imgsz: int | None


_TORCH_CACHE_MAX_MODELS = 2
_torch_cache: OrderedDict[str, tuple[float, _TorchYoloModel]] = OrderedDict()
_torch_cache_lock = threading.Lock()


def run_inference(
    *,
    weights_path: Path,
    image: np.ndarray | None = None,
    conf: float = 0.25,
    iou: float = 0.5,
    imgsz: int | None = None,
) -> list[YoloDetection]:
    if image is None:
        raise ValueError("Передайте изображение для проверки")

    return _run_inference_torch(
        weights_path=weights_path,
        image=np.ascontiguousarray(image),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
    )


def warmup_model(*, weights_path: Path, imgsz: int) -> None:
    t0 = time.perf_counter()
    model = _load_torch_model(weights_path)
    t_load = time.perf_counter()

    actual_imgsz = imgsz or model.imgsz or 640
    dummy = np.zeros((actual_imgsz, actual_imgsz, 3), dtype=np.uint8)

    try:
        _run_inference_torch(
            weights_path=weights_path,
            image=dummy,
            conf=0.25,
            iou=0.5,
            imgsz=actual_imgsz,
        )
    except Exception as exc:
        log_event(
            logger,
            "warning",
            "inspection.yolo.warmup.failed",
            weights_path=weights_path,
            imgsz=actual_imgsz,
            error_type=type(exc).__name__,
            exception=exc,
        )
        return

    t_warmup = time.perf_counter()
    log_event(
        logger,
        "info",
        "inspection.yolo.warmup.finished",
        weights_path=weights_path,
        imgsz=actual_imgsz,
        load_ms=round((t_load - t0) * 1000, 1),
        warmup_ms=round((t_warmup - t_load) * 1000, 1),
        device=_model_device(model.model),
    )


def _load_torch_model(weights_path: Path) -> _TorchYoloModel:
    if weights_path.suffix.lower() != ".pt":
        raise ValueError(
            "ONNX больше не поддерживается. Для проверки нужна PyTorch-модель .pt: "
            f"{weights_path}"
        )

    key = str(weights_path.resolve())
    mtime = weights_path.stat().st_mtime

    with _torch_cache_lock:
        cached = _torch_cache.get(key)
        if cached is not None and cached[0] == mtime:
            _torch_cache.move_to_end(key)
            return cached[1]

    load_started_at = time.perf_counter()
    log_event(
        logger,
        "info",
        "inspection.yolo.model.load_started",
        weights_path=key,
    )

    yolo = YOLO(str(weights_path))
    class_names = _extract_class_names(yolo)
    model = _TorchYoloModel(
        model=yolo,
        class_names=class_names,
        lock=threading.RLock(),
        imgsz=_extract_default_imgsz(yolo),
    )

    evicted_models: list[_TorchYoloModel] = []

    with _torch_cache_lock:
        _torch_cache[key] = (mtime, model)
        _torch_cache.move_to_end(key)

        while len(_torch_cache) > _TORCH_CACHE_MAX_MODELS:
            _, (_, evicted_model) = _torch_cache.popitem(last=False)
            evicted_models.append(evicted_model)

    for evicted_model in evicted_models:
        _dispose_yolo_model(evicted_model)
    evicted_models.clear()
    _release_torch_memory()

    log_event(
        logger,
        "info",
        "inspection.yolo.model.loaded",
        weights_path=key,
        imgsz=model.imgsz,
        num_classes=len(class_names),
        task=getattr(yolo, "task", None),
        device=_model_device(yolo),
        duration_ms=round((time.perf_counter() - load_started_at) * 1000, 1),
    )

    return model


def _run_inference_torch(
    *,
    weights_path: Path,
    image: np.ndarray,
    conf: float,
    iou: float,
    imgsz: int | None,
) -> list[YoloDetection]:
    cached = _load_torch_model(weights_path)
    target_imgsz = imgsz or cached.imgsz or 640
    device = _resolve_yolo_device()
    use_half = _should_use_half(device)

    with cached.lock, torch.inference_mode():
        results = cached.model.predict(
            source=image,
            conf=conf,
            iou=iou,
            imgsz=target_imgsz,
            device=device,
            half=use_half,
            verbose=False,
        )

    if not results:
        return []

    return _build_detections(results[0], class_names=cached.class_names)


def _build_detections(result, *, class_names: dict[int, str]) -> list[YoloDetection]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return []

    xyxy_values = boxes.xyxy.cpu().tolist()
    conf_values = boxes.conf.cpu().tolist()
    class_values = boxes.cls.cpu().tolist()
    polygons = _extract_polygons(result)

    detections: list[YoloDetection] = []
    for idx, xyxy in enumerate(xyxy_values):
        x1, y1, x2, y2 = (float(value) for value in xyxy)
        class_idx = int(class_values[idx])
        polygon = polygons[idx] if idx < len(polygons) else None

        detections.append(
            YoloDetection(
                class_key=class_names.get(class_idx, str(class_idx)),
                confidence=float(conf_values[idx]),
                bbox={
                    "x": x1,
                    "y": y1,
                    "w": max(0.0, x2 - x1),
                    "h": max(0.0, y2 - y1),
                },
                polygon=polygon,
            )
        )

    return detections


def _extract_polygons(result) -> list[list[list[float]] | None]:
    masks = getattr(result, "masks", None)
    if masks is None or not getattr(masks, "xy", None):
        boxes = getattr(result, "boxes", None)
        count = len(boxes) if boxes is not None else 0
        return [None] * count

    polygons: list[list[list[float]] | None] = []
    for points in masks.xy:
        if points is None or len(points) < 3:
            polygons.append(None)
            continue

        polygon = [[float(x), float(y)] for x, y in points.tolist()]
        polygons.append(polygon if len(polygon) >= 3 else None)

    return polygons


def _extract_class_names(model: Any) -> dict[int, str]:
    names = getattr(model, "names", None) or getattr(
        getattr(model, "model", None), "names", None
    )
    if isinstance(names, dict):
        return {int(idx): str(name) for idx, name in names.items()}
    if isinstance(names, list):
        return {idx: str(name) for idx, name in enumerate(names)}
    return {}


def _extract_default_imgsz(model: Any) -> int | None:
    candidates = (
        getattr(model, "overrides", {}).get("imgsz"),
        getattr(getattr(model, "model", None), "args", {}).get("imgsz"),
    )
    for candidate in candidates:
        if isinstance(candidate, int) and candidate > 0:
            return candidate
        if isinstance(candidate, (list, tuple)) and candidate:
            first = candidate[0]
            if isinstance(first, int) and first > 0:
                return first
    return None


def _model_device(model: Any) -> str | None:
    torch_model = getattr(model, "model", None)
    if torch_model is None:
        return None

    try:
        return str(next(torch_model.parameters()).device)
    except (AttributeError, StopIteration, TypeError):
        return None


def _resolve_yolo_device() -> str:
    requested = settings.YOLO_DEVICE

    if requested == "cuda":
        if torch.cuda.is_available():
            return "0"

        log_event(
            logger,
            "warning",
            "inspection.yolo.device.fallback_cpu",
            requested_device=requested,
            actual_device="cpu",
        )
        return "cpu"

    if requested == "cpu":
        return "cpu"

    if torch.cuda.is_available():
        return "0"

    return "cpu"


def _should_use_half(device: str) -> bool:
    if not settings.YOLO_HALF:
        return False

    if device == "cpu":
        return False

    return torch.cuda.is_available()


def clear_model_cache() -> None:
    with _torch_cache_lock:
        cached_models = [cached_model for _, cached_model in _torch_cache.values()]
        _torch_cache.clear()

    for cached_model in cached_models:
        _dispose_yolo_model(cached_model)
    cached_models.clear()
    _release_torch_memory()


def _dispose_yolo_model(cached_model: _TorchYoloModel) -> None:
    torch_model = getattr(cached_model.model, "model", None)
    if torch_model is not None and hasattr(torch_model, "cpu"):
        with suppress(Exception):
            torch_model.cpu()

    del cached_model


def _release_torch_memory() -> None:
    gc.collect()

    if torch.cuda.is_available():
        with suppress(Exception):
            torch.cuda.empty_cache()
