from __future__ import annotations

import os
import threading
import time
from contextlib import nullcontext
from typing import Any

import cv2 as cv
import numpy as np
import torch
from app.config import settings
from app.observability import log_event

from .cache import (
    cache_get,
    cache_put,
    restore_predictor_state,
    snapshot_predictor_state,
)
import structlog

logger = structlog.get_logger(__name__)


_predictor_lock = threading.Lock()
_predictor: Any = None


def prewarm_embeddings(
    *,
    image_id: str,
    image_path: str,
) -> None:
    predictor = _get_or_create_predictor()
    with _predictor_lock:
        _ensure_image_embeddings(
            predictor,
            image_id=image_id,
            image_path=image_path,
        )


def predict_masks_from_clicks(
    *,
    image_id: str,
    image_path: str,
    point_coords: list[list[float]],
    point_labels: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    coords = np.asarray(point_coords, dtype=np.float32)
    labels = np.asarray(point_labels, dtype=np.int32)

    predictor = _get_or_create_predictor()

    with _predictor_lock:
        _ensure_image_embeddings(
            predictor,
            image_id=image_id,
            image_path=image_path,
        )

        inference_ctx, autocast_ctx = _build_inference_contexts()
        with inference_ctx, autocast_ctx:
            masks, scores, _ = predictor.predict(
                point_coords=coords,
                point_labels=labels,
                multimask_output=len(point_coords) == 1,
            )

    return np.asarray(masks), np.asarray(scores).reshape(-1)


def _get_or_create_predictor() -> Any:
    global _predictor

    if _predictor is None:
        with _predictor_lock:
            if _predictor is None:
                _predictor = _build_predictor()

    return _predictor


def _resolve_device() -> str:
    if settings.SAM2_DEVICE == "cuda" and not torch.cuda.is_available():
        return "cpu"
    return settings.SAM2_DEVICE


def _build_inference_contexts() -> tuple[Any, Any]:
    if _resolve_device() == "cuda":
        return torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16)

    return torch.inference_mode(), nullcontext()


def _build_predictor() -> Any:
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor
    except ModuleNotFoundError as exc:
        raise RuntimeError("SAM2 не установлен") from exc

    _configure_cpu_threads()

    model = build_sam2(
        config_file=settings.SAM2_MODEL_CFG,
        ckpt_path=settings.SAM2_CHECKPOINT,
        device=_resolve_device(),
    )
    model.eval()

    return SAM2ImagePredictor(model)


def _configure_cpu_threads() -> None:
    if _resolve_device() != "cpu":
        return

    try:
        usable = len(os.sched_getaffinity(0))
    except AttributeError:
        usable = os.cpu_count() or 4

    torch.set_num_threads(max(1, usable - 1))


def _ensure_image_embeddings(
    predictor: Any,
    *,
    image_id: str,
    image_path: str,
) -> None:
    cached_state = cache_get(image_id)
    if cached_state is not None:
        restore_predictor_state(predictor, cached_state)
        return

    started_at = time.perf_counter()

    image_bgr = cv.imread(image_path, cv.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError("Не удалось загрузить изображение")

    image_rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    loaded_at = time.perf_counter()

    inference_ctx, autocast_ctx = _build_inference_contexts()
    with inference_ctx, autocast_ctx:
        predictor.set_image(image_rgb)
    encoded_at = time.perf_counter()

    log_event(
        logger,
        "info",
        "sam.embeddings.computed",
        image_id=image_id,
        io_ms=round((loaded_at - started_at) * 1000, 1),
        encoder_ms=round((encoded_at - loaded_at) * 1000, 1),
        image_shape=list(image_rgb.shape),
        device=_resolve_device(),
    )

    cache_put(image_id, snapshot_predictor_state(predictor))
