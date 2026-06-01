from __future__ import annotations

import asyncio
import time
from pathlib import Path
from uuid import UUID

import cv2 as cv
import numpy as np
from app.exception import InternalServerError, ValidationError
from app.observability import elapsed_ms, log_event
from infra.storage.file_storage import resolve_storage_path
from modules.core.standards import crud as standards_crud
from sqlalchemy.ext.asyncio import AsyncSession
import structlog

from .predictor import predict_masks_from_clicks, prewarm_embeddings
from .schemas import SamClickRequest, SamClickResponse

logger = structlog.get_logger(__name__)


async def predict_from_clicks(
    db: AsyncSession,
    data: SamClickRequest,
) -> SamClickResponse:
    started_at = time.perf_counter()
    image = await standards_crud.get_image(db, image_id=data.image_id)
    image_path = _resolve_image_path(image.image_path)
    log_event(
        logger,
        "info",
        "sam.inference.started",
        image_id=data.image_id,
        points_count=len(data.points),
        image_path=image.image_path,
    )

    try:
        result = await asyncio.to_thread(
            _segment_from_clicks,
            image_id=str(data.image_id),
            image_path=str(image_path),
            point_coords=[[point.x, point.y] for point in data.points],
            point_labels=[point.label for point in data.points],
        )
    except ValueError as exc:
        log_event(
            logger,
            "warning",
            "sam.inference.failed",
            image_id=data.image_id,
            duration_ms=elapsed_ms(started_at),
            error_type=type(exc).__name__,
            reason=str(exc),
        )
        raise ValidationError(str(exc)) from exc
    except RuntimeError as exc:
        log_event(
            logger,
            "error",
            "sam.inference.failed",
            image_id=data.image_id,
            duration_ms=elapsed_ms(started_at),
            error_type=type(exc).__name__,
            exception=exc,
        )
        raise InternalServerError(str(exc)) from exc

    log_event(
        logger,
        "info",
        "sam.inference.finished",
        image_id=data.image_id,
        duration_ms=elapsed_ms(started_at),
        points_count=len(result["points"]),
        score=result["score"],
    )
    return SamClickResponse.model_validate(result)


async def warmup_image(
    db: AsyncSession,
    image_id: UUID,
) -> None:
    started_at = time.perf_counter()
    image = await standards_crud.get_image(db, image_id=image_id)
    image_path = _resolve_image_path(image.image_path)

    try:
        await asyncio.to_thread(
            prewarm_embeddings,
            image_id=str(image_id),
            image_path=str(image_path),
        )
    except ValueError as exc:
        log_event(
            logger,
            "warning",
            "sam.inference.failed",
            image_id=image_id,
            duration_ms=elapsed_ms(started_at),
            error_type=type(exc).__name__,
            reason=str(exc),
        )
        raise ValidationError(str(exc)) from exc
    except RuntimeError as exc:
        log_event(
            logger,
            "error",
            "sam.inference.failed",
            image_id=image_id,
            duration_ms=elapsed_ms(started_at),
            error_type=type(exc).__name__,
            exception=exc,
        )
        raise InternalServerError(str(exc)) from exc

    log_event(
        logger,
        "info",
        "sam.inference.finished",
        image_id=image_id,
        duration_ms=elapsed_ms(started_at),
        action="warmup",
    )


def _resolve_image_path(image_path: str) -> Path:
    resolved_path = resolve_storage_path(image_path)
    if not resolved_path.is_file():
        raise ValidationError("Файл изображения не найден")
    return resolved_path


def _segment_from_clicks(
    *,
    image_id: str,
    image_path: str,
    point_coords: list[list[float]],
    point_labels: list[int],
) -> dict[str, object]:
    if not point_coords:
        raise ValueError("Нужен хотя бы один клик")

    masks, scores = predict_masks_from_clicks(
        image_id=image_id,
        image_path=image_path,
        point_coords=point_coords,
        point_labels=point_labels,
    )

    best_mask, best_score = _pick_best_mask_and_score(masks, scores)
    contour = _extract_largest_contour(best_mask)
    polygon = _contour_to_polygon(contour)

    return {
        "points": polygon,
        "score": best_score,
    }


def _pick_best_mask_and_score(
    masks: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, float]:
    best_idx = int(np.argmax(scores))
    best_mask = np.asarray(masks[best_idx])
    best_score = float(scores[best_idx])

    return best_mask, best_score


def _extract_largest_contour(
    mask: np.ndarray,
) -> np.ndarray:
    mask_u8 = mask.astype(np.uint8) * 255
    contours, _ = cv.findContours(mask_u8, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    if not contours:
        raise ValueError("SAM2 не смог выделить объект")

    return max(contours, key=cv.contourArea)


def _contour_to_polygon(
    contour: np.ndarray,
) -> list[list[float]]:
    perimeter = cv.arcLength(contour, closed=True)
    epsilon = max(1.0, perimeter * 0.002)

    approx = cv.approxPolyDP(contour, epsilon=epsilon, closed=True).reshape(-1, 2)
    if len(approx) < 3:
        approx = contour.reshape(-1, 2)

    return approx.astype(float).tolist()
