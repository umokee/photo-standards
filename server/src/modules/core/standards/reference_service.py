from __future__ import annotations

import time
from datetime import UTC, datetime
from uuid import UUID

import structlog
from app.config import settings
from app.exception import NotFoundError
from app.observability import elapsed_ms, log_event
from infra.storage.file_storage import resolve_storage_path
from modules.core.standards.models import StandardImage
from modules.core.standards.reference_features import compute_features, load_image
from modules.core.standards.reference_constants import (
    SUPERPOINT_OFFLINE_MAX_KEYPOINTS,
    SUPERPOINT_OFFLINE_MAX_SIDE,
)
from modules.core.standards.reference_storage import (
    features_file_is_compatible,
    features_rel_path,
    save_features,
)
from sqlalchemy.ext.asyncio import AsyncSession

logger = structlog.get_logger(__name__)


async def compute_and_save_features(
    db: AsyncSession,
    *,
    image_id: UUID,
) -> None:
    started_at = time.perf_counter()
    image = await db.get(StandardImage, image_id)
    if image is None:
        raise NotFoundError("Фото", image_id)

    image_absolute = resolve_storage_path(image.image_path)
    image_array = load_image(image_absolute)
    features = compute_features(
        image_array,
        max_side=SUPERPOINT_OFFLINE_MAX_SIDE,
        max_keypoints=SUPERPOINT_OFFLINE_MAX_KEYPOINTS,
    )

    rel_path = features_rel_path(
        standard_id=image.standard_id,
        image_id=image.id,
    )
    save_features(features, rel_path)

    image.features_path = rel_path
    image.features_keypoint_count = features.count
    image.features_computed_at = datetime.now(UTC).replace(tzinfo=None)
    await db.commit()

    log_event(
        logger,
        "info",
        "standard.features.computed",
        standard_id=image.standard_id,
        image_id=image_id,
        image_path=image.image_path,
        features_path=rel_path,
        keypoint_count=features.count,
        alignment_backend=settings.ALIGNMENT_BACKEND,
        alignment_device=settings.ALIGNMENT_DEVICE,
        duration_ms=elapsed_ms(started_at),
    )


def features_are_ready(image: StandardImage) -> bool:
    if not image.features_path or image.features_computed_at is None:
        return False
    return features_file_is_compatible(image.features_path)
