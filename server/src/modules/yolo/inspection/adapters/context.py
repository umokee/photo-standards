from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.exception import ValidationError
from modules.core.segments.models import SegmentClass
from modules.core.standards.models import Standard, StandardImage
from modules.core.standards.reference_features import ImageFeatures
from modules.core.standards.reference_service import (
    compute_and_save_features,
    features_are_ready,
)
from modules.core.standards.reference_storage import load_features
from modules.yolo.training.adapters import repository as training_repository
from modules.yolo.training.adapters import storage as training_storage
from modules.yolo.training.models import MlModel
from sqlalchemy.ext.asyncio import AsyncSession

from .payloads import build_native_to_internal_key_map
from .repository import get_standard_for_inspection


@dataclass(slots=True)
class InspectionContext:
    standard: Standard
    model: MlModel
    reference_image: StandardImage
    reference_features: ImageFeatures
    selected_classes: list[SegmentClass]
    _native_to_internal_cache: dict[str, str] | None = field(default=None, repr=False)

    @property
    def native_to_internal(self) -> dict[str, str]:
        if self._native_to_internal_cache is None:
            self._native_to_internal_cache = build_native_to_internal_key_map(
                self.model
            )
        return self._native_to_internal_cache


async def load_inspection_context(
    db: AsyncSession,
    *,
    standard_id: UUID,
    selected_segment_class_ids: list[UUID],
) -> InspectionContext:
    standard = await get_standard_for_inspection(db, standard_id=standard_id)

    reference_image = next((img for img in standard.images if img.is_reference), None)
    if reference_image is None:
        raise ValidationError("У эталона нет reference-фото")
    if not reference_image.annotations:
        raise ValidationError("Reference-фото не содержит аннотаций")
    if not features_are_ready(reference_image):
        await compute_and_save_features(db, image_id=reference_image.id)

    reference_features = load_features(reference_image.features_path)

    selected_set = set(selected_segment_class_ids)
    if not selected_set:
        raise ValidationError("Не выбраны классы для проверки")

    selected_classes = [
        segment_class
        for segment_class in standard.group.segment_classes
        if segment_class.id in selected_set
    ]
    if not selected_classes:
        raise ValidationError("Не выбраны классы для проверки")
    if len({item.id for item in selected_classes}) != len(selected_set):
        raise ValidationError("Часть выбранных классов не принадлежит группе эталона")

    model = await training_repository.get_active_model(
        db,
        group_id=standard.group_id,
    )
    training_storage.ensure_model_weights_ready(model)

    return InspectionContext(
        standard=standard,
        model=model,
        reference_image=reference_image,
        reference_features=reference_features,
        selected_classes=selected_classes,
    )
