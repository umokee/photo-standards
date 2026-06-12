from __future__ import annotations

from dataclasses import dataclass, field
from uuid import UUID

from app.config import settings
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
class ReferenceView:
    image: StandardImage
    features: ImageFeatures
    is_primary: bool = False


@dataclass(slots=True)
class InspectionContext:
    standard: Standard
    model: MlModel
    reference_image: StandardImage
    reference_features: ImageFeatures
    selected_classes: list[SegmentClass]
    reference_views: list[ReferenceView] = field(default_factory=list)
    _native_to_internal_cache: dict[str, str] | None = field(default=None, repr=False)

    @property
    def native_to_internal(self) -> dict[str, str]:
        if self._native_to_internal_cache is None:
            self._native_to_internal_cache = build_native_to_internal_key_map(
                self.model
            )
        return self._native_to_internal_cache

    def with_reference_view(self, view: ReferenceView) -> InspectionContext:
        return InspectionContext(
            standard=self.standard,
            model=self.model,
            reference_image=view.image,
            reference_features=view.features,
            selected_classes=self.selected_classes,
            reference_views=self.reference_views,
            _native_to_internal_cache=self._native_to_internal_cache,
        )


async def load_inspection_context(
    db: AsyncSession,
    *,
    standard_id: UUID,
    selected_segment_class_ids: list[UUID],
) -> InspectionContext:
    standard = await get_standard_for_inspection(db, standard_id=standard_id)

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

    candidate_images = _reference_candidate_images(
        standard,
        selected_segment_class_ids=selected_set,
    )
    if not candidate_images:
        raise ValidationError("В эталоне нет размеченных фото, включённых в проверку")

    reference_views: list[ReferenceView] = []
    for image in candidate_images:
        if not features_are_ready(image):
            await compute_and_save_features(db, image_id=image.id)
            # compute_and_save_features commits and updates DB. Reload the already
            # attached object fields for the feature path/count used below.
            await db.refresh(image)

        reference_views.append(
            ReferenceView(
                image=image,
                features=load_features(image.features_path),
                is_primary=image.is_reference,
            )
        )

    reference_view = reference_views[0]

    model = await training_repository.get_active_model(
        db,
        group_id=standard.group_id,
    )
    training_storage.ensure_model_weights_ready(model)

    return InspectionContext(
        standard=standard,
        model=model,
        reference_image=reference_view.image,
        reference_features=reference_view.features,
        selected_classes=selected_classes,
        reference_views=reference_views,
    )


def _reference_candidate_images(
    standard: Standard,
    *,
    selected_segment_class_ids: set[UUID],
) -> list[StandardImage]:
    candidates: list[StandardImage] = []

    for image in standard.images:
        if not image.is_reference:
            continue

        annotated_class_ids = {
            annotation.segment_class_id
            for annotation in image.annotations
            if annotation.points and annotation.segment_class_id in selected_segment_class_ids
        }
        if selected_segment_class_ids.issubset(annotated_class_ids):
            candidates.append(image)

    candidates.sort(key=lambda image: (image.created_at, str(image.id)))

    if not settings.INSPECTION_MULTI_REFERENCE_ENABLED:
        return candidates[:1]

    max_candidates = max(1, int(settings.INSPECTION_MULTI_REFERENCE_MAX_CANDIDATES))
    return candidates[:max_candidates]
