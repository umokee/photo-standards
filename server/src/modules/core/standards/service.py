from __future__ import annotations

from uuid import UUID, uuid4

import structlog
from fastapi import UploadFile
from app.observability import log_event
from infra.storage import file_storage
from modules.core.schemas import AnnotationPoints
from modules.core.segments import crud as segment_crud
from modules.core.segments.models import SegmentClass, SegmentClassGroup
from modules.core.standards.reference_service import (
    compute_and_save_features,
    features_are_ready,
)
from modules.core.standards.reference_storage import delete_features
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud
from .models import Standard, StandardImage
from .schemas import (
    StandardCreate,
    StandardDetailResponse,
    StandardImageDetailResponse,
    StandardImageResponse,
    StandardMutationResponse,
    StandardSegmentClassCategoryResponse,
    StandardSegmentClassResponse,
    StandardSegmentClassWithPointsResponse,
    StandardStatsResponse,
    StandardUpdate,
)

logger = structlog.get_logger(__name__)


async def get_standard(
    db: AsyncSession,
    standard_id: UUID,
) -> StandardDetailResponse:
    standard = await crud.get_standard_with_relations(db, standard_id=standard_id)
    return _build_standard_detail_response(standard)


async def create_standard(
    db: AsyncSession,
    data: StandardCreate,
) -> StandardMutationResponse:
    standard = await crud.create_standard(
        db,
        group_id=data.group_id,
        name=data.name,
        angle=data.angle,
    )
    return _build_standard_mutation_response(standard)


async def update_standard(
    db: AsyncSession,
    standard_id: UUID,
    data: StandardUpdate,
) -> StandardMutationResponse:
    standard = await crud.get_standard(db, standard_id=standard_id)

    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(standard, key, value)

    await db.commit()
    await db.refresh(standard)
    return _build_standard_mutation_response(standard)


async def delete_standard(
    db: AsyncSession,
    standard_id: UUID,
) -> None:
    standard = await crud.get_standard(db, standard_id=standard_id)
    path_pairs = await crud.list_standard_image_paths(db, standard_id=standard_id)

    await db.delete(standard)
    await db.commit()

    for image_path, features_path in path_pairs:
        await file_storage.delete_file(image_path)
        if features_path:
            delete_features(features_path)


async def upload_images(
    db: AsyncSession,
    standard_id: UUID,
    images: list[UploadFile],
) -> list[StandardImageResponse]:
    standard = await crud.get_standard(db, standard_id=standard_id)

    pending: list[tuple[UUID, str]] = []
    try:
        for upload in images:
            image_id = uuid4()
            image_path = await file_storage.save_upload(
                upload,
                f"standards/{standard.id}",
                str(image_id),
            )
            pending.append((image_id, image_path))
    except Exception:
        for _, path in pending:
            await file_storage.delete_file(path)
        raise

    try:
        await crud.create_standard_images(
            db,
            standard_id=standard_id,
            items=pending,
        )
    except Exception:
        await db.rollback()
        for _, path in pending:
            await file_storage.delete_file(path)
        raise

    items = await crud.list_standard_images(
        db,
        image_ids=[image_id for image_id, _ in pending],
    )
    items_by_id = {item.id: item for item in items}

    return [
        _build_standard_image_response(items_by_id[image_id])
        for image_id, _ in pending
        if image_id in items_by_id
    ]


async def set_reference(
    db: AsyncSession,
    image_id: UUID,
) -> StandardImageResponse:
    image = await crud.get_image(db, image_id=image_id)

    if not image.is_reference and not features_are_ready(image):
        await compute_and_save_features(db, image_id=image_id)

    image = await crud.set_standard_reference(db, image=image)

    image = await crud.get_image_with_annotations(db, image_id=image_id)
    log_event(
        logger,
        "info",
        "standard.reference.toggled",
        standard_id=image.standard_id,
        image_id=image.id,
        image_path=image.image_path,
        is_reference=image.is_reference,
        annotation_count=sum(1 for annotation in image.annotations if annotation.points),
    )
    return _build_standard_image_response(image)


async def get_image(
    db: AsyncSession,
    image_id: UUID,
) -> StandardImageDetailResponse:
    image = await crud.get_image_with_annotations(db, image_id=image_id)
    categories, ungrouped_classes = await segment_crud.get_group_tree(
        db,
        group_id=image.standard.group_id,
    )
    segment_classes = sorted(
        [
            *[
                segment_class
                for category in categories
                for segment_class in category.segment_classes
            ],
            *ungrouped_classes,
        ],
        key=lambda item: item.name.lower(),
    )

    annotation_points_by_class_id = {
        annotation.segment_class_id: annotation.points
        for annotation in image.annotations
        if annotation.points
    }

    return _build_standard_image_detail_response(
        image,
        segment_classes,
        annotation_points_by_class_id,
    )


async def delete_image(
    db: AsyncSession,
    image_id: UUID,
) -> None:
    image = await crud.get_image(db, image_id=image_id)

    image_path = image.image_path
    features_path = image.features_path

    await crud.delete_standard_image(db, image=image)

    await file_storage.delete_file(image_path)
    if features_path:
        delete_features(features_path)


def _build_standard_mutation_response(
    standard: Standard,
) -> StandardMutationResponse:
    return StandardMutationResponse.model_validate(standard)


def _build_standard_image_response(
    image: StandardImage,
) -> StandardImageResponse:
    annotation_count = sum(1 for annotation in image.annotations if annotation.points)
    response = StandardImageResponse.model_validate(image)
    return response.model_copy(
        update={
            "annotation_count": annotation_count,
        }
    )


def _build_standard_segment_class_response(
    segment_class: SegmentClass,
) -> StandardSegmentClassResponse:
    return StandardSegmentClassResponse(
        id=segment_class.id,
        group_id=segment_class.group_id,
        class_group_id=segment_class.class_group_id,
        name=segment_class.name,
        hue=segment_class.hue,
    )


def _build_standard_segment_class_category_response(
    category: SegmentClassGroup,
    *,
    allowed_class_ids: set[UUID] | None = None,
) -> StandardSegmentClassCategoryResponse:
    items = sorted(category.segment_classes, key=lambda item: item.name.lower())
    if allowed_class_ids is not None:
        items = [item for item in items if item.id in allowed_class_ids]

    return StandardSegmentClassCategoryResponse(
        id=category.id,
        group_id=category.group_id,
        name=category.name,
        segment_classes=[
            _build_standard_segment_class_response(item) for item in items
        ],
    )


def _build_standard_stats_response(
    standard: Standard,
) -> StandardStatsResponse:
    reference_image = next(
        (image for image in standard.images if image.is_reference),
        None,
    )
    annotated_images_count = sum(
        1
        for image in standard.images
        if any(annotation.points for annotation in image.annotations)
    )

    return StandardStatsResponse(
        images_count=len(standard.images),
        annotated_images_count=annotated_images_count,
        unannotated_images_count=len(standard.images) - annotated_images_count,
        segment_classes_count=len(standard.group.segment_classes),
        segment_class_categories_count=len(standard.group.segment_class_groups),
        reference_image_id=reference_image.id if reference_image else None,
        reference_path=reference_image.image_path if reference_image else None,
    )


def _collect_used_segment_class_ids(standard: Standard) -> set[UUID]:
    return {
        annotation.segment_class_id
        for image in standard.images
        for annotation in image.annotations
        if annotation.points
    }


def _build_standard_detail_response(
    standard: Standard,
) -> StandardDetailResponse:
    images = sorted(
        standard.images,
        key=lambda image: (not image.is_reference, image.created_at),
    )
    categories = sorted(
        standard.group.segment_class_groups,
        key=lambda item: item.name.lower(),
    )
    ungrouped_classes = sorted(
        [
            item
            for item in standard.group.segment_classes
            if item.class_group_id is None
        ],
        key=lambda item: item.name.lower(),
    )
    used_segment_class_ids = _collect_used_segment_class_ids(standard)
    used_categories = [
        category
        for category in categories
        if any(item.id in used_segment_class_ids for item in category.segment_classes)
    ]
    used_ungrouped_classes = [
        item for item in ungrouped_classes if item.id in used_segment_class_ids
    ]

    return StandardDetailResponse(
        id=standard.id,
        group_id=standard.group_id,
        name=standard.name,
        angle=standard.angle,
        is_active=standard.is_active,
        created_at=standard.created_at,
        stats=_build_standard_stats_response(standard),
        images=[_build_standard_image_response(item) for item in images],
        segment_class_categories=[
            _build_standard_segment_class_category_response(item) for item in categories
        ],
        ungrouped_segment_classes=[
            _build_standard_segment_class_response(item) for item in ungrouped_classes
        ],
        used_segment_class_categories=[
            _build_standard_segment_class_category_response(
                item,
                allowed_class_ids=used_segment_class_ids,
            )
            for item in used_categories
        ],
        used_ungrouped_segment_classes=[
            _build_standard_segment_class_response(item) for item in used_ungrouped_classes
        ],
    )


def _build_standard_image_detail_response(
    image: StandardImage,
    segment_classes: list[SegmentClass],
    points_by_class_id: dict[UUID, AnnotationPoints],
) -> StandardImageDetailResponse:
    annotation_count = sum(1 for annotation in image.annotations if annotation.points)

    return StandardImageDetailResponse(
        id=image.id,
        standard_id=image.standard_id,
        image_path=image.image_path,
        is_reference=image.is_reference,
        annotation_count=annotation_count,
        created_at=image.created_at,
        segment_classes=[
            StandardSegmentClassWithPointsResponse(
                id=item.id,
                group_id=item.group_id,
                class_group_id=item.class_group_id,
                name=item.name,
                hue=item.hue,
                points=points_by_class_id.get(item.id, []),
            )
            for item in segment_classes
        ],
    )
