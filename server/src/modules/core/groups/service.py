from __future__ import annotations

import shutil
from contextlib import suppress
from uuid import UUID

from infra.storage.file_storage import delete_storage_file, resolve_storage_path
from modules.core.segments import crud as segment_crud
from modules.core.segments.models import SegmentClass, SegmentClassGroup
from modules.core.standards.crud import list_standard_image_paths
from modules.core.standards.models import Standard
from modules.core.standards.reference_storage import delete_features
from sqlalchemy.ext.asyncio import AsyncSession

from . import crud
from .crud import GroupStatsData
from .schemas import (
    GroupActiveModelResponse,
    GroupCreate,
    GroupDetailResponse,
    GroupListItemResponse,
    GroupMutationResponse,
    GroupSegmentClassCategoryResponse,
    GroupSegmentClassResponse,
    GroupStandardResponse,
    GroupStatsResponse,
    GroupUpdate,
)

async def get_groups(
    db: AsyncSession,
) -> list[GroupListItemResponse]:
    groups = await crud.list_groups(db)
    if not groups:
        return []

    stats_map = await crud.get_group_stats_map(
        db,
        group_ids=[group.id for group in groups],
    )

    return [
        GroupListItemResponse(
            id=group.id,
            name=group.name,
            description=group.description,
            created_at=group.created_at,
            stats=_build_group_stats_response(
                stats_map.get(group.id, GroupStatsData())
            ),
        )
        for group in groups
    ]


async def get_group(
    db: AsyncSession,
    group_id: UUID,
) -> GroupDetailResponse:
    group = await crud.get_group(db, group_id=group_id)
    stats_map = await crud.get_group_stats_map(db, group_ids=[group_id])
    standards = await crud.list_group_standards(db, group_id=group_id)
    active_model = await crud.get_active_group_model(db, group_id=group_id)
    categories, ungrouped_classes = await segment_crud.get_group_tree(
        db,
        group_id=group_id,
    )

    sorted_categories = sorted(categories, key=lambda item: item.name.lower())
    sorted_ungrouped = sorted(ungrouped_classes, key=lambda item: item.name.lower())

    return GroupDetailResponse(
        id=group.id,
        name=group.name,
        description=group.description,
        created_at=group.created_at,
        stats=_build_group_stats_response(stats_map.get(group_id, GroupStatsData())),
        standards=[_build_group_standard_response(item) for item in standards],
        active_model=(
            GroupActiveModelResponse.model_validate(active_model)
            if active_model is not None
            else None
        ),
        segment_class_categories=[
            _build_group_segment_class_category_response(item)
            for item in sorted_categories
        ],
        ungrouped_segment_classes=[
            _build_group_segment_class_response(item) for item in sorted_ungrouped
        ],
    )


async def create_group(
    db: AsyncSession,
    data: GroupCreate,
) -> GroupMutationResponse:
    await crud.ensure_group_name_unique(db, name=data.name)
    group = await crud.create_group(
        db,
        name=data.name,
        description=data.description,
    )
    return GroupMutationResponse.model_validate(group)


async def update_group(
    db: AsyncSession,
    group_id: UUID,
    data: GroupUpdate,
) -> GroupMutationResponse:
    group = await crud.get_group(db, group_id=group_id)

    if data.name is not None and data.name != group.name:
        await crud.ensure_group_name_unique(
            db,
            name=data.name,
            exclude_id=group_id,
        )

    for key, value in data.model_dump(exclude_unset=True).items():
        setattr(group, key, value)

    group = await crud.save_group(db, group=group)
    return GroupMutationResponse.model_validate(group)


async def delete_group(
    db: AsyncSession,
    group_id: UUID,
) -> None:
    group = await crud.get_group(db, group_id=group_id)
    standards = await crud.list_group_standards(db, group_id=group_id)

    image_paths: list[str] = []
    features_paths: list[str] = []

    for standard in standards:
        path_pairs = await list_standard_image_paths(db, standard_id=standard.id)
        for image_path, features_path in path_pairs:
            image_paths.append(image_path)
            if features_path:
                features_paths.append(features_path)
    await crud.delete_group(db, group=group)

    for image_path in image_paths:
        with suppress(Exception):
            delete_storage_file(image_path)

    for features_path in features_paths:
        with suppress(Exception):
            delete_features(features_path)

    shutil.rmtree(
        resolve_storage_path(f"models/{group_id}"),
        ignore_errors=True,
    )


def _build_group_stats_response(
    stats: GroupStatsData,
) -> GroupStatsResponse:
    return GroupStatsResponse(
        standards_count=stats.standards_count,
        images_count=stats.images_count,
        annotated_images_count=stats.annotated_images_count,
        polygons_count=stats.polygons_count,
        segment_class_groups_count=stats.segment_class_groups_count,
        segment_classes_count=stats.segment_classes_count,
        models_count=stats.models_count,
        inspections_count=stats.inspections_count,
    )


def _build_group_standard_response(
    standard: Standard,
) -> GroupStandardResponse:
    reference_image = next(
        (image for image in standard.images if image.is_reference),
        None,
    )
    annotated_images_count = sum(
        1
        for image in standard.images
        if any(annotation.points for annotation in image.annotations)
    )

    return GroupStandardResponse(
        id=standard.id,
        group_id=standard.group_id,
        name=standard.name,
        angle=standard.angle,
        is_active=standard.is_active,
        created_at=standard.created_at,
        reference_path=reference_image.image_path if reference_image else None,
        images_count=len(standard.images),
        annotated_images_count=annotated_images_count,
    )


def _build_group_segment_class_response(
    segment_class: SegmentClass,
) -> GroupSegmentClassResponse:
    return GroupSegmentClassResponse(
        id=segment_class.id,
        group_id=segment_class.group_id,
        class_group_id=segment_class.class_group_id,
        name=segment_class.name,
        hue=segment_class.hue,
    )


def _build_group_segment_class_category_response(
    category: SegmentClassGroup,
) -> GroupSegmentClassCategoryResponse:
    items = sorted(category.segment_classes, key=lambda item: item.name.lower())

    return GroupSegmentClassCategoryResponse(
        id=category.id,
        group_id=category.group_id,
        name=category.name,
        segment_classes=[_build_group_segment_class_response(item) for item in items],
    )
