from __future__ import annotations

from uuid import UUID

from app.exception import ConflictError, NotFoundError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import SegmentAnnotation, SegmentClass, SegmentClassGroup


async def get_segment_class(
    db: AsyncSession,
    *,
    segment_class_id: UUID,
) -> SegmentClass:
    segment_class = await db.get(SegmentClass, segment_class_id)
    if segment_class is None:
        raise NotFoundError("Класс", segment_class_id)
    return segment_class


async def get_segment_class_or_none(
    db: AsyncSession,
    *,
    segment_class_id: UUID,
) -> SegmentClass | None:
    return await db.get(SegmentClass, segment_class_id)


async def get_category(
    db: AsyncSession,
    *,
    category_id: UUID,
) -> SegmentClassGroup:
    category = await db.get(SegmentClassGroup, category_id)
    if category is None:
        raise NotFoundError("Категория", category_id)
    return category


async def get_category_or_none(
    db: AsyncSession,
    *,
    category_id: UUID,
) -> SegmentClassGroup | None:
    return await db.get(SegmentClassGroup, category_id)


async def get_group_tree(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> tuple[list[SegmentClassGroup], list[SegmentClass]]:
    category_result = await db.execute(
        select(SegmentClassGroup)
        .options(selectinload(SegmentClassGroup.segment_classes))
        .where(SegmentClassGroup.group_id == group_id)
    )
    categories = list(category_result.scalars().all())

    ungrouped_result = await db.execute(
        select(SegmentClass)
        .where(
            SegmentClass.group_id == group_id,
            SegmentClass.class_group_id.is_(None),
        )
        .order_by(SegmentClass.name.asc())
    )
    ungrouped_classes = list(ungrouped_result.scalars().all())

    return categories, ungrouped_classes


async def get_annotation(
    db: AsyncSession,
    *,
    segment_class_id: UUID,
    image_id: UUID,
) -> SegmentAnnotation | None:
    result = await db.execute(
        select(SegmentAnnotation).where(
            SegmentAnnotation.segment_class_id == segment_class_id,
            SegmentAnnotation.image_id == image_id,
        )
    )
    return result.scalar_one_or_none()


async def ensure_category_name_unique(
    db: AsyncSession,
    *,
    group_id: UUID,
    name: str,
    exclude_category_id: UUID | None = None,
) -> None:
    query = select(SegmentClassGroup.id).where(
        SegmentClassGroup.group_id == group_id,
        SegmentClassGroup.name == name,
    )
    if exclude_category_id is not None:
        query = query.where(SegmentClassGroup.id != exclude_category_id)

    existing = await db.scalar(query)
    if existing is not None:
        raise ConflictError(
            "Категория уже существует",
            details={
                "entity": "segment_class_group",
                "field": "name",
                "value": name,
                "group_id": str(group_id),
            },
        )


async def ensure_segment_class_name_unique(
    db: AsyncSession,
    *,
    group_id: UUID,
    name: str,
    exclude_segment_class_id: UUID | None = None,
) -> None:
    query = select(SegmentClass.id).where(
        SegmentClass.group_id == group_id,
        SegmentClass.name == name,
    )
    if exclude_segment_class_id is not None:
        query = query.where(SegmentClass.id != exclude_segment_class_id)

    existing = await db.scalar(query)
    if existing is not None:
        raise ConflictError(
            "Класс уже существует в этой группе",
            details={
                "entity": "segment_class",
                "field": "name",
                "value": name,
                "group_id": str(group_id),
            },
        )


async def list_category_segment_classes(
    db: AsyncSession,
    *,
    category_id: UUID,
) -> list[SegmentClass]:
    result = await db.execute(
        select(SegmentClass).where(SegmentClass.class_group_id == category_id)
    )
    return list(result.scalars().all())
