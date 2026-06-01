from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from uuid import UUID

from app.exception import ConflictError, NotFoundError
from modules.core.segments.models import (
    SegmentAnnotation,
    SegmentClass,
    SegmentClassGroup,
)
from modules.core.standards.models import Standard, StandardImage
from modules.yolo.inspection.models import InspectionResult
from modules.yolo.training.models import MlModel
from sqlalchemy import distinct, func, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload

from .models import Group


@dataclass(slots=True)
class GroupStatsData:
    standards_count: int = 0
    images_count: int = 0
    annotated_images_count: int = 0
    polygons_count: int = 0
    segment_class_groups_count: int = 0
    segment_classes_count: int = 0
    models_count: int = 0
    inspections_count: int = 0


async def list_groups(
    db: AsyncSession,
) -> list[Group]:
    result = await db.execute(select(Group).order_by(Group.created_at.desc()))
    return list(result.scalars().all())


async def get_group(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> Group:
    group = await db.get(Group, group_id)
    if group is None:
        raise NotFoundError("Группа", group_id)
    return group


async def create_group(
    db: AsyncSession,
    *,
    name: str,
    description: str | None,
) -> Group:
    group = Group(name=name, description=description)
    db.add(group)
    await db.commit()
    await db.refresh(group)
    return group


async def save_group(
    db: AsyncSession,
    *,
    group: Group,
) -> Group:
    await db.commit()
    await db.refresh(group)
    return group


async def delete_group(
    db: AsyncSession,
    *,
    group: Group,
) -> None:
    await db.delete(group)
    await db.commit()


async def ensure_group_name_unique(
    db: AsyncSession,
    *,
    name: str,
    exclude_id: UUID | None = None,
) -> None:
    query = select(Group.id).where(Group.name == name)

    if exclude_id is not None:
        query = query.where(Group.id != exclude_id)

    if await db.scalar(query) is not None:
        raise ConflictError(
            "Группа уже существует",
            details={
                "entity": "group",
                "entity_label": "Группа",
                "field": "name",
                "value": name,
            },
        )


async def get_group_stats_map(
    db: AsyncSession,
    *,
    group_ids: list[UUID],
) -> dict[UUID, GroupStatsData]:
    if not group_ids:
        return {}

    stats: dict[UUID, GroupStatsData] = defaultdict(GroupStatsData)

    await _fill_direct_count(
        db,
        stats,
        group_ids,
        model=Standard,
        attr="standards_count",
    )
    await _fill_direct_count(
        db,
        stats,
        group_ids,
        model=SegmentClassGroup,
        attr="segment_class_groups_count",
    )
    await _fill_direct_count(
        db,
        stats,
        group_ids,
        model=SegmentClass,
        attr="segment_classes_count",
    )
    await _fill_direct_count(
        db,
        stats,
        group_ids,
        model=MlModel,
        attr="models_count",
    )
    await _fill_inspection_count(
        db,
        stats,
        group_ids,
    )

    images_rows = await db.execute(
        select(Standard.group_id, func.count(StandardImage.id))
        .select_from(Standard)
        .join(StandardImage, StandardImage.standard_id == Standard.id)
        .where(Standard.group_id.in_(group_ids))
        .group_by(Standard.group_id)
    )
    for group_id, count in images_rows:
        stats[group_id].images_count = int(count)

    annotations_rows = await db.execute(
        select(
            Standard.group_id,
            func.count(distinct(StandardImage.id)).label("annotated"),
            func.count(SegmentAnnotation.id).label("polygons"),
        )
        .select_from(Standard)
        .join(StandardImage, StandardImage.standard_id == Standard.id)
        .join(SegmentAnnotation, SegmentAnnotation.image_id == StandardImage.id)
        .where(Standard.group_id.in_(group_ids))
        .group_by(Standard.group_id)
    )
    for group_id, annotated, polygons in annotations_rows:
        stats[group_id].annotated_images_count = int(annotated)
        stats[group_id].polygons_count = int(polygons)

    return dict(stats)


async def list_group_standards(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> list[Standard]:
    result = await db.execute(
        select(Standard)
        .options(
            selectinload(Standard.images).selectinload(StandardImage.annotations),
        )
        .where(Standard.group_id == group_id)
        .order_by(Standard.created_at.desc())
    )
    return list(result.scalars().all())


async def get_active_group_model(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> MlModel | None:
    result = await db.execute(
        select(MlModel)
        .where(MlModel.group_id == group_id, MlModel.is_active.is_(True))
        .order_by(MlModel.version.desc(), MlModel.created_at.desc())
        .limit(1)
    )
    return result.scalars().first()


async def _fill_direct_count(
    db: AsyncSession,
    stats: dict[UUID, GroupStatsData],
    group_ids: list[UUID],
    *,
    model: type,
    attr: str,
) -> None:
    rows = await db.execute(
        select(model.group_id, func.count(model.id))
        .where(model.group_id.in_(group_ids))
        .group_by(model.group_id)
    )
    for group_id, count in rows:
        setattr(stats[group_id], attr, int(count))


async def _fill_inspection_count(
    db: AsyncSession,
    stats: dict[UUID, GroupStatsData],
    group_ids: list[UUID],
) -> None:
    rows = await db.execute(
        select(Standard.group_id, func.count(InspectionResult.id))
        .select_from(InspectionResult)
        .join(Standard, InspectionResult.standard_id == Standard.id)
        .where(Standard.group_id.in_(group_ids))
        .group_by(Standard.group_id)
    )

    for group_id, count in rows:
        stats[group_id].inspections_count = int(count)
