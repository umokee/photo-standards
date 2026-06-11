from uuid import UUID

from app.exception import NotFoundError
from modules.core.groups.models import Group
from modules.core.segments.models import SegmentAnnotation, SegmentClass
from modules.core.standards.models import Standard, StandardImage
from modules.yolo.inspection.models import InspectionResult, InspectionSegmentResult
from modules.yolo.training.models import MlModel
from sqlalchemy import or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload


async def get_inspection(
    db: AsyncSession,
    *,
    inspection_id: UUID,
) -> InspectionResult:
    result = await db.execute(
        select(InspectionResult)
        .options(
            selectinload(InspectionResult.segment_results),
            selectinload(InspectionResult.standard).selectinload(Standard.images),
            selectinload(InspectionResult.ml_model),
            selectinload(InspectionResult.camera),
            selectinload(InspectionResult.user),
        )
        .where(InspectionResult.id == inspection_id)
    )
    inspection = result.scalar_one_or_none()
    if not inspection:
        raise NotFoundError("Результат проверки", inspection_id)
    return inspection


async def list_inspections_history(
    db: AsyncSession,
    *,
    group_id: UUID | None = None,
) -> list[InspectionResult]:
    stmt = (
        select(InspectionResult)
        .outerjoin(Standard, InspectionResult.standard_id == Standard.id)
        .outerjoin(MlModel, InspectionResult.model_id == MlModel.id)
        .options(
            selectinload(InspectionResult.standard).selectinload(Standard.images),
            selectinload(InspectionResult.ml_model),
            selectinload(InspectionResult.camera),
        )
        .order_by(InspectionResult.inspected_at.desc())
    )

    if group_id is not None:
        stmt = stmt.where(
            or_(Standard.group_id == group_id, MlModel.group_id == group_id)
        )

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def has_inspection_with_image_path(
    db: AsyncSession,
    *,
    image_path: str,
) -> bool:
    result = await db.execute(
        select(InspectionResult.id)
        .where(InspectionResult.image_path == image_path)
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def has_inspection_with_result_image_path(
    db: AsyncSession,
    *,
    result_image_path: str,
) -> bool:
    result = await db.execute(
        select(InspectionResult.id)
        .where(InspectionResult.result_image_path == result_image_path)
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def get_standard_for_inspection(
    db: AsyncSession,
    *,
    standard_id: UUID,
) -> Standard:
    result = await db.execute(
        select(Standard)
        .options(
            selectinload(Standard.images)
            .selectinload(StandardImage.annotations)
            .selectinload(SegmentAnnotation.segment_class),
            selectinload(Standard.group)
            .selectinload(Group.segment_classes)
            .selectinload(SegmentClass.class_group),
        )
        .where(Standard.id == standard_id)
    )
    standard = result.scalar_one_or_none()
    if not standard:
        raise NotFoundError("Эталон", standard_id)
    return standard


async def create_inspection(
    db: AsyncSession,
    *,
    inspection: InspectionResult,
    segment_results: list[InspectionSegmentResult],
    commit: bool = True,
) -> InspectionResult:
    db.add(inspection)
    await db.flush()

    for segment_result in segment_results:
        segment_result.inspection_id = inspection.id
        db.add(segment_result)

    if commit:
        await db.commit()
        await db.refresh(inspection)

    return inspection
