from uuid import UUID

from modules.yolo.inspection.adapters import repository
from modules.yolo.inspection.models import InspectionResult
from sqlalchemy.ext.asyncio import AsyncSession


async def list_history(
    db: AsyncSession,
    *,
    group_id: UUID | None = None,
) -> list[InspectionResult]:
    return await repository.list_inspections_history(db, group_id=group_id)
