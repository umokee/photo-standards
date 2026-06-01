from uuid import UUID

from modules.yolo.inspection.adapters import repository
from modules.yolo.inspection.models import InspectionResult
from sqlalchemy.ext.asyncio import AsyncSession


async def get_inspection(
    db: AsyncSession,
    *,
    inspection_id: UUID,
) -> InspectionResult:
    return await repository.get_inspection(db, inspection_id=inspection_id)
