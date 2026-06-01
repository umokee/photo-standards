from __future__ import annotations

from uuid import UUID

from modules.yolo.training.adapters import repository
from modules.yolo.training.models import MlModel
from sqlalchemy.ext.asyncio import AsyncSession


async def list_models(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> list[MlModel]:
    return await repository.list_models(db, group_id=group_id)
