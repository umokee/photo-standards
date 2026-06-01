from __future__ import annotations

from uuid import UUID

from modules.yolo.training.adapters import repository
from modules.yolo.training.models import MlModel
from sqlalchemy.ext.asyncio import AsyncSession


async def get_model(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> MlModel:
    return await repository.get_model(db, model_id=model_id)
