from __future__ import annotations

from uuid import UUID

from app.exception import ValidationError
from modules.yolo.training.adapters import repository, storage
from modules.yolo.training.models import MlModel
from sqlalchemy.ext.asyncio import AsyncSession


async def activate_model(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> MlModel:
    model = await repository.get_model(db, model_id=model_id)

    if not model.trained_at:
        raise ValidationError("Активировать можно только обученную модель")

    storage.ensure_model_weights_ready(model)
    model = await repository.set_active_model(db, model=model)

    await db.commit()
    await db.refresh(model)

    return model
