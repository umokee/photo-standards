from __future__ import annotations

from uuid import UUID

from app.exception import ConflictError
from modules.yolo.training.adapters import repository, storage
from sqlalchemy.ext.asyncio import AsyncSession


async def delete_model(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> None:
    model = await repository.get_model(db, model_id=model_id)

    if model.is_active:
        raise ConflictError("Нельзя удалить активную модель")

    has_running_task = await repository.has_blocking_training_task(
        db, model_id=model_id
    )
    if has_running_task:
        raise ConflictError(
            "Нельзя удалить модель с активной или приостановленной задачей обучения"
        )

    await repository.delete_model_record(db, model=model)
    await db.commit()

    storage.delete_model_artifacts(
        model_id=model_id,
        weights_path=model.weights_path,
    )
