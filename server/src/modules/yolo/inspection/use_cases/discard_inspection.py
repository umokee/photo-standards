from uuid import UUID

from app.exception import ValidationError
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.service import get_task
from modules.yolo.inspection.adapters import storage
from sqlalchemy.ext.asyncio import AsyncSession


async def discard_inspection(
    db: AsyncSession,
    *,
    task_id: UUID,
) -> None:
    task = await get_task(db, task_id)

    if task.type != tasks_constants.types.inspection:
        raise ValidationError("Переданная задача не является задачей проверки")

    if task.status not in (
        tasks_constants.statuses.succeeded,
        tasks_constants.statuses.failed,
    ):
        raise ValidationError(
            "Дождитесь завершения проверки, прежде чем сбросить результат"
        )

    await storage.cleanup_unreferenced_inspection_task_files(
        db,
        payload=task.payload,
        result=task.result,
    )

    await db.delete(task)
    await db.commit()
