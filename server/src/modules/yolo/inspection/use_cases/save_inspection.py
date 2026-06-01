from uuid import UUID

from app.exception import ValidationError
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.service import get_task
from modules.yolo.inspection.adapters import repository
from modules.yolo.inspection.adapters.entities import build_inspection_entities
from modules.yolo.inspection.models import InspectionResult
from sqlalchemy.ext.asyncio import AsyncSession


async def save_inspection(
    db: AsyncSession,
    *,
    task_id: UUID,
    notes: str | None,
) -> InspectionResult:
    task = await get_task(db, task_id)
    if task.type != tasks_constants.types.inspection:
        raise ValidationError("Переданная задача не является задачей проверки")
    if task.status != tasks_constants.statuses.succeeded or not isinstance(
        task.result, dict
    ):
        raise ValidationError("Задача ещё не завершена или не содержит результата")

    data = task.result
    inspection_status = data.get("inspection_status") or data.get("status")
    if not inspection_status:
        raise ValidationError("В результате задачи отсутствует статус проверки")

    inspection, segment_results = build_inspection_entities(
        data=task.result,
        notes=notes,
    )

    return await repository.create_inspection(
        db,
        inspection=inspection,
        segment_results=segment_results,
    )
