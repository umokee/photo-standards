from uuid import UUID, uuid4

from app.exception import ValidationError
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.service import get_task
from modules.yolo.inspection.adapters import repository, storage
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

    inspection_id = uuid4()
    original_image_path = data.get("image_path")
    original_result_image_path = data.get("result_image_path")

    saved_image_path, saved_result_image_path = storage.materialize_saved_inspection_files(
        inspection_id=inspection_id,
        image_path=data["image_path"],
        result_image_path=data.get("result_image_path"),
    )

    persisted_data = {
        **data,
        "image_path": saved_image_path,
        "result_image_path": saved_result_image_path,
    }

    try:
        inspection, segment_results = build_inspection_entities(
            data=persisted_data,
            notes=notes,
            inspection_id=inspection_id,
        )

        inspection = await repository.create_inspection(
            db,
            inspection=inspection,
            segment_results=segment_results,
        )
    except Exception:
        storage.unlink_storage_file(
            saved_image_path,
            log_message="Failed to cleanup copied inspection image %s",
        )
        storage.unlink_storage_file(
            saved_result_image_path,
            log_message="Failed to cleanup copied inspection result image %s",
        )
        raise

    task.result = {
        **persisted_data,
        "inspection_id": str(inspection.id),
    }
    await db.commit()

    if original_image_path != saved_image_path:
        storage.unlink_storage_file(
            original_image_path,
            log_message="Failed to cleanup temporary inspection image %s",
        )
    if original_result_image_path != saved_result_image_path:
        storage.unlink_storage_file(
            original_result_image_path,
            log_message="Failed to cleanup temporary inspection result image %s",
        )

    return inspection
