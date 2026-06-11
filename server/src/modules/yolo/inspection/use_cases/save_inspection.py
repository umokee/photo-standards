from uuid import UUID, uuid4

from app.exception import ValidationError
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.service import get_task_for_update
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
    task = await get_task_for_update(db, task_id)
    if task.type != tasks_constants.types.inspection:
        await db.rollback()
        raise ValidationError("Переданная задача не является задачей проверки")
    if task.status != tasks_constants.statuses.succeeded or not isinstance(
        task.result, dict
    ):
        await db.rollback()
        raise ValidationError("Задача ещё не завершена или не содержит результата")

    data = task.result
    inspection_status = data.get("inspection_status") or data.get("status")
    if not inspection_status:
        await db.rollback()
        raise ValidationError("В результате задачи отсутствует статус проверки")

    existing_inspection_id = data.get("inspection_id")
    if existing_inspection_id:
        try:
            await db.rollback()
            return await repository.get_inspection(
                db,
                inspection_id=UUID(str(existing_inspection_id)),
            )
        except Exception as exc:
            raise ValidationError(
                "Задача уже содержит ссылку на сохранённую проверку, но запись истории недоступна"
            ) from exc

    inspection_id = uuid4()
    original_image_path = data.get("image_path")
    original_result_image_path = data.get("result_image_path")

    try:
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

        inspection, segment_results = build_inspection_entities(
            data=persisted_data,
            notes=notes,
            inspection_id=inspection_id,
        )

        inspection = await repository.create_inspection(
            db,
            inspection=inspection,
            segment_results=segment_results,
            commit=False,
        )

        task.result = {
            **persisted_data,
            "inspection_id": str(inspection.id),
        }
        await db.commit()
    except Exception:
        await db.rollback()
        storage.unlink_storage_file(
            locals().get("saved_image_path"),
            log_message="Failed to cleanup copied inspection image %s",
        )
        storage.unlink_storage_file(
            locals().get("saved_result_image_path"),
            log_message="Failed to cleanup copied inspection result image %s",
        )
        raise

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
