from __future__ import annotations

from datetime import UTC, datetime
from uuid import UUID

from app.exception import ConflictError, NotFoundError, ValidationError
from sqlalchemy import desc, select
from sqlalchemy.ext.asyncio import AsyncSession

from .constants import tasks as tasks_constants
from .models import Task
from .notifier import TaskNotifier

_UNSET = object()


async def get_task(
    db: AsyncSession,
    task_id: UUID,
) -> Task:
    task = await db.get(Task, task_id)
    if task is None:
        raise NotFoundError("Задача", task_id)
    return task


async def get_task_for_update(
    db: AsyncSession,
    task_id: UUID,
) -> Task:
    result = await db.execute(select(Task).where(Task.id == task_id).with_for_update())
    task = result.scalar_one_or_none()
    if task is None:
        raise NotFoundError("Задача", task_id)
    return task


async def list_tasks(
    db: AsyncSession,
    *,
    group_id: UUID | None = None,
    type: str | None = None,
    status: str | None = None,
    entity_type: str | None = None,
    entity_id: UUID | None = None,
) -> list[Task]:
    stmt = select(Task).order_by(desc(Task.created_at))
    if group_id is not None:
        stmt = stmt.where(Task.group_id == group_id)
    if type is not None:
        stmt = stmt.where(Task.type == type)
    if status is not None:
        stmt = stmt.where(Task.status == status)
    if entity_type is not None:
        stmt = stmt.where(Task.entity_type == entity_type)
    if entity_id is not None:
        stmt = stmt.where(Task.entity_id == entity_id)

    result = await db.execute(stmt)
    return list(result.scalars().all())


async def create_task(
    db: AsyncSession,
    *,
    type: str,
    status: str,
    queue: str | None,
    priority: int,
    task_id: UUID | None = None,
    payload: dict | None = None,
    entity_type: str | None = None,
    entity_id: UUID | None = None,
    group_id: UUID | None = None,
    auto_resume: bool = False,
    checkpoint_path: str | None = None,
    run_dir: str | None = None,
) -> Task:
    task = Task(
        type=type,
        status=status,
        queue=queue,
        priority=priority,
        payload=payload,
        entity_type=entity_type,
        entity_id=entity_id,
        group_id=group_id,
        auto_resume=auto_resume,
        checkpoint_path=checkpoint_path,
        run_dir=run_dir,
    )
    if task_id is not None:
        task.id = task_id

    db.add(task)
    await db.flush()
    return task


async def update_task_status(
    db: AsyncSession,
    *,
    task_id: UUID,
    status: str,
    stage: str | None = None,
    message: str | None = None,
    error: str | None = None,
    result: dict | None = None,
    external_job_id: str | None | object = _UNSET,
) -> Task:
    task = await get_task(db, task_id)
    _apply_state(
        task,
        status=status,
        stage=stage,
        message=message,
        error=error,
        result=result,
        external_job_id=external_job_id,
    )
    await TaskNotifier(db).status_changed(
        task_id,
        status=task.status,
        stage=task.stage,
        message=task.message,
        error=task.error,
    )
    await db.commit()
    await db.refresh(task)
    return task


async def update_task_progress(
    db: AsyncSession,
    *,
    task_id: UUID,
    current: int,
    total: int,
    stage: str,
    message: str | None = None,
    metrics: dict[str, float | None] | None = None,
) -> Task:
    task = await get_task(db, task_id)
    task.progress_current = current
    task.progress_total = total
    task.progress_percent = round(current / total * 100) if total else None
    task.stage = stage
    if message is not None:
        task.message = message
    task.heartbeat_at = _now()

    await TaskNotifier(db).progress(
        task_id,
        current=current,
        total=total,
        percent=task.progress_percent,
        stage=stage,
    )
    await db.commit()
    await db.refresh(task)
    return task


async def heartbeat_task(
    db: AsyncSession,
    task_id: UUID,
) -> None:
    task = await get_task(db, task_id)
    task.heartbeat_at = _now()
    await TaskNotifier(db).heartbeat(task_id)
    await db.commit()


async def pause_training_task(
    db: AsyncSession,
    *,
    task_id: UUID,
) -> Task:
    task = await get_task(db, task_id)

    if task.type != tasks_constants.types.training:
        raise ValidationError("Приостановить можно только задачу обучения")

    pausable_statuses = {
        tasks_constants.statuses.pending,
        tasks_constants.statuses.queued,
        tasks_constants.statuses.running,
        tasks_constants.statuses.resuming,
    }

    if task.status not in pausable_statuses:
        return task

    if task.status in {
        tasks_constants.statuses.pending,
        tasks_constants.statuses.queued,
    }:
        task.abort_requested = False
        task.auto_resume = False
        _apply_state(
            task,
            status=tasks_constants.statuses.paused,
            stage="Приостановлено",
            message="Обучение приостановлено пользователем",
        )
    else:
        task.abort_requested = True
        task.auto_resume = False
        _apply_state(
            task,
            status=tasks_constants.statuses.pausing,
            stage="Останавливаем обучение",
            message="Обучение будет приостановлено после текущей эпохи",
        )

    await TaskNotifier(db).status_changed(
        task_id,
        status=task.status,
        stage=task.stage,
        message=task.message,
        error=task.error,
    )
    await db.commit()
    await db.refresh(task)
    return task


async def resume_training_task(
    db: AsyncSession,
    *,
    task_id: UUID,
) -> Task:
    from infra.queue.scheduler import enqueue_training_resume

    task = await get_task(db, task_id)

    if task.type != tasks_constants.types.training:
        raise ValidationError("Возобновить можно только задачу обучения")

    if task.status != tasks_constants.statuses.paused:
        raise ConflictError("Задача не находится в состоянии паузы")

    return await enqueue_training_resume(db, task)


async def cancel_task(
    db: AsyncSession,
    *,
    task_id: UUID,
) -> Task:
    task = await get_task(db, task_id)

    if task.status in tasks_constants.statuses.terminal:
        return task

    task.abort_requested = True
    task.auto_resume = False

    _apply_state(
        task,
        status=tasks_constants.statuses.cancelled,
        stage="Отменено",
        message="Задача отменена пользователем",
    )

    await TaskNotifier(db).status_changed(
        task_id,
        status=task.status,
        stage=task.stage,
        message=task.message,
        error=task.error,
    )

    await db.commit()
    await db.refresh(task)

    return task


def _apply_state(
    task: Task,
    *,
    status: str | None = None,
    stage: str | None = None,
    message: str | None = None,
    error: str | None = None,
    result: dict | None = None,
    external_job_id: str | None | object = _UNSET,
) -> None:
    if status is not None:
        task.status = status
    if stage is not None:
        task.stage = stage
    if message is not None:
        task.message = message
    if error is not None:
        task.error = error[:2000]
    if result is not None:
        task.result = result
    if external_job_id is not _UNSET:
        task.external_job_id = external_job_id

    if (
        task.status
        in {
            tasks_constants.statuses.running,
            tasks_constants.statuses.resuming,
        }
        and task.started_at is None
    ):
        task.started_at = _now()

    if task.status in tasks_constants.statuses.terminal and task.finished_at is None:
        task.finished_at = _now()

    if task.status == tasks_constants.statuses.cancelled and task.cancelled_at is None:
        task.cancelled_at = _now()

    if task.status in {
        tasks_constants.statuses.running,
        tasks_constants.statuses.resuming,
        tasks_constants.statuses.pausing,
    }:
        task.heartbeat_at = _now()


def _now() -> datetime:
    return datetime.now(UTC).replace(tzinfo=None)
