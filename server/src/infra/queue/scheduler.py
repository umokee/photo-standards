from __future__ import annotations

from uuid import UUID

from app.exception import ConflictError
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.models import Task
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession


async def ensure_no_active_task_for_group(
    db: AsyncSession,
    *,
    group_id: UUID,
    type: str,
) -> None:
    blocking = frozenset(tasks_constants.statuses.active) | {
        tasks_constants.statuses.paused,
    }

    result = await db.execute(
        select(Task)
        .where(
            Task.group_id == group_id,
            Task.type == type,
            Task.status.in_(blocking),
        )
        .limit(1)
    )

    if result.scalar_one_or_none() is not None:
        raise ConflictError("Для этой группы уже есть активная задача этого типа")


async def enqueue_training(
    db: AsyncSession,
    task: Task,
) -> Task:
    return await _commit_queued_task(
        db,
        task=task,
        queue=tasks_constants.queues.gpu,
        priority=tasks_constants.priorities.training,
        stage="В очереди",
        message="Обучение поставлено в GPU-очередь",
    )


async def enqueue_inspection(
    db: AsyncSession,
    task: Task,
) -> Task:
    return await _commit_queued_task(
        db,
        task=task,
        queue=tasks_constants.queues.gpu,
        priority=tasks_constants.priorities.inspection,
        stage="В очереди",
        message="Проверка поставлена в GPU-очередь",
    )


async def enqueue_training_resume(
    db: AsyncSession,
    task: Task,
) -> Task:
    task.abort_requested = False
    task.auto_resume = True

    return await _commit_queued_task(
        db,
        task=task,
        queue=tasks_constants.queues.gpu,
        priority=tasks_constants.priorities.training_resume,
        stage="В очереди",
        message="Возобновление после паузы",
    )


async def request_training_pause_for_inspection(
    db: AsyncSession,
) -> Task | None:
    result = await db.execute(
        select(Task)
        .where(
            Task.type == tasks_constants.types.training,
            Task.queue == tasks_constants.queues.gpu,
            Task.status.in_(
                {
                    tasks_constants.statuses.running,
                    tasks_constants.statuses.resuming,
                    tasks_constants.statuses.pausing,
                }
            ),
        )
        .order_by(Task.created_at.asc())
        .limit(1)
    )
    task = result.scalar_one_or_none()
    if task is None:
        return None

    task.abort_requested = True
    task.auto_resume = True
    task.status = tasks_constants.statuses.pausing
    task.stage = "Останавливаем обучение"
    task.message = "Обучение приостановлено ради приоритетной проверки"
    await db.commit()
    await db.refresh(task)
    return task


async def enqueue_model_import(
    db: AsyncSession,
    task: Task,
) -> Task:
    return await _commit_queued_task(
        db,
        task=task,
        queue=tasks_constants.queues.gpu,
        priority=tasks_constants.priorities.model_import,
        stage="В очереди",
        message="Импорт модели поставлен в GPU-очередь",
    )


async def maybe_resume_paused_training(
    db: AsyncSession,
    *,
    exclude_inspection_task_id: UUID | None = None,
) -> Task | None:
    active_inspections = await _count_active_inspections(
        db,
        exclude_task_id=exclude_inspection_task_id,
    )

    if active_inspections > 0:
        return None

    result = await db.execute(
        select(Task)
        .where(
            Task.type == tasks_constants.types.training,
            Task.queue == tasks_constants.queues.gpu,
            Task.status == tasks_constants.statuses.paused,
            Task.auto_resume.is_(True),
        )
        .order_by(Task.created_at.asc())
        .limit(1)
    )
    task = result.scalar_one_or_none()
    if task is None:
        return None

    return await enqueue_training_resume(db, task)


async def _count_active_inspections(
    db: AsyncSession,
    *,
    exclude_task_id: UUID | None = None,
) -> int:
    stmt = select(Task).where(
        Task.type == tasks_constants.types.inspection,
        Task.queue == tasks_constants.queues.gpu,
        Task.status.in_(tasks_constants.statuses.active),
    )

    if exclude_task_id is not None:
        stmt = stmt.where(Task.id != exclude_task_id)

    result = await db.execute(stmt)
    return len(list(result.scalars().all()))


async def _commit_queued_task(
    db: AsyncSession,
    *,
    task: Task,
    queue: str,
    priority: int,
    stage: str,
    message: str,
) -> Task:
    task.status = tasks_constants.statuses.queued
    task.queue = queue
    task.priority = priority
    task.stage = stage
    task.message = message
    task.external_job_id = None
    await db.commit()
    await db.refresh(task)
    return task
