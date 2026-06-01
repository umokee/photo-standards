from __future__ import annotations

from uuid import UUID

from infra.queue.scheduler import (
    enqueue_training as _enqueue_training,
)
from infra.queue.scheduler import (
    enqueue_training_resume as _enqueue_training_resume,
)
from infra.queue.scheduler import (
    ensure_no_active_task_for_group,
)
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.models import Task
from sqlalchemy.ext.asyncio import AsyncSession


async def ensure_no_active_training_task(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> None:
    await ensure_no_active_task_for_group(
        db,
        group_id=group_id,
        type=tasks_constants.types.training,
    )


async def enqueue_training(
    db: AsyncSession,
    task: Task,
) -> Task:
    return await _enqueue_training(db, task)


async def enqueue_training_resume(
    db: AsyncSession,
    task: Task,
) -> Task:
    return await _enqueue_training_resume(db, task)
