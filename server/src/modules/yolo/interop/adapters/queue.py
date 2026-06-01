from infra.queue.scheduler import enqueue_model_import as _enqueue_model_import
from modules.tasks.models import Task
from sqlalchemy.ext.asyncio import AsyncSession


async def enqueue_model_import(
    db: AsyncSession,
    task: Task,
) -> Task:
    return await _enqueue_model_import(db, task)
