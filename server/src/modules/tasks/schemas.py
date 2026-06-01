from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict


class TaskResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    type: str
    status: str

    queue: str | None = None
    priority: int

    progress_current: int | None = None
    progress_total: int | None = None
    progress_percent: int | None = None

    stage: str | None = None
    message: str | None = None
    error: str | None = None

    payload: dict | None = None
    result: dict | None = None

    entity_type: str | None = None
    entity_id: UUID | None = None
    group_id: UUID | None = None
    created_by_id: UUID | None = None

    external_job_id: str | None = None
    abort_requested: bool
    auto_resume: bool
    checkpoint_path: str | None = None
    run_dir: str | None = None
    heartbeat_at: datetime | None = None

    created_at: datetime
    started_at: datetime | None = None
    finished_at: datetime | None = None
    cancelled_at: datetime | None = None


class TaskCancelResponse(BaseModel):
    task: TaskResponse
