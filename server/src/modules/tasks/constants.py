from __future__ import annotations

from constants_base import ConstModel


class TaskQueues(ConstModel):
    cpu: str
    gpu: str


class TaskTypes(ConstModel):
    training: str
    inspection: str
    model_import: str


class TaskStatuses(ConstModel):
    pending: str
    queued: str
    running: str
    pausing: str
    paused: str
    resuming: str
    succeeded: str
    failed: str
    cancelled: str
    active: tuple[str, ...]
    terminal: tuple[str, ...]


class TaskPriorities(ConstModel):
    training: int
    training_resume: int
    inspection: int
    model_import: int


class TasksConstants(ConstModel):
    queues: TaskQueues
    types: TaskTypes
    statuses: TaskStatuses
    priorities: TaskPriorities


tasks = TasksConstants(
    queues=TaskQueues(
        cpu="cpu",
        gpu="gpu",
    ),
    types=TaskTypes(
        training="model_training",
        inspection="inspection_run",
        model_import="model_import",
    ),
    statuses=TaskStatuses(
        pending="pending",
        queued="queued",
        running="running",
        pausing="pausing",
        paused="paused",
        resuming="resuming",
        succeeded="succeeded",
        failed="failed",
        cancelled="cancelled",
        active=("pending", "queued", "running", "pausing", "resuming"),
        terminal=("succeeded", "failed", "cancelled"),
    ),
    priorities=TaskPriorities(
        training=40,
        training_resume=60,
        model_import=80,
        inspection=100,
    ),
)
