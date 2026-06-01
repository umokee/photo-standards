from __future__ import annotations

import json
import logging
from typing import Final
from uuid import UUID

from app.live.bus import CHANNEL
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

logger = logging.getLogger(__name__)

_MAX_PAYLOAD_BYTES: Final = 7500


class TaskNotifier:
    def __init__(self, db: AsyncSession) -> None:
        self._db = db

    async def status_changed(
        self,
        task_id: UUID,
        *,
        status: str,
        stage: str | None = None,
        message: str | None = None,
        error: str | None = None,
    ) -> None:
        await self._notify(
            {
                "kind": "task",
                "task_id": str(task_id),
                "event": "status",
                "status": status,
                "stage": stage,
                "message": message,
                "error": error,
            }
        )

    async def progress(
        self,
        task_id: UUID,
        *,
        current: int,
        total: int,
        percent: int | None,
        stage: str,
    ) -> None:
        await self._notify(
            {
                "kind": "task",
                "task_id": str(task_id),
                "event": "progress",
                "current": current,
                "total": total,
                "percent": percent,
                "stage": stage,
            }
        )

    async def heartbeat(self, task_id: UUID) -> None:
        await self._notify(
            {
                "kind": "task",
                "task_id": str(task_id),
                "event": "heartbeat",
            }
        )

    async def _notify(self, payload: dict) -> None:
        raw = json.dumps(payload, ensure_ascii=False)
        if len(raw.encode("utf-8")) > _MAX_PAYLOAD_BYTES:
            logger.error(
                "task_notifier.payload_too_large",
                extra={
                    "event": "task_notifier.payload_too_large",
                    "task_id": payload.get("task_id"),
                    "size": len(raw),
                },
            )
            return
        await self._db.execute(
            text("SELECT pg_notify(:channel, :payload)"),
            {"channel": CHANNEL, "payload": raw},
        )
