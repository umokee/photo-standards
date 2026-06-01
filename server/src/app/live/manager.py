from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from uuid import UUID, uuid4

from fastapi import WebSocket
from fastapi.websockets import WebSocketState

from .bus import EventBus

logger = logging.getLogger(__name__)

SEND_TIMEOUT_SEC = 5.0


class ConnectionManager:
    def __init__(self, bus: EventBus) -> None:
        self._bus = bus
        self._subs: dict[UUID, dict[UUID, WebSocket]] = defaultdict(dict)
        self._lock = asyncio.Lock()
        self._unsubscribe_bus: callable | None = None

    async def start(self) -> None:
        if self._unsubscribe_bus is not None:
            return
        self._unsubscribe_bus = self._bus.add_handler(self._on_event)

    async def stop(self) -> None:
        if self._unsubscribe_bus is not None:
            self._unsubscribe_bus()
            self._unsubscribe_bus = None

        async with self._lock:
            sockets: list[WebSocket] = []
            for clients in self._subs.values():
                sockets.extend(clients.values())
            self._subs.clear()

        for ws in sockets:
            with suppress(Exception):
                await ws.close(code=1001)

    @asynccontextmanager
    async def subscribe_task(self, task_id: UUID, ws: WebSocket) -> AsyncIterator[None]:
        sub_id = uuid4()
        async with self._lock:
            self._subs[task_id][sub_id] = ws
        logger.debug(
            "live.ws.subscribe_task",
            extra={"event": "live.ws.subscribe_task", "task_id": str(task_id)},
        )
        try:
            yield
        finally:
            async with self._lock:
                self._subs[task_id].pop(sub_id, None)
                if not self._subs[task_id]:
                    self._subs.pop(task_id, None)
            logger.debug(
                "live.ws.unsubscribe_task",
                extra={"event": "live.ws.unsubscribe_task", "task_id": str(task_id)},
            )

    async def _on_event(self, data: dict) -> None:
        if data.get("kind") != "task":
            return

        try:
            task_id = UUID(data["task_id"])
        except (KeyError, ValueError, TypeError):
            logger.warning(
                "live.ws.bad_task_event",
                extra={"event": "live.ws.bad_task_event", "payload": data},
            )
            return

        async with self._lock:
            subscribers = list(self._subs.get(task_id, {}).items())

        if not subscribers:
            return

        results = await asyncio.gather(
            *(self._send(sub_id, ws, data) for sub_id, ws in subscribers),
            return_exceptions=True,
        )

        dead = [
            sub_id
            for (sub_id, _), result in zip(subscribers, results, strict=True)
            if result is False or isinstance(result, Exception)
        ]

        if not dead:
            return

        async with self._lock:
            bucket = self._subs.get(task_id, {})

            for sub_id in dead:
                bucket.pop(sub_id, None)

            if not bucket:
                self._subs.pop(task_id, None)

    async def _send(self, sub_id: UUID, ws: WebSocket, data: dict) -> bool:
        if ws.client_state != WebSocketState.CONNECTED:
            return False

        try:
            await asyncio.wait_for(ws.send_json(data), timeout=SEND_TIMEOUT_SEC)
            return True
        except Exception as exc:
            logger.debug(
                "live.ws.send_failed",
                extra={
                    "event": "live.ws.send_failed",
                    "sub_id": str(sub_id),
                    "error": str(exc),
                },
            )
            return False
