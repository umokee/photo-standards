from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from contextlib import suppress

import asyncpg
from app.config import settings
from app.observability import log_event
import structlog

logger = structlog.get_logger(__name__)

CHANNEL = "photoetalon_events"

NotifyHandler = Callable[[dict], Awaitable[None]]


class EventBus:
    def __init__(self, dsn: str, *, reconnect_base_delay: float = 1.0) -> None:
        self._dsn = dsn
        self._reconnect_base_delay = reconnect_base_delay

        self._handlers: list[NotifyHandler] = []
        self._conn: asyncpg.Connection | None = None
        self._task: asyncio.Task[None] | None = None
        self._stopping = asyncio.Event()

    async def start(self) -> None:
        if self._task is not None and not self._task.done():
            return

        self._stopping.clear()
        self._task = asyncio.create_task(self._loop(), name="live-event-bus")

        log_event(logger, "info", "live.event_bus.start", channel=CHANNEL)

    async def stop(self) -> None:
        self._stopping.set()

        if self._task is not None:
            self._task.cancel()
            with suppress(asyncio.CancelledError):
                await self._task
            self._task = None

        if self._conn is not None and not self._conn.is_closed():
            await self._conn.close()
        self._conn = None

        log_event(logger, "info", "live.event_bus.stop")

    def add_handler(self, handler: NotifyHandler) -> Callable[[], None]:
        self._handlers.append(handler)

        def unsubscribe() -> None:
            with suppress(ValueError):
                self._handlers.remove(handler)

        return unsubscribe

    async def _loop(self) -> None:
        attempt = 0

        while not self._stopping.is_set():
            exc, had_connection = await self._listen_once()
            if exc is None:
                break
            attempt = 0 if had_connection else attempt
            attempt += 1
            delay = min(self._reconnect_base_delay * 2 ** (attempt - 1), 30.0)

            log_event(
                logger,
                "warning",
                "live.event_bus.reconnect",
                attempt=attempt,
                delay_sec=delay,
                reason=str(exc),
            )

            with suppress(asyncio.TimeoutError):
                await asyncio.wait_for(self._stopping.wait(), timeout=delay)

    async def _listen_once(self) -> tuple[Exception | None, bool]:
        had_connection = False

        try:
            self._conn = await asyncpg.connect(self._dsn)
            had_connection = True
            await self._conn.add_listener(CHANNEL, self._on_notify)

            log_event(logger, "info", "live.event_bus.connected", channel=CHANNEL)
            while not self._stopping.is_set():
                if self._conn.is_closed():
                    raise ConnectionError("asyncpg соединение закрыто")
                await asyncio.sleep(1.0)
            return None, had_connection
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            return exc, had_connection
        finally:
            if self._conn is not None and not self._conn.is_closed():
                with suppress(Exception):
                    await self._conn.remove_listener(CHANNEL, self._on_notify)
                with suppress(Exception):
                    await self._conn.close()
            self._conn = None

    def _on_notify(
        self,
        _conn: asyncpg.Connection,
        _pid: int,
        _channel: str,
        payload: str,
    ) -> None:
        try:
            data = json.loads(payload)
        except json.JSONDecodeError:
            log_event(
                logger,
                "warning",
                "live.event_bus.bad_payload",
                payload=payload[:500],
            )
            return

        if not isinstance(data, dict):
            log_event(logger, "warning", "live.event_bus.bad_payload_type")
            return

        for handler in list(self._handlers):
            asyncio.create_task(self._safe_call_handler(handler, data))

    async def _safe_call_handler(self, handler: NotifyHandler, data: dict) -> None:
        try:
            await handler(data)
        except Exception as exc:
            log_event(
                logger,
                "error",
                "live.event_bus.handler_failed",
                payload=data,
                error_type=type(exc).__name__,
                exception=exc,
            )


event_bus = EventBus(settings.database_url_async_for_listen)
