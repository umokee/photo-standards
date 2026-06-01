from __future__ import annotations

import asyncio
from types import TracebackType
from typing import TYPE_CHECKING
from uuid import UUID

import numpy as np
from app.observability import log_event

from .stream import CameraStream, Subscriber
import structlog

if TYPE_CHECKING:
    from .manager import CameraStreamManager

logger = structlog.get_logger(__name__)


class ManagedCameraFrameSource:
    def __init__(
        self,
        *,
        manager: CameraStreamManager,
        camera_id: UUID,
        stream: CameraStream,
        subscriber: Subscriber,
    ) -> None:
        self._manager = manager
        self._camera_id = camera_id
        self._stream = stream
        self._subscriber = subscriber
        self._closed = False

    @property
    def camera_id(self) -> UUID:
        return self._camera_id

    def get_latest_frame(self) -> tuple[np.ndarray, float] | None:
        return self._stream.get_latest_frame()

    async def wait_for_frame(self, *, timeout: float) -> np.ndarray:
        return await self._stream.wait_for_online_frame(timeout=timeout)

    async def close(self) -> None:
        if self._closed:
            return

        self._closed = True
        await self._manager.unsubscribe(self._camera_id, self._subscriber)

    def close_threadsafe(self, *, timeout: float = 2.0) -> None:
        if self._closed:
            return

        loop = self._manager.loop
        if loop is None or loop.is_closed():
            self._closed = True
            return

        try:
            running_loop = asyncio.get_running_loop()
        except RuntimeError:
            running_loop = None

        if running_loop is loop:
            raise RuntimeError("Use 'await source.close()' inside manager event loop")

        future = asyncio.run_coroutine_threadsafe(self.close(), loop)

        try:
            future.result(timeout=timeout)
        except Exception as exc:
            log_event(
                logger,
                "warning",
                "camera.stream.close_threadsafe_failed",
                camera_id=self._camera_id,
                error_type=type(exc).__name__,
                exception=exc,
            )

    async def __aenter__(self) -> ManagedCameraFrameSource:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.close()
