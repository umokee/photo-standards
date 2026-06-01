from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
from app.db import AsyncSessionLocal
from app.exception import NotFoundError, ValidationError
from app.observability import log_event, throttled_log
from modules.cameras.streaming.source import ManagedCameraFrameSource
from sqlalchemy import update as sa_update
import structlog

from ..models import Camera
from .capture import CaptureState
from .stream import MIN_SUBSCRIBER_FPS, CameraStream, Subscriber

logger = structlog.get_logger(__name__)

IDLE_TIMEOUT_SEC = 15.0


class CameraStreamManager:
    def __init__(self) -> None:
        self._streams: dict[UUID, CameraStream] = {}
        self._idle_timers: dict[UUID, asyncio.TimerHandle] = {}
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._stopped = False

    @property
    def loop(self) -> asyncio.AbstractEventLoop | None:
        return self._loop

    async def acquire_source(
        self,
        camera_id: UUID,
        *,
        fps: int = MIN_SUBSCRIBER_FPS,
    ) -> ManagedCameraFrameSource:
        subscriber = await self.subscribe(camera_id, fps=fps)

        try:
            async with self._lock:
                stream = self._streams.get(camera_id)

            if stream is None:
                raise RuntimeError("Стрим был неожиданно остановлен")

            return ManagedCameraFrameSource(
                manager=self,
                camera_id=camera_id,
                stream=stream,
                subscriber=subscriber,
            )
        except Exception:
            await self.unsubscribe(camera_id, subscriber)
            raise

    async def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._stopped = False

    async def get_stream(self, camera_id: UUID) -> CameraStream | None:
        async with self._lock:
            return self._streams.get(camera_id)

    async def stop(self) -> None:
        self._stopped = True

        async with self._lock:
            streams = list(self._streams.values())
            self._streams.clear()

            for timer in self._idle_timers.values():
                timer.cancel()
            self._idle_timers.clear()

        for stream in streams:
            try:
                await stream.stop()
                log_event(
                    logger,
                    "info",
                    "camera.stream.stopped",
                    camera_id=stream.camera_id,
                    reason="manager_stop",
                )
            except Exception as exc:  # noqa: BLE001
                throttled_log(
                    logger,
                    event="camera.stream.error",
                    repeated_event="camera.stream.error.repeated",
                    error=exc,
                    throttle_key=f"camera-manager-stop:{stream.camera_id}",
                    interval_sec=10.0,
                    camera_id=stream.camera_id,
                )

    async def subscribe(
        self,
        camera_id: UUID,
        fps: int,
    ) -> Subscriber:
        if self._stopped:
            raise RuntimeError("CameraStreamManager остановлен")

        stale_stream: CameraStream | None = None

        async with self._lock:
            existing = self._streams.get(camera_id)
            if existing is not None:
                self._cancel_idle_timer(camera_id)
                if existing.state == "offline":
                    stale_stream = self._streams.pop(camera_id, None)
                    existing = None

        if existing is not None:
            return await existing.subscribe(fps=fps)

        if stale_stream is not None:
            await stale_stream.stop()

        stream = await self._create_stream(camera_id)

        async with self._lock:
            already_present = self._streams.get(camera_id)
            if already_present is not None:
                use_existing = True
            else:
                self._streams[camera_id] = stream
                use_existing = False

        if use_existing:
            await stream.stop()
            return await already_present.subscribe(fps=fps)

        stream.start()
        return await stream.subscribe(fps=fps)

    async def unsubscribe(self, camera_id: UUID, subscriber: Subscriber) -> None:
        async with self._lock:
            stream = self._streams.get(camera_id)

        if stream is None:
            return

        await stream.unsubscribe(subscriber.subscriber_id)

    async def capture_one_frame(
        self,
        camera_id: UUID,
        *,
        timeout: float = 10.0,
    ) -> np.ndarray:
        async with await self.acquire_source(
            camera_id,
            fps=MIN_SUBSCRIBER_FPS,
        ) as source:
            return await source.wait_for_frame(timeout=timeout)

    async def update_subscriber_fps(
        self,
        camera_id: UUID,
        subscriber: Subscriber,
        fps: int,
    ) -> None:
        async with self._lock:
            stream = self._streams.get(camera_id)

        if stream is None:
            return

        await stream.update_subscriber_fps(subscriber.subscriber_id, fps)

    async def restart_camera(self, camera_id: UUID) -> None:
        async with self._lock:
            stream = self._streams.pop(camera_id, None)
            self._cancel_idle_timer(camera_id)

        if stream is not None:
            await stream.stop()
            log_event(
                logger,
                "info",
                "camera.stream.stopped",
                camera_id=camera_id,
                reason="restart_camera",
            )

    async def _create_stream(self, camera_id: UUID) -> CameraStream:
        async with AsyncSessionLocal() as db:
            camera = await db.get(Camera, camera_id)
            if camera is None:
                raise NotFoundError("Камера", camera_id)
            if not camera.is_active:
                raise ValidationError("Камера отключена")

            assert self._loop is not None
            stream = CameraStream(
                camera=camera,
                loop=self._loop,
                on_empty=self._on_stream_empty,
                on_state_change=lambda state, error, cid=camera_id: (
                    self._on_state_change(cid, state, error)
                ),
            )
            return stream

    def _on_stream_empty(self, stream: CameraStream) -> None:
        if self._stopped or self._loop is None:
            return

        camera_id = stream.camera_id

        existing_timer = self._idle_timers.get(camera_id)
        if existing_timer is not None:
            existing_timer.cancel()

        timer = self._loop.call_later(
            IDLE_TIMEOUT_SEC,
            lambda cid=camera_id: asyncio.create_task(self._idle_expire(cid)),
        )
        self._idle_timers[camera_id] = timer

    async def _idle_expire(self, camera_id: UUID) -> None:
        async with self._lock:
            timer = self._idle_timers.pop(camera_id, None)
            if timer is None:
                return

            stream = self._streams.get(camera_id)
            if stream is None:
                return

            if stream.subscriber_count > 0:
                return

            self._streams.pop(camera_id, None)

        try:
            await stream.stop()
            log_event(
                logger,
                "info",
                "camera.stream.stopped",
                camera_id=camera_id,
                reason="idle_timeout",
            )
        except Exception as exc:  # noqa: BLE001
            throttled_log(
                logger,
                event="camera.stream.error",
                repeated_event="camera.stream.error.repeated",
                error=exc,
                throttle_key=f"camera-manager-idle:{camera_id}",
                interval_sec=10.0,
                camera_id=camera_id,
            )

    def _cancel_idle_timer(self, camera_id: UUID) -> None:
        timer = self._idle_timers.pop(camera_id, None)
        if timer is not None:
            timer.cancel()

    def _on_state_change(
        self,
        camera_id: UUID,
        new_state: CaptureState,
        error: str | None,
    ) -> None:
        if self._stopped:
            return
        asyncio.create_task(self._persist_state_change(camera_id, new_state, error))

    async def _persist_state_change(
        self,
        camera_id: UUID,
        new_state: CaptureState,
        error: str | None,
    ) -> None:
        event_name = {
            "connecting": "camera.stream.started",
            "online": "camera.stream.online",
            "reconnecting": "camera.stream.reconnecting",
            "offline": "camera.stream.offline",
        }[new_state]
        log_event(
            logger,
            "info" if new_state in {"connecting", "online"} else "warning",
            event_name,
            camera_id=camera_id,
            reason=error,
        )
        last_status = "online" if new_state == "online" else "offline"

        try:
            async with AsyncSessionLocal() as db:
                await db.execute(
                    sa_update(Camera)
                    .where(Camera.id == camera_id)
                    .values(
                        last_status=last_status,
                        last_checked_at=datetime.now(UTC).replace(tzinfo=None),
                        last_error=(error[:500] if error else None),
                    )
                )
                await db.commit()
        except Exception as exc:  # noqa: BLE001
            throttled_log(
                logger,
                event="camera.stream.error",
                repeated_event="camera.stream.error.repeated",
                error=exc,
                throttle_key=f"camera-manager-persist:{camera_id}",
                interval_sec=10.0,
                camera_id=camera_id,
                state=new_state,
            )
