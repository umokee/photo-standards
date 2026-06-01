from __future__ import annotations

import asyncio
import time
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Literal
from uuid import UUID

from app.db import AsyncSessionLocal
from app.observability import log_event
from modules.cameras import service as camera_service
from modules.cameras.models import Camera
from modules.cameras.schemas import CameraResponse
import structlog

logger = structlog.get_logger(__name__)

CameraStatusScope = Literal["cameras", "inspection"]

_CAMERA_MIN_PROBE_INTERVAL_SEC = 8.0

_CAMERA_STATUS_SLEEP_SEC = 2.0
_CAMERA_STATUS_MAX_PER_TICK = 2

_CAMERA_ONLINE_INTERVAL_CAMERAS_SEC = 30.0
_CAMERA_OFFLINE_INTERVAL_CAMERAS_SEC = 15.0
_CAMERA_UNKNOWN_INTERVAL_CAMERAS_SEC = 5.0

_CAMERA_ONLINE_INTERVAL_INSPECTION_SEC = 20.0
_CAMERA_OFFLINE_INTERVAL_INSPECTION_SEC = 10.0
_CAMERA_UNKNOWN_INTERVAL_INSPECTION_SEC = 3.0


def normalize_camera_status_scope(value: str | None) -> CameraStatusScope:
    if value == "inspection":
        return "inspection"

    return "cameras"


class CameraLiveStatusService:
    def __init__(self) -> None:
        self._probe_locks: dict[UUID, asyncio.Lock] = {}
        self._last_probe_at: dict[UUID, float] = {}
        self._failure_streak: dict[UUID, int] = {}

    async def watch(
        self,
        *,
        scope: CameraStatusScope,
    ) -> AsyncIterator[dict]:
        yield await self.build_snapshot_event()

        while True:
            changed_cameras = await self.probe_due_cameras(scope=scope)

            for camera_payload in changed_cameras:
                yield {
                    "kind": "camera_status",
                    "event": "camera",
                    "camera": camera_payload,
                }

            if changed_cameras:
                yield await self.build_snapshot_event()

            await asyncio.sleep(_CAMERA_STATUS_SLEEP_SEC)

    async def build_snapshot_event(self) -> dict:
        return {
            "kind": "camera_status",
            "event": "snapshot",
            "cameras": await self.list_cameras_payload(),
        }

    async def list_cameras_payload(self) -> list[dict]:
        async with AsyncSessionLocal() as db:
            cameras = await camera_service.list_cameras(db)

        return [self._camera_payload(camera) for camera in cameras]

    async def probe_due_cameras(
        self,
        *,
        scope: CameraStatusScope,
    ) -> list[dict]:
        async with AsyncSessionLocal() as db:
            cameras = await camera_service.list_cameras(db)

        now = self._now()

        due_cameras = [
            camera
            for camera in cameras
            if self._camera_is_due(camera, scope=scope, now=now)
        ][:_CAMERA_STATUS_MAX_PER_TICK]

        payloads: list[dict] = []

        for camera in due_cameras:
            payload = await self.probe_camera(camera.id)
            if payload is not None:
                payloads.append(payload)

        return payloads

    async def probe_camera(self, camera_id: UUID) -> dict | None:
        if not self._can_probe_now(camera_id):
            return None

        lock = self._probe_locks.setdefault(camera_id, asyncio.Lock())

        async with lock:
            if not self._can_probe_now(camera_id):
                return None

            self._last_probe_at[camera_id] = time.monotonic()

            async with AsyncSessionLocal() as db:
                camera = await db.get(Camera, camera_id)

                if camera is None:
                    return None

                previous_status = camera.last_status

                if not camera.is_active:
                    camera.last_status = "unknown"
                    camera.last_checked_at = self._now()
                    camera.last_error = "Камера отключена"
                    self._failure_streak.pop(camera_id, None)

                    await db.commit()
                    return self._camera_payload(camera)

                failures_before_probe = self._failure_streak.get(camera_id, 0)
                result = await camera_service.check_camera_health(
                    db,
                    camera,
                    failure_streak=failures_before_probe,
                )

                if result.is_available:
                    self._failure_streak.pop(camera_id, None)

                    if previous_status != "online":
                        log_event(
                            logger,
                            "info",
                            "camera.probe.success",
                            camera_id=camera_id,
                            width=result.width,
                            height=result.height,
                        )
                else:
                    failures = failures_before_probe + 1
                    self._failure_streak[camera_id] = failures

                    if camera.last_status != previous_status:
                        log_event(
                            logger,
                            "warning",
                            "camera.probe.failed",
                            camera_id=camera_id,
                            failures=failures,
                            status=camera.last_status,
                            reason=result.message,
                        )
                    elif previous_status == "online" and failures == 1:
                        log_event(
                            logger,
                            "debug",
                            "camera.probe.failed",
                            camera_id=camera_id,
                            failures=failures,
                            status=camera.last_status,
                            reason=result.message,
                        )

                return self._camera_payload(camera)

    def _can_probe_now(self, camera_id: UUID) -> bool:
        last_probe_at = self._last_probe_at.get(camera_id)

        if last_probe_at is None:
            return True

        return time.monotonic() - last_probe_at >= _CAMERA_MIN_PROBE_INTERVAL_SEC

    def _camera_is_due(
        self,
        camera: Camera,
        *,
        scope: CameraStatusScope,
        now: datetime,
    ) -> bool:
        if not camera.is_active:
            return False

        if camera.last_checked_at is None:
            return True

        age_sec = max(0.0, (now - camera.last_checked_at).total_seconds())

        if scope == "inspection":
            if camera.last_status == "online":
                return age_sec >= _CAMERA_ONLINE_INTERVAL_INSPECTION_SEC
            if camera.last_status == "offline":
                return age_sec >= _CAMERA_OFFLINE_INTERVAL_INSPECTION_SEC
            return age_sec >= _CAMERA_UNKNOWN_INTERVAL_INSPECTION_SEC

        if camera.last_status == "online":
            return age_sec >= _CAMERA_ONLINE_INTERVAL_CAMERAS_SEC
        if camera.last_status == "offline":
            return age_sec >= _CAMERA_OFFLINE_INTERVAL_CAMERAS_SEC
        return age_sec >= _CAMERA_UNKNOWN_INTERVAL_CAMERAS_SEC

    def _camera_payload(self, camera: Camera) -> dict:
        return CameraResponse.model_validate(camera).model_dump(mode="json")

    def _now(self) -> datetime:
        return datetime.now(UTC).replace(tzinfo=None)
