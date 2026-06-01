from __future__ import annotations

import asyncio
import time
from fractions import Fraction
from threading import Event
from typing import Protocol
from uuid import UUID

import av
import numpy as np
from aiortc import VideoStreamTrack
from aiortc.mediastreams import MediaStreamError
from modules.cameras.streaming.manager import CameraStreamManager
from modules.cameras.streaming.source import ManagedCameraFrameSource

from .stream import MIN_SUBSCRIBER_FPS


class InspectionFrameSource(Protocol):
    stopped: Event

    def get_latest_video_frame(self) -> tuple[np.ndarray | None, int]: ...


class _BaseVideoTrack(VideoStreamTrack):
    kind = "video"

    def __init__(self, *, fps: int) -> None:
        super().__init__()
        self._fps = max(1, min(fps, 30))
        self._frame_interval = 1.0 / self._fps
        self._next_frame_at = time.perf_counter()

        self._start_time: float | None = None
        self._time_base = Fraction(1, 90_000)

    async def _throttle(self) -> None:
        now = time.perf_counter()
        delay = self._next_frame_at - now

        if delay > 0:
            await asyncio.sleep(delay)

        current = time.perf_counter()
        self._next_frame_at = max(
            self._next_frame_at + self._frame_interval,
            current,
        )

    def _build_video_frame(self, frame: np.ndarray) -> av.VideoFrame:
        if self._start_time is None:
            self._start_time = time.perf_counter()

        video_frame = av.VideoFrame.from_ndarray(frame, format="bgr24")
        elapsed = time.perf_counter() - self._start_time
        video_frame.pts = int(elapsed * 90_000)
        video_frame.time_base = self._time_base
        return video_frame


class CameraPreviewVideoTrack(_BaseVideoTrack):
    kind = "video"

    def __init__(
        self,
        *,
        manager: CameraStreamManager,
        camera_id: UUID,
        fps: int = 20,
    ) -> None:
        super().__init__(fps=fps)
        self._manager = manager
        self._camera_id = camera_id

        self._source: ManagedCameraFrameSource | None = None
        self._last_frame_at: float | None = None

    async def start_preview(self) -> None:
        self._source = await self._manager.acquire_source(
            self._camera_id,
            fps=MIN_SUBSCRIBER_FPS,
        )

    async def wait_ready(self, *, timeout: float) -> None:
        if self._source is None:
            raise TimeoutError("Источник камеры не создан")
        await self._source.wait_for_frame(timeout=timeout)

    async def stop_preview(self) -> None:
        if self._source is not None:
            await self._source.close()
            self._source = None

    async def recv(self) -> av.VideoFrame:
        if self.readyState != "live":
            raise MediaStreamError

        await self._throttle()

        source = self._source
        if source is None:
            raise MediaStreamError

        frame = None

        while self.readyState == "live":
            frame_pair = source.get_latest_frame()
            if frame_pair is not None:
                candidate, captured_at = frame_pair
                if self._last_frame_at is None or captured_at > self._last_frame_at:
                    frame = candidate.copy()
                    self._last_frame_at = captured_at
                    break

            await asyncio.sleep(0.005)

        if frame is None:
            raise MediaStreamError

        return self._build_video_frame(frame)


class InspectionVideoTrack(_BaseVideoTrack):
    def __init__(
        self,
        source: InspectionFrameSource,
        *,
        fps: int = 20,
    ) -> None:
        super().__init__(fps=fps)
        self._source = source
        self._last_version = 0

    async def recv(self) -> av.VideoFrame:
        if self.readyState != "live" or self._source.stopped.is_set():
            raise MediaStreamError

        await self._throttle()

        frame = None

        while not self._source.stopped.is_set():
            candidate, version = self._source.get_latest_video_frame()
            if candidate is not None and version != self._last_version:
                frame = candidate
                self._last_version = version
                break

            await asyncio.sleep(0.005)

        if frame is None:
            raise MediaStreamError

        return self._build_video_frame(frame)
