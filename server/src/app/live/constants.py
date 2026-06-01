from __future__ import annotations

from constants_base import ConstModel


class RealtimeInspection(ConstModel):
    inference_interval_sec: float
    camera_fps: int
    frame_wait_interval_sec: float
    stream_online_timeout_sec: float


class RealtimeHeartbeat(ConstModel):
    interval_sec: float
    grace_sec: float


class RealtimeCloseCodes(ConstModel):
    limit_reached: int
    invalid_params: int
    internal_error: int


class RealtimeConstants(ConstModel):
    inspection: RealtimeInspection
    heartbeat: RealtimeHeartbeat
    close_codes: RealtimeCloseCodes


realtime = RealtimeConstants(
    inspection=RealtimeInspection(
        inference_interval_sec=0.2,
        camera_fps=10,
        frame_wait_interval_sec=0.05,
        stream_online_timeout_sec=15.0,
    ),
    heartbeat=RealtimeHeartbeat(
        interval_sec=15.0,
        grace_sec=30.0,
    ),
    close_codes=RealtimeCloseCodes(
        limit_reached=1013,
        invalid_params=1008,
        internal_error=1011,
    ),
)
