from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterator
from contextlib import contextmanager
from threading import Lock

import structlog
from app.config import settings
from app.observability import log_event

logger = structlog.get_logger(__name__)

_ENABLED = settings.PHOTOAPP_REALTIME_PROFILER
_SUMMARY_WINDOW = settings.PHOTOAPP_REALTIME_PROFILER_WINDOW
_SUMMARY_EVERY = settings.PHOTOAPP_REALTIME_PROFILER_SUMMARY_EVERY

_HISTORY_LOCK = Lock()
_SESSION_STAGE_HISTORY: dict[str, dict[str, deque[float]]] = {}
_SESSION_FRAME_COUNTS: dict[str, int] = {}

_RESERVED_LOG_KEYS: frozenset[str] = frozenset(
    {
        "message",
        "asctime",
        "args",
        "created",
        "exc_info",
        "filename",
        "funcName",
        "levelname",
        "levelno",
        "lineno",
        "module",
        "msecs",
        "msg",
        "name",
        "pathname",
        "process",
        "processName",
        "relativeCreated",
        "stack_info",
        "thread",
        "threadName",
    }
)


class FrameProfiler:
    def __init__(self, session_id: str) -> None:
        self._session_id = session_id
        self._stages: dict[str, float] = {}
        self._started_at = time.perf_counter()

    @classmethod
    def is_enabled(cls) -> bool:
        return _ENABLED

    @classmethod
    def clear_session(cls, session_id: str) -> None:
        if not _ENABLED:
            return

        with _HISTORY_LOCK:
            _SESSION_STAGE_HISTORY.pop(session_id, None)
            _SESSION_FRAME_COUNTS.pop(session_id, None)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not _ENABLED:
            yield
            return

        t0 = time.perf_counter()
        try:
            yield
        finally:
            self._stages[f"{name}_ms"] = round((time.perf_counter() - t0) * 1000, 1)

    def record(self, name: str, ms: float) -> None:
        if not _ENABLED:
            return
        metric_name = name if name.endswith("_ms") else f"{name}_ms"
        self._stages[metric_name] = round(ms, 1)

    def commit(self, *, extra: dict | None = None) -> None:
        if not _ENABLED:
            return

        total_ms = round((time.perf_counter() - self._started_at) * 1000, 1)

        payload = {
            "session_id": self._session_id,
            "total_ms": total_ms,
            **self._stages,
        }
        if extra:
            payload.update(
                {k: v for k, v in extra.items() if k not in _RESERVED_LOG_KEYS}
            )
        log_event(
            logger,
            "info",
            "realtime.frame.profile",
            **_normalize_profile_payload(payload),
        )

        summary_payload = _build_summary_payload(self._session_id, payload)
        if summary_payload is not None:
            log_event(
                logger,
                "info",
                "realtime.frame.profile.summary",
                **summary_payload,
            )


def _build_summary_payload(
    session_id: str,
    payload: dict[str, object],
) -> dict[str, object] | None:
    stage_values = {
        key: float(value)
        for key, value in payload.items()
        if key.endswith("_ms") and isinstance(value, (int, float))
    }
    if not stage_values:
        return None

    with _HISTORY_LOCK:
        history = _SESSION_STAGE_HISTORY.setdefault(session_id, {})

        for key, value in stage_values.items():
            values = history.get(key)
            if values is None:
                values = deque(maxlen=_SUMMARY_WINDOW)
                history[key] = values
            values.append(value)

        frame_count = _SESSION_FRAME_COUNTS.get(session_id, 0) + 1
        _SESSION_FRAME_COUNTS[session_id] = frame_count

        if frame_count % _SUMMARY_EVERY != 0:
            return None

        summary: dict[str, object] = {
            "session_id": session_id,
            "frame_count": frame_count,
            "window_size": _SUMMARY_WINDOW,
            "frames_in_window": min(frame_count, _SUMMARY_WINDOW),
        }

        for metric_name, values in history.items():
            if not values:
                continue

            stage_name = (
                metric_name[:-3] if metric_name.endswith("_ms") else metric_name
            )
            summary[f"{stage_name}_p50_ms"] = _percentile(values, 0.50)
            summary[f"{stage_name}_p95_ms"] = _percentile(values, 0.95)

        return summary


def _percentile(values: deque[float], ratio: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return round(ordered[0], 1)

    position = (len(ordered) - 1) * ratio
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)

    lower_value = ordered[lower_index]
    upper_value = ordered[upper_index]
    interpolated = lower_value + (upper_value - lower_value) * (position - lower_index)
    return round(interpolated, 1)


def _normalize_profile_payload(payload: dict[str, object]) -> dict[str, object]:
    normalized = dict(payload)
    aliases = {
        "inspect_alignment_ms": "alignment_ms",
        "inspect_detection_ms": "detection_ms",
        "inspect_matching_ms": "matching_ms",
        "inspect_parallel_wait_ms": "parallel_wait_ms",
        "render_overlay_ms": "render_ms",
    }

    for old_key, new_key in aliases.items():
        if old_key in normalized and new_key not in normalized:
            normalized[new_key] = normalized[old_key]

    return normalized
