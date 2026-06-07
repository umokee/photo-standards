from __future__ import annotations

import json
import re
import threading
import time
import traceback
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from pathlib import Path
from types import TracebackType
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import UUID, uuid4

import numpy as np
import structlog
from structlog.stdlib import ProcessorFormatter
from structlog.types import EventDict, WrappedLogger

try:
    import torch
except Exception:  # noqa: BLE001
    torch = None


_SECRET_KEYS = {
    "password",
    "token",
    "secret",
    "authorization",
    "camera_password",
    "db_pass",
}
_PRIORITY_FIELDS = (
    "timestamp",
    "level",
    "logger",
    "event",
    "message",
    "request_id",
    "task_id",
    "session_id",
    "group_id",
    "model_id",
    "camera_id",
    "duration_ms",
    "exception",
)
_MAX_STRING_LENGTH = 1_024
_MAX_COLLECTION_ITEMS = 20
_MAX_DEPTH = 4
_URL_AUTH_RE = re.compile(r"(?P<scheme>[a-z][a-z0-9+.-]*://)(?P<auth>[^/@]+)@")
_TIMESTAMPER = structlog.processors.TimeStamper(fmt="iso", utc=True, key="timestamp")
_THROTTLE_LOCK = threading.Lock()


@dataclass(slots=True)
class _ThrottleEntry:
    last_traceback_at: float = 0.0
    repeat_count: int = 0


_THROTTLE_STATE: dict[str, _ThrottleEntry] = {}


def configure_structlog(debug: bool) -> None:
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            _TIMESTAMPER,
            structlog.processors.StackInfoRenderer(),
            ProcessorFormatter.wrap_for_formatter,
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def bind_context(**fields: Any) -> dict[str, Any]:
    cleaned = safe_extra(**fields)
    if cleaned:
        structlog.contextvars.bind_contextvars(**cleaned)
    return cleaned


def clear_context() -> None:
    structlog.contextvars.clear_contextvars()


def safe_extra(**fields: Any) -> dict[str, Any]:
    return _sanitize_mapping(fields, depth=0)


def log_event(
    logger: Any,
    level: str,
    event: str,
    message: str | None = None,
    **fields: Any,
) -> None:
    payload = safe_extra(**fields)
    if message:
        payload["message"] = message

    log_method = getattr(logger, level.lower(), None)
    if log_method is None:
        raise AttributeError(f"Logger has no level method: {level}")

    log_method(event, **payload)


def elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


def throttled_log(
    logger: Any,
    *,
    event: str,
    error: BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None],
    throttle_key: str,
    interval_sec: float = 10.0,
    level: str = "error",
    repeated_event: str = "realtime.error.repeated",
    **fields: Any,
) -> bool:
    now = time.monotonic()
    with _THROTTLE_LOCK:
        state = _THROTTLE_STATE.setdefault(throttle_key, _ThrottleEntry())
        should_log_full = now - state.last_traceback_at >= interval_sec
        if should_log_full:
            suppressed_count = state.repeat_count
            state.repeat_count = 0
            state.last_traceback_at = now
        else:
            state.repeat_count += 1
            suppressed_count = state.repeat_count

    error_type = _error_type_name(error)
    payload = dict(fields)
    payload["error_type"] = error_type
    payload["throttle_key"] = throttle_key
    payload["interval_sec"] = interval_sec

    if should_log_full:
        if suppressed_count:
            payload["suppressed_count"] = suppressed_count
        payload["exception"] = _format_exception(error)
        log_event(logger, level, event, **payload)
        return True

    log_event(
        logger,
        "warning",
        repeated_event,
        count=suppressed_count,
        **payload,
    )
    return False


class RequestContextMiddleware:
    def __init__(self, app: Any) -> None:
        self.app = app

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        scope_type = scope.get("type")
        if scope_type not in {"http", "websocket"}:
            await self.app(scope, receive, send)
            return

        request_id = _resolve_request_id(scope)
        scope.setdefault("state", {})["request_id"] = request_id

        clear_context()
        bind_context(request_id=request_id)

        if scope_type == "websocket":
            try:
                await self.app(scope, receive, send)
            finally:
                clear_context()
            return

        async def send_wrapper(message: dict[str, Any]) -> None:
            if message.get("type") == "http.response.start":
                headers = list(message.get("headers", []))
                headers.append((b"x-request-id", request_id.encode("utf-8")))
                message["headers"] = headers
            await send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        finally:
            clear_context()


def build_console_formatter() -> ProcessorFormatter:
    return ProcessorFormatter(
        foreign_pre_chain=_build_foreign_pre_chain(),
        processors=[
            ProcessorFormatter.remove_processors_meta,
            _normalize_event_dict,
            _render_console,
        ],
    )


def build_json_formatter() -> ProcessorFormatter:
    return ProcessorFormatter(
        foreign_pre_chain=_build_foreign_pre_chain(),
        processors=[
            ProcessorFormatter.remove_processors_meta,
            _normalize_event_dict,
            _render_json,
        ],
    )


def _build_foreign_pre_chain() -> list[Any]:
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.ExtraAdder(),
        _TIMESTAMPER,
    ]


def _normalize_event_dict(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> EventDict:
    normalized = dict(event_dict)

    event = normalized.get("event")
    if event is None:
        event = ""
    normalized["event"] = str(event)

    exc_info = normalized.pop("exc_info", None)
    if exc_info:
        normalized["exception"] = _format_exception(exc_info)

    stack_info = normalized.pop("stack_info", None)
    if stack_info:
        normalized["stack"] = str(stack_info)

    return _order_event_dict(_sanitize_mapping(normalized, depth=0))


def _render_console(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> str:
    payload = dict(event_dict)
    exception = payload.pop("exception", None)
    timestamp = str(payload.pop("timestamp", ""))
    level = str(payload.pop("level", "info"))
    payload.pop("logger", None)
    event = str(payload.pop("event", ""))
    message = payload.pop("message", None)

    parts = [part for part in [timestamp, f"[{level}]", event] if part]
    rendered = " ".join(parts)

    if message:
        rendered = f"{rendered} message={_format_console_value(message)}"

    if payload:
        extras = " ".join(
            f"{key}={_format_console_value(value)}" for key, value in payload.items()
        )
        rendered = f"{rendered} {extras}".strip()

    if exception:
        rendered = f"{rendered}\n{exception}"

    return rendered


def _render_json(
    _logger: WrappedLogger,
    _method_name: str,
    event_dict: EventDict,
) -> str:
    return json.dumps(
        dict(event_dict),
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def _sanitize_mapping(
    values: Mapping[str, Any],
    *,
    depth: int,
) -> dict[str, Any]:
    cleaned: dict[str, Any] = {}

    for key, value in values.items():
        if value is None:
            continue
        sanitized = _sanitize_value(key=str(key), value=value, depth=depth)
        if sanitized is None:
            continue
        cleaned[str(key)] = sanitized

    return cleaned


def _sanitize_value(*, key: str, value: Any, depth: int) -> Any:
    key_lower = key.lower()
    if _is_secret_key(key_lower):
        return "***"

    if isinstance(value, UUID):
        return str(value)

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, (datetime, date)):
        return value.isoformat()

    if isinstance(value, Enum):
        return _sanitize_value(key=key, value=value.value, depth=depth)

    if isinstance(value, BaseException):
        return _format_exception(value)

    if isinstance(value, str):
        return _sanitize_string(key_lower, value)

    if isinstance(value, bytes | bytearray | memoryview):
        return f"<bytes:{len(value)}>"

    if isinstance(value, np.generic):
        return value.item()

    if isinstance(value, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(value.shape),
            "dtype": str(value.dtype),
        }

    if torch is not None and torch.is_tensor(value):
        shape = list(value.shape) if hasattr(value, "shape") else []
        return {
            "type": "tensor",
            "shape": shape,
            "dtype": str(getattr(value, "dtype", "unknown")),
            "device": str(getattr(value, "device", "unknown")),
        }

    if isinstance(value, Mapping):
        if depth >= _MAX_DEPTH:
            return {
                "type": "mapping",
                "size": len(value),
            }

        items = list(value.items())
        cleaned = _sanitize_mapping(
            dict(items[:_MAX_COLLECTION_ITEMS]),
            depth=depth + 1,
        )
        if len(items) > _MAX_COLLECTION_ITEMS:
            cleaned["__truncated_items__"] = len(items) - _MAX_COLLECTION_ITEMS
        return cleaned

    if isinstance(value, Sequence) and not isinstance(value, str):
        if depth >= _MAX_DEPTH:
            return {
                "type": "sequence",
                "size": len(value),
            }

        cleaned_items = [
            _sanitize_value(key=key, value=item, depth=depth + 1)
            for item in list(value)[:_MAX_COLLECTION_ITEMS]
        ]
        cleaned_items = [item for item in cleaned_items if item is not None]
        if len(value) > _MAX_COLLECTION_ITEMS:
            cleaned_items.append(
                f"<truncated:{len(value) - _MAX_COLLECTION_ITEMS} items>"
            )
        return cleaned_items

    if hasattr(value, "model_dump") and callable(value.model_dump):
        try:
            dumped = value.model_dump(mode="json")
        except Exception:  # noqa: BLE001
            return _sanitize_string(key_lower, repr(value))
        return _sanitize_value(key=key, value=dumped, depth=depth + 1)

    if isinstance(value, (bool, int, float)):
        return value

    return _sanitize_string(key_lower, repr(value))


def _sanitize_string(key_lower: str, value: str) -> str:
    if _looks_like_url_key(key_lower):
        value = _strip_url_auth(value)

    if key_lower == "exception":
        return value

    if len(value) > _MAX_STRING_LENGTH:
        value = f"{value[:_MAX_STRING_LENGTH]}...<truncated:{len(value)} chars>"

    return value


def _order_event_dict(payload: Mapping[str, Any]) -> dict[str, Any]:
    ordered: dict[str, Any] = {}

    for key in _PRIORITY_FIELDS:
        if key in payload:
            ordered[key] = payload[key]

    for key, value in payload.items():
        if key not in ordered:
            ordered[key] = value

    return ordered


def _format_console_value(value: Any) -> str:
    if isinstance(value, str):
        if not value:
            return '""'
        if any(char.isspace() for char in value) or any(
            char in value for char in ['"', "=", "{", "}", "[", "]", ","]
        ):
            return json.dumps(value, ensure_ascii=False)
        return value

    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))

    return str(value)


def _format_exception(
    exc_info: BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None],
) -> str:
    if isinstance(exc_info, BaseException):
        return "".join(
            traceback.format_exception(
                type(exc_info),
                exc_info,
                exc_info.__traceback__,
            )
        ).strip()

    if (
        isinstance(exc_info, tuple)
        and len(exc_info) == 3
        and isinstance(exc_info[0], type)
    ):
        return "".join(traceback.format_exception(*exc_info)).strip()

    return str(exc_info)


def _error_type_name(
    error: BaseException
    | tuple[type[BaseException], BaseException, TracebackType | None],
) -> str:
    if isinstance(error, BaseException):
        return type(error).__name__

    if isinstance(error, tuple) and len(error) == 3 and isinstance(error[0], type):
        return error[0].__name__

    return type(error).__name__


def _is_secret_key(key_lower: str) -> bool:
    return any(secret in key_lower for secret in _SECRET_KEYS)


def _looks_like_url_key(key_lower: str) -> bool:
    return (
        "url" in key_lower or key_lower.endswith("_dsn") or key_lower.endswith("_uri")
    )


def _strip_url_auth(value: str) -> str:
    try:
        parsed = urlsplit(value)
    except ValueError:
        return _URL_AUTH_RE.sub(r"\g<scheme>***@", value)

    if not parsed.scheme or not parsed.netloc or "@" not in parsed.netloc:
        return value

    host = parsed.netloc.rsplit("@", 1)[-1]
    return urlunsplit((parsed.scheme, host, parsed.path, parsed.query, parsed.fragment))


def _resolve_request_id(scope: Mapping[str, Any]) -> str:
    for key, value in scope.get("headers", []):
        if key.lower() == b"x-request-id":
            request_id = value.decode("utf-8", errors="ignore").strip()
            if request_id:
                return request_id[:200]
    return str(uuid4())
