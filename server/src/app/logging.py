from __future__ import annotations

import logging
import logging.config
from typing import Any

from app.config import settings
from app.observability import (
    build_console_formatter,
    build_json_formatter,
    configure_structlog,
)
from rich.traceback import install as install_rich_traceback

HEALTH_PATHS = (
    "GET /api/health",
    "HEAD /api/health",
    "GET /health",
    "HEAD /health",
)


class IgnoreHealthcheckFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(path in message for path in HEALTH_PATHS)


def build_logging_config(debug: bool = False) -> dict[str, Any]:
    level = "DEBUG" if debug else "INFO"
    ultralytics_level = "INFO" if debug else "WARNING"

    settings.server_logs_path.mkdir(parents=True, exist_ok=True)

    server_log_path = settings.server_logs_path / "server.log"
    server_error_log_path = settings.server_logs_path / "server.error.log"

    common_handlers = ["console", "file", "error_file"]

    return {
        "version": 1,
        "disable_existing_loggers": False,
        "filters": {
            "ignore_health": {
                "()": "app.logging.IgnoreHealthcheckFilter",
            },
        },
        "formatters": {
            "console": {
                "()": "app.observability.build_console_formatter",
            },
            "json": {
                "()": "app.observability.build_json_formatter",
            },
        },
        "handlers": {
            "console": {
                "class": "rich.logging.RichHandler",
                "level": level,
                "formatter": "console",
                "rich_tracebacks": False,
                "tracebacks_show_locals": False,
                "show_time": False,
                "show_level": False,
                "show_path": False,
                "markup": False,
            },
            "access_console": {
                "class": "rich.logging.RichHandler",
                "level": "INFO",
                "formatter": "console",
                "filters": ["ignore_health"],
                "rich_tracebacks": False,
                "show_time": False,
                "show_level": False,
                "show_path": False,
                "markup": False,
            },
            "file": {
                "class": "logging.FileHandler",
                "level": level,
                "formatter": "json",
                "filename": str(server_log_path),
                "encoding": "utf-8",
            },
            "error_file": {
                "class": "logging.FileHandler",
                "level": "WARNING",
                "formatter": "json",
                "filename": str(server_error_log_path),
                "encoding": "utf-8",
            },
            "access_file": {
                "class": "logging.FileHandler",
                "level": "INFO",
                "formatter": "json",
                "filename": str(server_log_path),
                "encoding": "utf-8",
                "filters": ["ignore_health"],
            },
        },
        "root": {
            "handlers": common_handlers,
            "level": level,
        },
        "loggers": {
            "uvicorn": {
                "handlers": common_handlers,
                "level": level,
                "propagate": False,
            },
            "uvicorn.error": {
                "handlers": common_handlers,
                "level": level,
                "propagate": False,
            },
            "uvicorn.access": {
                "handlers": ["access_console", "access_file"],
                "level": "INFO",
                "propagate": False,
            },
            "fastapi": {
                "handlers": common_handlers,
                "level": level,
                "propagate": False,
            },
            "sqlalchemy": {
                "handlers": common_handlers,
                "level": "WARNING",
                "propagate": False,
            },
            "sqlalchemy.engine": {
                "handlers": common_handlers,
                "level": "WARNING",
                "propagate": False,
            },
            "matplotlib": {
                "handlers": common_handlers,
                "level": "WARNING",
                "propagate": False,
            },
            "matplotlib.font_manager": {
                "handlers": common_handlers,
                "level": "WARNING",
                "propagate": False,
            },
            "ultralytics": {
                "handlers": common_handlers,
                "level": ultralytics_level,
                "propagate": False,
            },
            "aiortc": {
                "handlers": common_handlers,
                "level": "WARNING",
                "propagate": False,
            },
            "aioice": {
                "handlers": common_handlers,
                "level": "WARNING",
                "propagate": False,
            },
        },
    }


def configure_logging(debug: bool = False) -> None:
    if debug:
        install_rich_traceback(show_locals=False)

    configure_structlog(debug=debug)
    logging.config.dictConfig(build_logging_config(debug=debug))
    logging.captureWarnings(True)
