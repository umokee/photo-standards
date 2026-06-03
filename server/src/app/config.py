import os
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Literal
from urllib.parse import quote_plus

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SERVER_ROOT = PROJECT_ROOT / "server"
DEFAULT_SAM2_ROOT = PROJECT_ROOT / "storage" / "weights"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / "storage" / "matplotlib"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / "storage"))
os.environ.setdefault("YOLO_VERBOSE", "false")


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=SERVER_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    APP_NAME: str = "PhotoStandards API"
    DEBUG: bool = False

    # Realtime observability.
    PHOTOAPP_REALTIME_PROFILER: bool = False
    PHOTOAPP_REALTIME_PROFILER_WINDOW: int = Field(default=120, ge=1)
    PHOTOAPP_REALTIME_PROFILER_SUMMARY_EVERY: int = Field(default=30, ge=1)

    # Inspection behavior.
    INSPECTION_VERIFICATION_MODE: Literal["alignment", "yolo_count"] = "alignment"

    INSPECTION_SLOT_SEARCH_EXPANSION: float = Field(default=2.35, ge=1.0, le=8.0)
    INSPECTION_SLOT_MIN_SCORE: float = Field(default=0.43, ge=0.0, le=1.0)
    INSPECTION_SLOT_MIN_DETECTION_CONTAINMENT: float = Field(default=0.10, ge=0.0, le=1.0)
    INSPECTION_SLOT_MIN_YOLO_CONFIDENCE: float = Field(default=0.08, ge=0.0, le=1.0)
    INSPECTION_SLOT_MIN_FEATURE_SUPPORT: int = Field(default=4, ge=0, le=64)
    INSPECTION_SLOT_FEATURE_SEARCH_EXPANSION: float = Field(default=2.75, ge=1.0, le=10.0)

    INSPECTION_MISSING_POLYGON_REFINEMENT: bool = True
    INSPECTION_MISSING_POLYGON_MIN_FEATURE_SUPPORT: int = Field(default=4, ge=0, le=64)
    INSPECTION_MISSING_POLYGON_MAX_REPROJECTION_ERROR: float = Field(
        default=14.0,
        ge=0.0,
        le=80.0,
    )
    INSPECTION_MISSING_POLYGON_EDGE_REFINEMENT: bool = True
    INSPECTION_MISSING_POLYGON_EDGE_SNAP_RADIUS: int = Field(default=14, ge=0, le=64)
    INSPECTION_MISSING_POLYGON_EDGE_BLEND: float = Field(default=0.65, ge=0.0, le=1.0)
    INSPECTION_MISSING_POLYGON_EDGE_DENSIFY_STEP: int = Field(default=8, ge=2, le=32)
    INSPECTION_MISSING_POLYGON_EDGE_SMOOTHING: float = Field(
        default=0.18,
        ge=0.0,
        le=0.45,
    )
    INSPECTION_MISSING_POLYGON_EDGE_SMOOTH_ITERATIONS: int = Field(
        default=2,
        ge=0,
        le=8,
    )

    MAX_REALTIME_INSPECTIONS: int = Field(default=2, ge=1, le=10)

    # Database connection.
    DB_HOST: str = "127.0.0.1"
    DB_PORT: int = 5432
    DB_NAME: str = "photo-standards"
    DB_USER: str = "postgres"
    DB_PASS: str = "postgres"  # noqa:S105
    DB_SOCKET_DIR: Path | None = None

    # Filesystem storage.
    STORAGE_ROOT: Path = PROJECT_ROOT / "storage"

    # Interactive segmentation.
    SAM2_DEVICE: str = "cpu"
    SAM2_ROOT: str = f"{STORAGE_ROOT}/weights"
    SAM2_MODEL_CFG: str = "configs/sam2.1/sam2.1_hiera_s.yaml"
    SAM2_CHECKPOINT: str = f"{DEFAULT_SAM2_ROOT}/sam2.1_hiera_small.pt"

    # Reference alignment.
    ALIGNMENT_BACKEND: Literal["auto", "torch", "orb"] = "auto"
    ALIGNMENT_DEVICE: Literal["auto", "cuda", "cpu"] = "auto"
    ALIGNMENT_ORB_FALLBACK: bool = True
    ALIGNMENT_IDENTITY_SHORTCUT: bool = True

    # YOLO training and inference.
    YOLO_DEVICE: Literal["auto", "cuda", "cpu"] = "auto"
    YOLO_DEFAULT_IMGSZ: int = Field(default=640, ge=32)
    YOLO_CONF_THRESHOLD: float = Field(default=0.05, ge=0.0, le=1.0)
    YOLO_REALTIME_CONF_THRESHOLD: float = Field(default=0.20, ge=0.0, le=1.0)
    YOLO_NMS_IOU: float = Field(default=0.55, ge=0.0, le=1.0)
    YOLO_EXTRA_CONF_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    YOLO_HALF: bool = False

    # Built-in background task runner.
    TASK_RUNNER_ENABLED: bool = True
    TASK_RUNNER_FAIL_ACTIVE_ON_STARTUP: bool = True
    TASK_RUNNER_POLL_INTERVAL_SEC: float = 1.0
    TASK_RUNNER_CPU_CONCURRENCY: int = 2
    TASK_RUNNER_GPU_CONCURRENCY: int = 2
    TASK_RUNNER_TRAINING_CONCURRENCY: int = 1
    TASK_RUNNER_INSPECTION_CONCURRENCY: int = 2
    TASK_RUNNER_MODEL_IMPORT_CONCURRENCY: int = 1
    TASK_RUNNER_HEARTBEAT_TIMEOUT_SEC: float = 900.0
    TASK_RUNNER_SERVICE_MONITOR_INTERVAL_SEC: float = 2.0
    TASK_RUNNER_SERVICE_RESTART_DELAY_SEC: float = 2.0
    TASK_RUNNER_SERVICE_MAX_RESTART_DELAY_SEC: float = 30.0
    TASK_RUNNER_SERVICE_STOP_TIMEOUT_SEC: float = 10.0
    TASK_RUNNER_SERVICE_PARENT_CHECK_INTERVAL_SEC: float = 3.0
    TASK_RUNNER_SERVICE_HEARTBEAT_INTERVAL_SEC: float = 3.0
    TASK_RUNNER_SERVICE_HEARTBEAT_TIMEOUT_SEC: float = 30.0
    TASK_RUNNER_SERVICE_MEMORY_LIMIT_MB: int = 0
    TASK_RUNNER_SERVICE_MIN_FREE_MEMORY_MB: int = 1024
    TASK_RUNNER_SERVICE_CPU_LIMIT_PERCENT: float = 95.0
    TASK_RUNNER_SERVICE_RESOURCE_GRACE_TICKS: int = 3

    @property
    def _database_host_query(self) -> str:
        if self.DB_SOCKET_DIR:
            return f"host={quote_plus(str(self.DB_SOCKET_DIR))}"

        return f"{self.DB_HOST}:{self.DB_PORT}"

    @property
    def database_url_async(self) -> str:
        if self.DB_SOCKET_DIR:
            return (
                f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}"
                f"@/{self.DB_NAME}?{self._database_host_query}"
            )

        return (
            f"postgresql+asyncpg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self._database_host_query}/{self.DB_NAME}"
        )

    @property
    def database_url_sync(self) -> str:
        if self.DB_SOCKET_DIR:
            return (
                f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}"
                f"@/{self.DB_NAME}?{self._database_host_query}"
            )

        return (
            f"postgresql+psycopg://{self.DB_USER}:{self.DB_PASS}"
            f"@{self._database_host_query}/{self.DB_NAME}"
        )

    @property
    def database_url_conninfo(self) -> str:
        if self.DB_SOCKET_DIR:
            return (
                f"postgresql://{self.DB_USER}:{self.DB_PASS}"
                f"@/{self.DB_NAME}?{self._database_host_query}"
            )

        return (
            f"postgresql://{self.DB_USER}:{self.DB_PASS}"
            f"@{self._database_host_query}/{self.DB_NAME}"
        )

    @property
    def database_url_async_for_listen(self) -> str:
        url = self.database_url_async
        return url.replace("postgresql+asyncpg", "postgresql")

    @property
    def standards_storage_path(self) -> Path:
        return self.STORAGE_ROOT / "standards"

    @property
    def inspections_storage_path(self) -> Path:
        return self.STORAGE_ROOT / "inspections"

    @property
    def models_storage_path(self) -> Path:
        return self.STORAGE_ROOT / "models"

    @property
    def logs_storage_path(self) -> Path:
        return self.STORAGE_ROOT / "logs"

    @property
    def current_logs_path(self) -> Path:
        current_date = datetime.now(UTC).date().isoformat()
        return self.logs_storage_path / current_date

    @property
    def worker_logs_path(self) -> Path:
        return self.current_logs_path

    @property
    def server_logs_path(self) -> Path:
        return self.current_logs_path


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


settings = get_settings()


__all__ = ["settings"]
