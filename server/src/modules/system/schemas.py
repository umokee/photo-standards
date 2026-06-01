from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel


class SystemGpuStatsResponse(BaseModel):
    available: bool
    name: str | None = None
    utilization_percent: int | None = None
    memory_used_mb: int | None = None
    memory_total_mb: int | None = None
    temperature_c: int | None = None


class SystemInfoResponse(BaseModel):
    hostname: str
    uptime_sec: int


class SystemResourceStatsResponse(BaseModel):
    cpu_percent: float
    cpu_count_logical: int
    memory_used_bytes: int
    memory_total_bytes: int
    disk_used_bytes: int
    disk_total_bytes: int


class SystemStorageCategoriesResponse(BaseModel):
    standards_bytes: int
    inspections_bytes: int
    models_bytes: int
    logs_bytes: int
    other_bytes: int


class SystemStorageStatsResponse(BaseModel):
    used_bytes: int
    categories: SystemStorageCategoriesResponse


class SystemStatsResponse(BaseModel):
    updated_at: datetime
    system: SystemInfoResponse
    resources: SystemResourceStatsResponse
    gpu: SystemGpuStatsResponse
    storage: SystemStorageStatsResponse
