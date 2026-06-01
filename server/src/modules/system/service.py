from __future__ import annotations

import os
import shutil
import socket
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import psutil
from app.config import settings

from .schemas import (
    SystemGpuStatsResponse,
    SystemInfoResponse,
    SystemResourceStatsResponse,
    SystemStatsResponse,
    SystemStorageCategoriesResponse,
    SystemStorageStatsResponse,
)

STARTED_AT = time.time()

psutil.cpu_percent(interval=None)


async def get_system_stats() -> SystemStatsResponse:
    memory = psutil.virtual_memory()
    disk = shutil.disk_usage(settings.STORAGE_ROOT)

    return SystemStatsResponse(
        updated_at=datetime.now(UTC),
        system=SystemInfoResponse(
            hostname=socket.gethostname(),
            uptime_sec=int(time.time() - STARTED_AT),
        ),
        resources=SystemResourceStatsResponse(
            cpu_percent=round(psutil.cpu_percent(interval=None), 1),
            cpu_count_logical=psutil.cpu_count(logical=True) or 0,
            memory_used_bytes=int(memory.used),
            memory_total_bytes=int(memory.total),
            disk_used_bytes=int(disk.used),
            disk_total_bytes=int(disk.total),
        ),
        gpu=_read_gpu_stats(),
        storage=_build_storage_stats(),
    )


def _build_storage_stats() -> SystemStorageStatsResponse:
    standards = _get_dir_size(settings.standards_storage_path)
    inspections = _get_dir_size(settings.inspections_storage_path)
    models = _get_dir_size(settings.models_storage_path)
    logs = _get_dir_size(settings.logs_storage_path)
    total = _get_dir_size(settings.STORAGE_ROOT)

    known = standards + inspections + models + logs
    other = max(0, total - known)

    return SystemStorageStatsResponse(
        used_bytes=total,
        categories=SystemStorageCategoriesResponse(
            standards_bytes=standards,
            inspections_bytes=inspections,
            models_bytes=models,
            logs_bytes=logs,
            other_bytes=other,
        ),
    )


def _get_dir_size(path: Path) -> int:
    if not path.exists():
        return 0

    total = 0
    for root, _dirs, files in os.walk(path):
        root_path = Path(root)
        for filename in files:
            file_path = root_path / filename
            try:
                total += file_path.stat().st_size
            except OSError:
                continue

    return total


def _read_gpu_stats() -> SystemGpuStatsResponse:
    nvidia_smi = shutil.which("nvidia-smi")
    if not nvidia_smi:
        return SystemGpuStatsResponse(available=False)

    try:
        result = subprocess.run(  # noqa: S603
            [
                nvidia_smi,
                "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=2,
            check=True,
        )
    except subprocess.SubprocessError:
        return SystemGpuStatsResponse(available=False)

    first = result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""
    if not first:
        return SystemGpuStatsResponse(available=False)

    parts = [item.strip() for item in first.split(",")]
    if len(parts) < 5:
        return SystemGpuStatsResponse(available=False)

    return SystemGpuStatsResponse(
        available=True,
        name=parts[0],
        utilization_percent=_parse_int(parts[1]),
        memory_used_mb=_parse_int(parts[2]),
        memory_total_mb=_parse_int(parts[3]),
        temperature_c=_parse_int(parts[4]),
    )


def _parse_int(value: str) -> int | None:
    try:
        return int(value.strip())
    except (TypeError, ValueError):
        return None
