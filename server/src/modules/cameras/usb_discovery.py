from __future__ import annotations

import platform
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import cv2


@dataclass(slots=True)
class UsbCameraDevice:
    name: str
    device_path: str
    index: int | None = None
    width: int | None = None
    height: int | None = None
    backend: str | None = None


def discover_usb_cameras(*, max_devices: int = 10) -> list[UsbCameraDevice]:
    system = platform.system()

    if system == "Linux":
        return _discover_linux_usb_cameras()

    return _discover_indexed_usb_cameras(max_devices=max_devices)


def _discover_linux_usb_cameras() -> list[UsbCameraDevice]:
    devices: list[UsbCameraDevice] = []

    paths = sorted(
        Path("/dev").glob("video*"),
        key=lambda p: _extract_index(p.name),
    )

    for path in paths:
        probed = _probe_device(str(path), label=path.name)
        if probed is not None:
            devices.append(probed)

    return devices


def _discover_indexed_usb_cameras(*, max_devices: int) -> list[UsbCameraDevice]:
    system = platform.system()
    names = _get_windows_camera_names() if system == "Windows" else []

    devices: list[UsbCameraDevice] = []

    for index in range(max_devices):
        label = names[index] if index < len(names) else f"USB камера {index}"
        probed = _probe_device(index, label=label)
        if probed is None:
            continue

        probed.device_path = str(index)
        probed.index = index
        devices.append(probed)

    return devices


def _probe_device(device: int | str, *, label: str) -> UsbCameraDevice | None:
    system = platform.system()

    if isinstance(device, int) and system == "Windows":
        capture = cv2.VideoCapture(device, cv2.CAP_DSHOW)
        backend = "dshow"
    else:
        capture = cv2.VideoCapture(device)
        backend = None

    try:
        if not capture.isOpened():
            return None

        ok = False
        frame = None
        for _ in range(3):
            ok, frame = capture.read()
            if ok and frame is not None and frame.size > 0:
                break

        if not ok or frame is None or frame.size == 0:
            return None

        height, width = frame.shape[:2]

        return UsbCameraDevice(
            name=_build_display_name(
                label=label, device=device, width=width, height=height
            ),
            device_path=str(device),
            index=device if isinstance(device, int) else None,
            width=int(width),
            height=int(height),
            backend=backend,
        )
    finally:
        capture.release()


def _get_windows_camera_names() -> list[str]:
    if platform.system() != "Windows":
        return []

    command = [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-Command",
        (
            "Get-CimInstance Win32_PnPEntity | "
            "Where-Object { "
            "$_.PNPClass -eq 'Camera' -or "
            "$_.PNPClass -eq 'Image' -or "
            "$_.Name -match 'camera|webcam|usb video' "
            "} | "
            "Select-Object -ExpandProperty Name"
        ),
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except Exception:
        return []

    if result.returncode != 0:
        return []

    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    return _unique(names)


def _build_display_name(
    *,
    label: str,
    device: int | str,
    width: int,
    height: int,
) -> str:
    return f"{label} · {device} · {width}×{height}"


def _extract_index(value: str) -> int:
    match = re.search(r"\d+", value)
    return int(match.group(0)) if match else 10_000


def _unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []

    for value in values:
        if value in seen:
            continue
        seen.add(value)
        result.append(value)

    return result
