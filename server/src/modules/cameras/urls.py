from __future__ import annotations

from .models import Camera


def build_camera_stream_url(camera: Camera) -> str:
    if camera.protocol == "usb":
        return camera.device_path or ""

    auth = ""
    if camera.username:
        auth = camera.username
        if camera.password:
            auth += f":{camera.password}"
        auth += "@"

    port_part = f":{camera.port}" if camera.port else ""
    selected_path = camera.stream_path or camera.path or ""
    if selected_path and not selected_path.startswith("/"):
        selected_path = "/" + selected_path

    return f"{camera.protocol}://{auth}{camera.host}{port_part}{selected_path}"


def build_camera_stream_url_no_auth(camera: Camera) -> str:
    if camera.protocol == "usb":
        return camera.device_path or ""

    port_part = f":{camera.port}" if camera.port else ""

    selected_path = camera.stream_path or camera.path or ""
    if selected_path and not selected_path.startswith("/"):
        selected_path = "/" + selected_path

    return f"{camera.protocol}://{camera.host}{port_part}{selected_path}"
