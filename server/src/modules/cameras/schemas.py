from datetime import datetime
from ipaddress import IPv4Address
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic_core import PydanticCustomError

_CAMERA_TEXT_MAX_LENGTH = 255


def _camera_validation_error(message: str) -> PydanticCustomError:
    return PydanticCustomError("camera_validation", message)


def validate_camera_connection_fields(
    *,
    protocol: Literal["rtsp", "http", "usb"],
    host: str | None,
    path: str | None,
    stream_path: str | None,
    device_path: str | None,
) -> None:
    if protocol == "usb":
        if not device_path:
            raise _camera_validation_error("Укажите путь к USB-устройству")
        return

    if not host:
        raise _camera_validation_error("Укажите IPv4-адрес камеры")

    if protocol == "http" and not (stream_path or path):
        raise _camera_validation_error("Укажите путь HTTP-потока")


def _normalize_optional_string(value: str | None) -> str | None:
    if value is None:
        return None

    stripped = value.strip()
    return stripped or None


def _validate_text_length(value: str, *, label: str, max_length: int = _CAMERA_TEXT_MAX_LENGTH) -> str:
    if len(value) > max_length:
        raise _camera_validation_error(
            f"Укажите {label} длиной не более {max_length} символов"
        )
    return value


def _is_valid_camera_host(value: str) -> bool:
    candidate = value.strip()
    if not candidate or any(char.isspace() for char in candidate):
        return False

    if any(char in candidate for char in ("/", "\\", "@", "?", "#")):
        return False

    if "://" in candidate:
        return False

    try:
        IPv4Address(candidate)
        return True
    except ValueError:
        return False


def _is_valid_stream_path(value: str) -> bool:
    if not value or any(char.isspace() for char in value):
        return False

    if "://" in value or "@" in value:
        return False

    return True


def _is_valid_device_path(value: str) -> bool:
    if not value or any(char.isspace() for char in value):
        return False

    if value.isdigit():
        return True

    return value.startswith("/dev/")


class CameraBase(BaseModel):
    name: str
    description: str | None = None
    protocol: Literal["rtsp", "http", "usb"] = "rtsp"
    host: str | None = None
    port: int | None = Field(default=554, ge=1, le=65535)
    path: str | None = None
    stream_path: str | None = None
    device_path: str | None = None
    username: str | None = None
    password: str | None = None
    location: str | None = None
    is_active: bool = True
    timeout_sec: int = Field(default=5, ge=1, le=60)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: str) -> str:
        return value.strip() if isinstance(value, str) else value

    @field_validator(
        "description",
        "host",
        "path",
        "stream_path",
        "device_path",
        "username",
        "password",
        "location",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _normalize_optional_string(value)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        if not value:
            raise _camera_validation_error("Укажите название")
        return _validate_text_length(value, label="название")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_length(value, label="описание")

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _validate_text_length(value, label="IPv4-адрес")
        if not _is_valid_camera_host(value):
            raise _camera_validation_error("Укажите корректный IPv4-адрес")
        return value

    @field_validator("path", "stream_path")
    @classmethod
    def validate_stream_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _validate_text_length(value, label="путь потока")
        if not _is_valid_stream_path(value):
            raise _camera_validation_error("Укажите путь потока без пробелов")
        return value

    @field_validator("device_path")
    @classmethod
    def validate_device_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _validate_text_length(value, label="путь устройства")
        if not _is_valid_device_path(value):
            raise _camera_validation_error(
                "Укажите корректный путь устройства, например /dev/video0"
            )
        return value

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_length(value, label="логин")

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_length(value, label="пароль")

    @field_validator("location")
    @classmethod
    def validate_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_length(value, label="местоположение")

class CameraCreate(CameraBase):
    @model_validator(mode="after")
    def validate_protocol_fields(self) -> Self:
        validate_camera_connection_fields(
            protocol=self.protocol,
            host=self.host,
            path=self.path,
            stream_path=self.stream_path,
            device_path=self.device_path,
        )
        return self


class CameraUpdate(BaseModel):
    name: str | None = None
    description: str | None = None
    protocol: Literal["rtsp", "http", "usb"] | None = None
    host: str | None = None
    port: int | None = Field(default=None, ge=1, le=65535)
    path: str | None = None
    stream_path: str | None = None
    device_path: str | None = None
    username: str | None = None
    password: str | None = None
    location: str | None = None
    is_active: bool | None = None
    timeout_sec: int | None = Field(default=None, ge=1, le=60)

    @field_validator("name", mode="before")
    @classmethod
    def strip_name(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value

    @field_validator(
        "description",
        "host",
        "path",
        "stream_path",
        "device_path",
        "username",
        "password",
        "location",
        mode="before",
    )
    @classmethod
    def normalize_optional_strings(cls, value: str | None) -> str | None:
        return _normalize_optional_string(value)

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str | None) -> str | None:
        if value == "":
            raise _camera_validation_error("Укажите название")
        if value is None:
            return None
        return _validate_text_length(value, label="название")

    @field_validator("description")
    @classmethod
    def validate_description(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_length(value, label="описание")

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _validate_text_length(value, label="IPv4-адрес")
        if not _is_valid_camera_host(value):
            raise _camera_validation_error("Укажите корректный IPv4-адрес")
        return value

    @field_validator("path", "stream_path")
    @classmethod
    def validate_stream_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _validate_text_length(value, label="путь потока")
        if not _is_valid_stream_path(value):
            raise _camera_validation_error("Укажите путь потока без пробелов")
        return value

    @field_validator("device_path")
    @classmethod
    def validate_device_path(cls, value: str | None) -> str | None:
        if value is None:
            return None
        _validate_text_length(value, label="путь устройства")
        if not _is_valid_device_path(value):
            raise _camera_validation_error(
                "Укажите корректный путь устройства, например /dev/video0"
            )
        return value

    @field_validator("username")
    @classmethod
    def validate_username(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_length(value, label="логин")

    @field_validator("password")
    @classmethod
    def validate_password(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_length(value, label="пароль")

    @field_validator("location")
    @classmethod
    def validate_location(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_text_length(value, label="местоположение")

    @model_validator(mode="after")
    def validate_not_empty(self) -> Self:
        if not self.model_dump(exclude_unset=True):
            raise _camera_validation_error("Укажите хотя бы одно поле")
        return self


class CameraResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    name: str
    description: str | None
    protocol: str
    host: str | None
    port: int | None
    path: str | None
    stream_path: str | None
    device_path: str | None
    username: str | None
    location: str | None
    is_active: bool
    timeout_sec: int
    last_status: str | None
    last_checked_at: datetime | None
    last_error: str | None
    created_at: datetime


class CameraTestResponse(BaseModel):
    status: str
    message: str
    checked_at: datetime
    width: int | None = None
    height: int | None = None


class UsbCameraDeviceResponse(BaseModel):
    name: str
    device_path: str
    index: int | None = None
    width: int | None = None
    height: int | None = None
    backend: str | None = None


class CameraWebRTCOfferRequest(BaseModel):
    sdp: str
    type: Literal["offer"] = "offer"
    fps: int = Field(default=20, ge=1, le=30)


class CameraWebRTCOfferResponse(BaseModel):
    sdp: str
    type: str
