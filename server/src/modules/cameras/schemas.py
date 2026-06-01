from datetime import datetime
from typing import Literal, Self
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, model_validator


class CameraBase(BaseModel):
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = None
    protocol: Literal["rtsp", "http", "usb"] = "rtsp"
    host: str | None = Field(default=None, min_length=1, max_length=255)
    port: int | None = Field(default=554, ge=1, le=65535)
    path: str | None = Field(default=None, min_length=1, max_length=255)
    stream_path: str | None = Field(default=None, min_length=1, max_length=255)
    device_path: str | None = Field(default=None, min_length=1, max_length=255)
    username: str | None = Field(default=None, min_length=1, max_length=255)
    password: str | None = Field(default=None, min_length=1, max_length=255)
    location: str | None = Field(default=None, min_length=1, max_length=255)
    is_active: bool = True
    timeout_sec: int = Field(default=5, ge=1, le=60)


class CameraCreate(CameraBase):
    @model_validator(mode="after")
    def validate_protocol_fields(self) -> Self:
        if self.protocol == "usb":
            if not self.device_path:
                raise ValueError("Для USB-камеры обязателен device_path")
        else:
            if not self.host:
                raise ValueError(f"Для {self.protocol.upper()}-камеры обязателен host")
        return self


class CameraUpdate(BaseModel):
    name: str | None = Field(default=None, min_length=1, max_length=255)
    description: str | None = None
    protocol: Literal["rtsp", "http", "usb"] | None = None
    host: str | None = Field(default=None, min_length=1, max_length=255)
    port: int | None = Field(default=None, ge=1, le=65535)
    path: str | None = Field(default=None, min_length=1, max_length=255)
    stream_path: str | None = Field(default=None, min_length=1, max_length=255)
    device_path: str | None = Field(default=None, min_length=1, max_length=255)
    username: str | None = Field(default=None, min_length=1, max_length=255)
    password: str | None = Field(default=None, min_length=1, max_length=255)
    location: str | None = Field(default=None, min_length=1, max_length=255)
    is_active: bool | None = None
    timeout_sec: int | None = Field(default=None, ge=1, le=60)

    @model_validator(mode="after")
    def validate_not_empty(self) -> Self:
        if not self.model_dump(exclude_unset=True):
            raise ValueError("Необходимо передать хотя бы одно поле")
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
