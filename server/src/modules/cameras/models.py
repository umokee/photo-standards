from datetime import datetime
from uuid import UUID, uuid4

from app.db import Base
from sqlalchemy import Enum, Integer, String, Text, func
from sqlalchemy.orm import Mapped, mapped_column


class Camera(Base):
    __tablename__ = "cameras"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    description: Mapped[str | None] = mapped_column(Text, default=None)

    protocol: Mapped[str] = mapped_column(
        Enum(*["rtsp", "http", "usb"], name="camera_protocol_enum"),
        default="rtsp",
    )
    host: Mapped[str | None] = mapped_column(String(255), nullable=True)
    port: Mapped[int | None] = mapped_column(Integer, default=554)
    path: Mapped[str | None] = mapped_column(String(255), default=None)
    stream_path: Mapped[str | None] = mapped_column(String(255), default=None)
    device_path: Mapped[str | None] = mapped_column(String(255), default=None)

    username: Mapped[str | None] = mapped_column(String(255), default=None)
    password: Mapped[str | None] = mapped_column(String(255), default=None)

    location: Mapped[str | None] = mapped_column(String(255), default=None)
    is_active: Mapped[bool] = mapped_column(default=True)
    timeout_sec: Mapped[int] = mapped_column(Integer, default=5)

    last_status: Mapped[str | None] = mapped_column(
        Enum(*["online", "offline", "unknown"], name="camera_status_enum"),
        default="unknown",
    )
    last_checked_at: Mapped[datetime | None] = mapped_column(default=None)
    last_error: Mapped[str | None] = mapped_column(String(500), default=None)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())
