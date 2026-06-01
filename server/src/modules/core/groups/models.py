from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from app.db import Base
from sqlalchemy import String, Text, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from modules.core.segments.models import SegmentClass, SegmentClassGroup
    from modules.core.standards.models import Standard
    from modules.yolo.training.models import MlModel


class Group(Base):
    __tablename__ = "groups"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    name: Mapped[str] = mapped_column(String(255), unique=True)
    description: Mapped[str | None] = mapped_column(Text, default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    standards: Mapped[list[Standard]] = relationship(
        back_populates="group",
        cascade="all, delete-orphan",
    )
    ml_models: Mapped[list[MlModel]] = relationship(
        back_populates="group",
        cascade="all, delete-orphan",
    )
    segment_class_groups: Mapped[list[SegmentClassGroup]] = relationship(
        back_populates="group",
        cascade="all, delete-orphan",
    )
    segment_classes: Mapped[list[SegmentClass]] = relationship(
        back_populates="group",
        cascade="all, delete-orphan",
    )
