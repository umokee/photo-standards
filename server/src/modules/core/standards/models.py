from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import sqlalchemy
from app.db import Base
from modules.core.standards.constants import standards
from sqlalchemy import ForeignKey, Integer, String, func
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from modules.core.groups.models import Group
    from modules.core.segments.models import SegmentAnnotation
    from modules.yolo.inspection.models import InspectionResult


class Standard(Base):
    __tablename__ = "standards"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    group_id: Mapped[UUID] = mapped_column(
        ForeignKey("groups.id", ondelete="CASCADE"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    angle: Mapped[str | None] = mapped_column(
        sqlalchemy.Enum(*standards.angles, name="angle_enum"),
        default=None,
    )
    is_active: Mapped[bool] = mapped_column(default=True)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    group: Mapped[Group] = relationship(back_populates="standards")
    images: Mapped[list[StandardImage]] = relationship(
        back_populates="standard",
        cascade="all, delete-orphan",
    )
    inspections: Mapped[list[InspectionResult]] = relationship(
        back_populates="standard"
    )


class StandardImage(Base):
    __tablename__ = "standard_images"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    standard_id: Mapped[UUID] = mapped_column(
        ForeignKey("standards.id", ondelete="CASCADE"),
        index=True,
    )
    image_path: Mapped[str] = mapped_column(String(500))
    is_reference: Mapped[bool] = mapped_column(default=False)

    features_path: Mapped[str | None] = mapped_column(String(500), default=None)
    features_keypoint_count: Mapped[int | None] = mapped_column(Integer, default=None)
    features_computed_at: Mapped[datetime | None] = mapped_column(default=None)

    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    standard: Mapped[Standard] = relationship(back_populates="images")
    annotations: Mapped[list[SegmentAnnotation]] = relationship(
        back_populates="image",
        cascade="all, delete-orphan",
    )
