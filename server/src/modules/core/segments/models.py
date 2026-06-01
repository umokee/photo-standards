from __future__ import annotations

from typing import TYPE_CHECKING
from uuid import UUID, uuid4

from app.db import Base
from modules.core.segments.constants import segments
from sqlalchemy import (
    JSON,
    CheckConstraint,
    ForeignKey,
    Integer,
    String,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from modules.core.groups.models import Group
    from modules.core.standards.models import StandardImage


class SegmentClassGroup(Base):
    __tablename__ = "segment_class_groups"
    __table_args__ = (
        UniqueConstraint("group_id", "name", name="uq_group_segment_class_group_name"),
    )

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    group_id: Mapped[UUID] = mapped_column(
        ForeignKey("groups.id", ondelete="CASCADE"),
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255))

    group: Mapped[Group] = relationship(back_populates="segment_class_groups")
    segment_classes: Mapped[list[SegmentClass]] = relationship(
        back_populates="class_group"
    )


class SegmentClass(Base):
    __tablename__ = "segment_classes"
    __table_args__ = (
        UniqueConstraint("group_id", "name", name="uq_group_segment_class_name"),
        CheckConstraint(
            f"hue BETWEEN {segments.hue.min} AND {segments.hue.max}",
            name="segment_class_hue_range",
        ),
    )

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    group_id: Mapped[UUID] = mapped_column(
        ForeignKey("groups.id", ondelete="CASCADE"),
        index=True,
    )
    class_group_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("segment_class_groups.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )
    name: Mapped[str] = mapped_column(String(255))
    hue: Mapped[int] = mapped_column(Integer, default=segments.hue.default)

    group: Mapped[Group] = relationship(back_populates="segment_classes")
    class_group: Mapped[SegmentClassGroup | None] = relationship(
        back_populates="segment_classes"
    )
    annotations: Mapped[list[SegmentAnnotation]] = relationship(
        back_populates="segment_class",
        cascade="all, delete-orphan",
    )


class SegmentAnnotation(Base):
    __tablename__ = "segment_annotations"
    __table_args__ = (
        UniqueConstraint(
            "image_id",
            "segment_class_id",
            name="uq_image_segment_class_annotation",
        ),
    )

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    image_id: Mapped[UUID] = mapped_column(
        ForeignKey("standard_images.id", ondelete="CASCADE"),
        index=True,
    )
    segment_class_id: Mapped[UUID] = mapped_column(
        ForeignKey("segment_classes.id", ondelete="CASCADE"),
        index=True,
    )
    points: Mapped[list] = mapped_column(JSON, default=list)

    image: Mapped[StandardImage] = relationship(back_populates="annotations")
    segment_class: Mapped[SegmentClass] = relationship(back_populates="annotations")
