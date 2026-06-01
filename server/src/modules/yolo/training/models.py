from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import sqlalchemy
from app.db import Base
from modules.yolo.training.constants import training
from sqlalchemy import (
    CheckConstraint,
    ForeignKey,
    Index,
    Integer,
    String,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from modules.core.groups.models import Group
    from modules.yolo.inspection.models import InspectionResult


class MlModel(Base):
    __tablename__ = "ml_models"
    __table_args__ = (
        Index(
            "uq_group_active_model",
            "group_id",
            unique=True,
            postgresql_where=text("is_active = true"),
        ),
        Index(
            "uq_group_model_version",
            "group_id",
            "version",
            unique=True,
        ),
        CheckConstraint("version > 0", name="version_positive"),
        CheckConstraint("epochs > 0", name="epochs_positive"),
        CheckConstraint(
            f"imgsz IN ({', '.join(map(str, training.image_size))})",
            name="imgsz_allowed",
        ),
        CheckConstraint("batch_size > 0", name="batch_size_positive"),
        CheckConstraint("num_classes > 0", name="num_classes_positive"),
        CheckConstraint("train_ratio BETWEEN 0 AND 100", name="train_ratio_range"),
        CheckConstraint("val_ratio BETWEEN 0 AND 100", name="val_ratio_range"),
        CheckConstraint("test_ratio BETWEEN 0 AND 100", name="test_ratio_range"),
        CheckConstraint("total_images >= 0", name="total_images_positive"),
        CheckConstraint("train_count >= 0", name="train_count_positive"),
        CheckConstraint("val_count >= 0", name="val_count_positive"),
        CheckConstraint("test_count >= 0", name="test_count_positive"),
    )

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    group_id: Mapped[UUID] = mapped_column(
        ForeignKey("groups.id", ondelete="CASCADE"),
        index=True,
    )

    architecture: Mapped[str] = mapped_column(
        sqlalchemy.Enum(*training.architectures, name="architecture_enum"),
        default=training.architectures.default,
    )
    weights_path: Mapped[str | None] = mapped_column(String(500), default=None)
    version: Mapped[int | None] = mapped_column(Integer, default=None)

    epochs: Mapped[int | None] = mapped_column(Integer, default=None)
    imgsz: Mapped[int] = mapped_column(Integer, default=training.image_size.default)
    batch_size: Mapped[int | None] = mapped_column(Integer, default=None)

    num_classes: Mapped[int | None] = mapped_column(Integer, default=None)
    class_keys: Mapped[list | None] = mapped_column(JSONB, default=None)
    class_meta: Mapped[list | None] = mapped_column(JSONB, default=None)
    metrics: Mapped[dict | None] = mapped_column(JSONB, default=None)

    train_ratio: Mapped[int | None] = mapped_column(Integer(), default=None)
    val_ratio: Mapped[int | None] = mapped_column(Integer(), default=None)
    test_ratio: Mapped[int | None] = mapped_column(Integer(), default=None)

    total_images: Mapped[int | None] = mapped_column(Integer(), default=None)
    train_count: Mapped[int | None] = mapped_column(Integer(), default=None)
    val_count: Mapped[int | None] = mapped_column(Integer(), default=None)
    test_count: Mapped[int | None] = mapped_column(Integer(), default=None)

    is_active: Mapped[bool] = mapped_column(default=False)
    trained_at: Mapped[datetime | None] = mapped_column(default=None)
    created_at: Mapped[datetime] = mapped_column(server_default=func.now())

    group: Mapped[Group] = relationship("Group", back_populates="ml_models")
    inspections: Mapped[list[InspectionResult]] = relationship(
        "InspectionResult",
        back_populates="ml_model",
    )
