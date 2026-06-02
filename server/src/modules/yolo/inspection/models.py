from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID, uuid4

import sqlalchemy
from app.db import Base
from modules.core.segments.constants import segments
from modules.yolo.inspection.constants import inspections
from sqlalchemy import CheckConstraint, ForeignKey, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship

if TYPE_CHECKING:
    from modules.cameras.models import Camera
    from modules.core.standards.models import Standard
    from modules.users.models import User
    from modules.yolo.training.models import MlModel


class InspectionResult(Base):
    __tablename__ = "inspection_results"

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    standard_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("standards.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )
    model_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("ml_models.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )
    camera_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("cameras.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )
    user_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("users.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )

    mode: Mapped[str] = mapped_column(
        sqlalchemy.Enum(*inspections.modes, name="inspection_mode_enum"),
        default=inspections.modes.default,
    )
    status: Mapped[str] = mapped_column(
        sqlalchemy.Enum(*inspections.statuses, name="inspection_status_enum"),
        default=inspections.statuses.default,
    )

    image_path: Mapped[str] = mapped_column(String(500))
    result_image_path: Mapped[str | None] = mapped_column(String(500), default=None)

    total_segments: Mapped[int] = mapped_column()
    matched_segments: Mapped[int] = mapped_column()

    alignment_status: Mapped[str | None] = mapped_column(
        sqlalchemy.Enum(
            *[
                "success",
                "insufficient_matches",
                "insufficient_inliers",
                "homography_failed",
            ],
            name="alignment_status_enum",
        ),
        default=None,
    )
    alignment_inlier_count: Mapped[int | None] = mapped_column(Integer, default=None)
    alignment_raw_match_count: Mapped[int | None] = mapped_column(Integer, default=None)
    homography: Mapped[list | None] = mapped_column(JSONB, default=None)

    notes: Mapped[str | None] = mapped_column(Text, default=None)
    debug_payload: Mapped[dict | None] = mapped_column(JSONB, default=None)

    inspected_at: Mapped[datetime] = mapped_column(
        server_default=func.now(),
        index=True,
    )

    standard: Mapped[Standard | None] = relationship(
        "Standard",
        back_populates="inspections",
    )
    ml_model: Mapped[MlModel | None] = relationship(
        "MlModel",
        back_populates="inspections",
    )
    camera: Mapped[Camera | None] = relationship("Camera")
    user: Mapped[User | None] = relationship("User")
    segment_results: Mapped[list[InspectionSegmentResult]] = relationship(
        "InspectionSegmentResult",
        back_populates="inspection",
        cascade="all, delete-orphan",
    )


class InspectionSegmentResult(Base):
    __tablename__ = "inspection_segment_results"
    __table_args__ = (
        CheckConstraint(
            f"hue IS NULL OR hue BETWEEN {segments.hue.min} AND {segments.hue.max}",
            name="hue_range",
        ),
        CheckConstraint(
            "iou IS NULL OR (iou >= 0 AND iou <= 1)",
            name="iou_range",
        ),
    )

    id: Mapped[UUID] = mapped_column(default=uuid4, primary_key=True, index=True)
    inspection_id: Mapped[UUID] = mapped_column(
        ForeignKey("inspection_results.id", ondelete="CASCADE"),
        index=True,
    )
    segment_annotation_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("segment_annotations.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )
    segment_class_id: Mapped[UUID | None] = mapped_column(
        ForeignKey("segment_classes.id", ondelete="SET NULL"),
        default=None,
        index=True,
    )

    class_key: Mapped[str] = mapped_column(String(255), index=True)
    name: Mapped[str] = mapped_column(String(255))
    hue: Mapped[int] = mapped_column(Integer, default=segments.hue.default)

    status: Mapped[str] = mapped_column(
        sqlalchemy.Enum(
            *["ok", "missing", "extra", "unmatched"],
            name="segment_result_status_enum",
        ),
    )
    iou: Mapped[float | None] = mapped_column(default=None)
    confidence: Mapped[float | None] = mapped_column(default=None)

    expected_polygon: Mapped[list | None] = mapped_column(JSONB, default=None)
    detected_polygon: Mapped[list | None] = mapped_column(JSONB, default=None)
    detected_bbox: Mapped[dict | None] = mapped_column(JSONB, default=None)

    inspection: Mapped[InspectionResult] = relationship(
        "InspectionResult",
        back_populates="segment_results",
    )
