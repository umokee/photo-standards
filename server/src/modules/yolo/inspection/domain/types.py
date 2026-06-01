from __future__ import annotations

from dataclasses import dataclass
from uuid import UUID


@dataclass(slots=True)
class YoloDetection:
    class_key: str
    confidence: float
    bbox: dict[str, float]
    polygon: list[list[float]] | None = None


@dataclass(slots=True)
class ExpectedSegment:
    annotation_id: UUID
    segment_class_id: UUID
    class_key: str
    name: str
    hue: int | None
    reference_polygon: list[list[float]]


@dataclass(slots=True)
class SegmentMatch:
    annotation_id: UUID | None
    segment_class_id: UUID | None
    class_key: str
    name: str
    hue: int | None
    status: str
    iou: float | None
    confidence: float | None
    expected_polygon: list[list[float]] | None
    detected_polygon: list[list[float]] | None
    detected_bbox: dict[str, float] | None
    detected_class_in_zone: str | None = None
