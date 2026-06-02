from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field


class InspectionStartResponse(BaseModel):
    kind: Literal["task", "session"]
    task_id: UUID | None = None
    session_id: UUID | None = None
    status: str
    message: str


class InspectionSegmentDetailResponse(BaseModel):
    annotation_id: UUID | str | None = None
    segment_class_id: UUID | str | None = None
    class_key: str
    name: str
    hue: int | None = None

    status: str
    iou: float | None = None
    confidence: float | None = None

    expected_polygon: list[list[float]] | None = None
    detected_polygon: list[list[float]] | None = None
    detected_bbox: dict[str, float] | None = None
    debug: dict[str, Any] | None = None


class InspectionTaskResultResponse(BaseModel):
    task_id: UUID
    inspection_id: UUID | None = None
    status: str
    passed: bool
    matched: int
    total: int
    missing: list[str]

    alignment_status: str | None = None
    alignment_inlier_count: int | None = None
    alignment_raw_match_count: int | None = None
    homography: list[list[float]] | None = None

    details: list[InspectionSegmentDetailResponse]
    mode: str
    model_name: str | None = None
    image_path: str
    result_image_path: str | None = None


class InspectionSaveRequest(BaseModel):
    task_id: UUID
    notes: str | None = None


class InspectionSaveResponse(BaseModel):
    inspection_id: UUID
    status: str
    message: str


class InspectionSegmentResultResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: UUID
    segment_annotation_id: UUID | None
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
    debug: dict[str, Any] | None = None


class InspectionHistoryItemResponse(BaseModel):
    id: UUID
    group_id: UUID | None = None
    standard_id: UUID | None = None
    standard_name: str | None = None
    standard_reference_path: str | None = None
    model_id: UUID | None = None
    model_name: str | None = None
    camera_id: UUID | None = None
    camera_name: str | None = None

    mode: str
    status: str

    image_path: str
    result_image_path: str | None = None

    total_segments: int
    matched_segments: int

    notes: str | None = None
    inspected_at: datetime


class InspectionResultResponse(BaseModel):
    id: UUID
    group_id: UUID | None = None
    standard_id: UUID | None
    standard_name: str | None = None
    standard_reference_path: str | None = None
    model_id: UUID | None
    model_name: str | None = None
    camera_id: UUID | None
    camera_name: str | None = None
    user_id: UUID | None
    user_name: str | None = None

    mode: str
    status: str

    image_path: str
    result_image_path: str | None

    total_segments: int
    matched_segments: int

    alignment_status: str | None
    alignment_inlier_count: int | None
    alignment_raw_match_count: int | None
    homography: list[list[float]] | None

    notes: str | None
    debug_payload: dict | None
    inspected_at: datetime

    segment_results: list[InspectionSegmentResultResponse] = Field(default_factory=list)


class InspectionRealtimeStatusResponse(BaseModel):
    state: str
    matched: int
    total: int
    missing: list[str]
    status: str
    passed: bool
    alignment_status: str
    alignment_inlier_count: int | None = None
    alignment_raw_match_count: int | None = None
    captured_at: datetime | None = None
    details: list[InspectionSegmentDetailResponse] = Field(default_factory=list)


class InspectionRealtimeSnapshotSaveRequest(BaseModel):
    notes: str | None = None


class InspectionWebRTCOfferRequest(BaseModel):
    sdp: str
    type: Literal["offer"] = "offer"
    fps: int = Field(default=15, ge=1, le=30)
    receive_video: bool = True


class InspectionWebRTCOfferResponse(BaseModel):
    sdp: str
    type: str
