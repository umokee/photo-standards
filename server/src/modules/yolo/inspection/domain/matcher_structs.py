from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from modules.yolo.inspection.domain.types import ExpectedSegment, YoloDetection

BBox = tuple[float, float, float, float]


@dataclass(slots=True)
class ProjectedExpected:
    index: int
    item: ExpectedSegment
    polygon: list[list[float]]
    bbox: BBox


@dataclass(slots=True)
class DetectionCandidate:
    index: int
    detection: YoloDetection
    polygon: list[list[float]]
    bbox: BBox


@dataclass(slots=True)
class ExpectedSlot:
    projected_bbox: BBox
    search_bbox: BBox
    reference_bbox: BBox | None
    feature_support: int
    feature_total: int


@dataclass(slots=True)
class MissingPolygonRefinement:
    polygon: list[list[float]]
    bbox: BBox
    feature_support: int
    feature_total: int
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    reference_spread: float
    frame_spread: float
    quadrant_count: int
    center_drift_factor: float
    area_score: float


@dataclass(slots=True)
class MissingFallbackProjection:
    polygon: list[list[float]] | None
    bbox: BBox | None
    hidden: bool
    debug: dict[str, Any]


@dataclass(slots=True)
class MissingFallbackState:
    polygon: list[list[float]] | None
    bbox: BBox | None
    debug: dict[str, Any]
    context_rescue_used: bool = False
    source: str = "expected_slot"
    global_candidate_bbox: BBox | None = None


@dataclass(slots=True)
class MissingTranslationRescue:
    polygon: list[list[float]]
    bbox: BBox
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    shift_x: float
    shift_y: float
    shift_factor: float
    source_counts: dict[str, int] | None = None


@dataclass(slots=True)
class MissingLocalDisplacement:
    polygon: list[list[float]]
    bbox: BBox
    area_score: float
    center_drift_factor: float


@dataclass(slots=True)
class TrustedAnchor:
    expected_index: int
    source: str
    residual: np.ndarray
    global_bbox: BBox
    local_bbox: BBox
    weight: float
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    shift_factor: float


@dataclass(slots=True)
class AnchorReleaseConsensus:
    shift: np.ndarray
    inlier_mask: np.ndarray
    candidate_count: int
    inlier_count: int
    inlier_ratio: float
    median_error: float
    dispersion: float


@dataclass(slots=True)
class AnchorOverlapDecision:
    allowed: bool
    reason: str | None
    overlap_count: int
    unresolved_count: int
    strong_anchor_count: int
    max_overlap: float
    max_resolved_overlap: float


@dataclass(slots=True)
class ProjectedExpectedBuildResult:
    projected: list[ProjectedExpected]
    unprojected: list[tuple[int, ExpectedSegment, dict[str, Any]]]
