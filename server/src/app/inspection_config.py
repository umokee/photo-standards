from __future__ import annotations

from typing import Literal


class InspectionSettingsMixin:
    # Runtime inspection behavior. Detailed matcher thresholds live with the matcher.
    INSPECTION_ALIGNMENT_PROFILE: Literal["balanced", "safe", "debug", "experimental"] = "balanced"
    INSPECTION_DEBUG_PAYLOAD: bool = True
    INSPECTION_FAILSAFE_REQUIRE_CONFIRMED_POSE: bool = True
    INSPECTION_VERIFICATION_MODE: Literal["alignment", "yolo_count"] = "alignment"
    # Multi-reference view selection. When enabled, inspection may use any
    # standard images marked is_reference=True as a reference pool and automatically
    # picks the view with the strongest alignment for the current frame.
    INSPECTION_MULTI_REFERENCE_ENABLED: bool = True
    INSPECTION_MULTI_REFERENCE_MAX_CANDIDATES: int = 8

    # Debug guard for the selected reference view. This does not fail the
    # inspection by itself; it marks weak view selection in debug payload so
    # bad projections are visible instead of silently using best-of-bad.
    INSPECTION_MULTI_REFERENCE_WARN_MIN_INLIERS: int = 8
    INSPECTION_MULTI_REFERENCE_WARN_MIN_INLIER_RATIO: float = 0.08
    INSPECTION_MULTI_REFERENCE_WARN_MAX_MEDIAN_ERROR: float = 25.0
    INSPECTION_MULTI_REFERENCE_WARN_MIN_VISIBLE_RATIO: float = 0.35

    # Production debug tuning. Keep compact summaries in API/history payloads and
    # avoid storing large projected polygons in high-level diagnostic arrays.
    INSPECTION_DEBUG_MAX_REFERENCE_SCORES: int = 8
    INSPECTION_DEBUG_KEEP_POLYGONS_IN_SUMMARIES: bool = False
