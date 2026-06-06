from __future__ import annotations

from modules.yolo.inspection.domain.alignment import (
    LocalProjectionData,
    project_polygon,
)
from modules.yolo.inspection.domain.matcher_context import (
    _missing_polygon_refinement_points,
    _missing_rescue_overlaps_other_expected,
)
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area_similarity,
    bbox_center_distance_factor,
    bbox_containment,
    bbox_from_polygon,
    is_visible_in_frame,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    ExpectedSlot,
    ProjectedExpected,
    ProjectionCandidate,
    ProjectionValidationResult,
)
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds

_MULTI_EXPECTED_SLOT_MIN_GLOBAL_AREA_SCORE = 0.68
_MULTI_EXPECTED_SLOT_MAX_GLOBAL_CENTER_FACTOR = 0.28


def validate_projection_candidate_shape(
    candidate: ProjectionCandidate,
) -> ProjectionValidationResult:
    if candidate.polygon is None or candidate.bbox is None:
        return ProjectionValidationResult(
            accepted=False,
            reject_reason="no_visible_fallback_polygon",
        )
    if len(candidate.polygon) < 3:
        return ProjectionValidationResult(
            accepted=False,
            reject_reason="invalid_polygon",
        )
    return ProjectionValidationResult(accepted=True)


def validate_translated_global_candidate(
    expected_item: ProjectedExpected,
    *,
    polygon: list[list[float]],
    bbox: BBox | None,
    global_bbox: BBox,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    max_center_factor: float,
    min_search_containment: float | None = None,
    max_other_overlap: float | None = None,
    all_expected: list[ProjectedExpected] | None = None,
) -> ProjectionValidationResult:
    if len(polygon) < 3:
        return ProjectionValidationResult(False, "invalid_polygon")
    if bbox is None:
        return ProjectionValidationResult(False, "invalid_bbox")
    if projection_data is not None and projection_data.frame_size is not None:
        if not is_visible_in_frame(
            bbox,
            projection_data.frame_size,
            min_visible_fraction=0.05,
        ):
            return ProjectionValidationResult(False, "outside_frame")

    area_score = bbox_area_similarity(bbox, global_bbox)
    if area_score < 0.92:
        return ProjectionValidationResult(False, "area_changed")

    center_factor = bbox_center_distance_factor(bbox, global_bbox)
    if center_factor > max_center_factor:
        return ProjectionValidationResult(False, "center_far_from_global")

    if min_search_containment is not None and slot is not None:
        if bbox_containment(bbox, slot.search_bbox) < min_search_containment:
            return ProjectionValidationResult(False, "low_search_containment")

    if max_other_overlap is not None and _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=bbox,
        all_expected=all_expected,
        max_overlap=float(max_other_overlap),
    ):
        return ProjectionValidationResult(False, "bad_overlap")

    return ProjectionValidationResult(True)


def _missing_global_projection(
    expected_item: ProjectedExpected,
    *,
    projection_data: LocalProjectionData | None,
) -> tuple[list[list[float]], BBox] | None:
    if projection_data is None or projection_data.global_homography is None:
        return None

    try:
        polygon = project_polygon(
            expected_item.item.reference_polygon,
            projection_data.global_homography,
        )
    except Exception:
        return None

    if len(polygon) < 3:
        return None

    polygon = [[float(point[0]), float(point[1])] for point in polygon]
    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return None

    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
    ):
        return None

    reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is not None:
        global_area_score = bbox_area_similarity(bbox, reference_bbox)
        if global_area_score < _thresholds.missing_fallback_min_global_area_score:
            return None

    return polygon, bbox


def _missing_expected_slot_has_global_consensus(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    projection_data: LocalProjectionData,
    fallback_bbox: BBox,
    all_expected: list[ProjectedExpected] | None,
    strong_support: int,
) -> bool:
    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return False

    _, global_bbox = global_projection
    area_score = bbox_area_similarity(fallback_bbox, global_bbox)
    center_factor = bbox_center_distance_factor(fallback_bbox, global_bbox)

    min_area_score = _MULTI_EXPECTED_SLOT_MIN_GLOBAL_AREA_SCORE
    max_center_factor = _MULTI_EXPECTED_SLOT_MAX_GLOBAL_CENTER_FACTOR

    if slot.feature_support >= strong_support:
        min_area_score = max(min_area_score, 0.78)
        max_center_factor = min(max_center_factor, 0.18)

    if area_score < min_area_score or center_factor > max_center_factor:
        return False

    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=fallback_bbox,
        all_expected=all_expected,
    ):
        return False

    return True


def _missing_fallback_projection_unsafe_reason(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    fallback_bbox: BBox | None,
    context_rescue_used: bool = False,
    fallback_source: str = "expected_slot",
    all_expected: list[ProjectedExpected] | None = None,
) -> str | None:
    if fallback_bbox is None:
        return "no_visible_fallback_polygon"
    if slot is None or projection_data is None:
        return None

    is_multi_object_context = all_expected is not None and len(all_expected) > 1
    min_support = max(3, _thresholds.missing_polygon_min_feature_support)
    strong_support = max(min_support * 3, 12)

    if is_multi_object_context and fallback_source == "expected_slot":
        if not _missing_expected_slot_has_global_consensus(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            fallback_bbox=fallback_bbox,
            all_expected=all_expected,
            strong_support=strong_support,
        ):
            return "multi_expected_slot_without_global_consensus"

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)

    if reference_bbox is not None:
        reference_area_score = bbox_area_similarity(fallback_bbox, reference_bbox)
        if reference_area_score < 0.12:
            return "fallback_area_not_reference_like"

    if slot.feature_support < strong_support:
        return None

    if context_rescue_used:
        return None

    local_reference, _ = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )

    if len(local_reference) >= strong_support:
        return "rich_context_but_no_safe_transform"

    return None
