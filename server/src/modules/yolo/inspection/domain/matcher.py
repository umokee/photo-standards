from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from app.config import settings
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds
from modules.core.standards.reference_constants import IOU_MATCH_THRESHOLD
from modules.yolo.inspection.domain.alignment import (
    LocalProjectionData,
    project_polygon,
    project_polygon_adaptive,
)
from modules.yolo.inspection.domain.types import (
    ExpectedSegment,
    SegmentMatch,
    YoloDetection,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    AnchorOverlapDecision,
    AnchorReleaseConsensus,
    DetectionCandidate,
    ExpectedSlot,
    MissingFallbackProjection,
    MissingFallbackState,
    MissingLocalDisplacement,
    MissingPolygonRefinement,
    MissingTranslationRescue,
    ProjectedExpected,
    ProjectedExpectedBuildResult,
    TrustedAnchor,
)
from modules.yolo.inspection.domain.matcher_anchor_diagnostics import (
    set_anchor_release_reject,
    trusted_anchor_source_counts,
    update_anchor_release_debug,
)
from modules.yolo.inspection.domain.matcher_missing_candidates import (
    append_missing_projection_candidate,
    finalize_missing_candidate_agreement_debug,
    max_overlap_with_other_expected,
    reference_area_score_for_expected,
)
from modules.yolo.inspection.domain.matcher_runtime_fusion import (
    build_runtime_yolo_anchor_trace,
    merge_runtime_yolo_anchor_trace_debug,
    merge_runtime_yolo_no_anchor_debug,
    record_runtime_yolo_anchor_trace_candidate,
    runtime_yolo_anchor_candidate_details,
    try_runtime_yolo_anchor_candidate,
)
from modules.yolo.inspection.domain.matcher_geometry import (
    as_match_points,
    affine_reprojection_median_error,
    bbox_area,
    bbox_area_similarity,
    bbox_center,
    bbox_center_distance_factor,
    bbox_containment,
    bbox_diag,
    bbox_from_polygon,
    bbox_iou,
    empty_match_points,
    expand_bbox,
    is_visible_in_frame,
    point_in_any_bbox,
    point_in_bbox,
    point_outside_np_polygon_margin,
    polygon_axis_delta,
    polygon_from_bbox,
    polygon_has_usable_area,
    polygon_iou,
    project_points_with_homography,
    project_polygon_by_affine,
    translate_bbox,
    translate_polygon,
)
from modules.yolo.inspection.domain.runtime_fusion_contract import (
    APPLIED_GEOMETRY_SOURCE_BASELINE,
    APPLIED_GEOMETRY_SOURCE_TRUSTED_ANCHOR_TRANSFORM,
)

_MIN_MATCH_SCORE = 0.42
_MAX_CENTER_DISTANCE_FACTOR = 0.85
_MIN_AREA_RATIO = 0.20
_MAX_AREA_RATIO = 5.00
_MAX_OPTIMAL_ASSIGNMENT_CANDIDATES = 72
_MAX_OPTIMAL_ASSIGNMENT_ITEMS = 18
_DUPLICATE_MATCHED_BBOX_IOU = 0.55
_DUPLICATE_MATCHED_CONTAINMENT = 0.80

_MISSING_POLYGON_MAX_LOCAL_POINTS = 96
_MISSING_POLYGON_MIN_CONTAINMENT = 0.18
_MISSING_POLYGON_MIN_AREA_SCORE = 0.16
_MISSING_POLYGON_MIN_AFFINE_SCALE = 0.45
_MISSING_POLYGON_MAX_AFFINE_SCALE = 2.80
_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_SUPPORT = 10
_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_SUPPORT_RATIO = 0.45
_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_REFERENCE_AREA_SCORE = 0.12
_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MAX_OTHER_OVERLAP = 0.72
_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_VISIBLE_FRACTION = 0.08
_SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MAX_CENTER_FACTOR = 4.20
_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_CENTER_FACTOR = 0.15
_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MAX_CENTER_FACTOR = 0.55
_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_AREA_SCORE = 0.70
_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MAX_OTHER_OVERLAP = 1.01
_SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_VISIBLE_FRACTION = 0.08
_WEAK_CONTEXT_AFFINE_DEMOTION_MAX_SUPPORT = 24
_WEAK_CONTEXT_AFFINE_DEMOTION_MAX_TOTAL = 42
_WEAK_CONTEXT_AFFINE_DEMOTION_SPARSE_SUPPORT = 8
_WEAK_CONTEXT_AFFINE_DEMOTION_SPARSE_CANDIDATES = 10
_WEAK_CONTEXT_AFFINE_DEMOTION_MIN_MEDIAN_ERROR = 3.25
_WEAK_CONTEXT_AFFINE_DEMOTION_MAX_INLIER_RATIO = 0.86
_WEAK_CONTEXT_AFFINE_DEMOTION_MIN_CENTER_FACTOR = 0.14
_WEAK_CONTEXT_AFFINE_DEMOTION_MIN_AREA_SCORE = 0.86
_MULTI_EXPECTED_SLOT_MIN_GLOBAL_AREA_SCORE = 0.68
_MULTI_EXPECTED_SLOT_MAX_GLOBAL_CENTER_FACTOR = 0.28


def build_expected_segments(
    reference_image: Any,
    selected_class_ids: set[Any],
) -> list[ExpectedSegment]:
    items: list[ExpectedSegment] = []

    for annotation in reference_image.annotations:
        if annotation.segment_class_id is None or annotation.segment_class is None:
            continue
        if annotation.segment_class_id not in selected_class_ids:
            continue

        segment_class = annotation.segment_class

        for polygon in annotation.points or []:
            if len(polygon) < 3:
                continue

            items.append(
                ExpectedSegment(
                    annotation_id=annotation.id,
                    segment_class_id=segment_class.id,
                    class_key=str(segment_class.id),
                    name=segment_class.name,
                    hue=segment_class.hue,
                    reference_polygon=polygon,
                )
            )

    return items


def _expected_segment_match(
    item: ExpectedSegment,
    *,
    status: str,
    iou: float | None = None,
    confidence: float | None = None,
    expected_polygon: list[list[float]] | None = None,
    detected_polygon: list[list[float]] | None = None,
    detected_bbox: dict[str, float] | None = None,
    debug: dict[str, Any] | None = None,
) -> SegmentMatch:
    return SegmentMatch(
        annotation_id=item.annotation_id,
        segment_class_id=item.segment_class_id,
        class_key=item.class_key,
        name=item.name,
        hue=item.hue,
        status=status,
        iou=iou,
        confidence=confidence,
        expected_polygon=expected_polygon,
        detected_polygon=detected_polygon,
        detected_bbox=detected_bbox,
        debug=debug,
    )


def _detection_segment_match(
    detection: YoloDetection,
    *,
    name: str,
    hue: int | None,
    status: str,
    detected_polygon: list[list[float]] | None,
    debug: dict[str, Any] | None = None,
) -> SegmentMatch:
    return SegmentMatch(
        annotation_id=None,
        segment_class_id=None,
        class_key=detection.class_key,
        name=name,
        hue=hue,
        status=status,
        iou=None,
        confidence=detection.confidence,
        expected_polygon=None,
        detected_polygon=detected_polygon,
        detected_bbox=detection.bbox,
        debug=debug,
    )


def _missing_projection_debug_payload(
    *,
    projection: str,
    safety: str,
    include_projection_alias: bool = False,
    reason: str | None = None,
    reason_code: str | None = None,
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "missing_polygon_projection": projection,
        "missing_polygon_projection_safety": safety,
    }
    if include_projection_alias:
        payload["projection"] = projection
    if reason is not None:
        payload["reason"] = reason
    if reason_code is not None:
        payload["reason_code"] = reason_code
    payload.update(extra)
    return payload


def _unsafe_missing_projection_debug() -> dict[str, Any]:
    return _missing_projection_debug_payload(
        projection="unsafe_hidden",
        safety="unsafe_hidden",
        reason="projection_hidden_as_unsafe",
        reason_code="missing_projection_low_confidence",
    )


def build_missing_matches(
    expected: list[ExpectedSegment],
    homography: np.ndarray | None = None,
    *,
    frame_size: tuple[int, int] | None = None,
    projection_data: LocalProjectionData | None = None,
    hide_unconfirmed_projection: bool = False,
) -> list[SegmentMatch]:
    matches: list[SegmentMatch] = []

    for item in expected:
        expected_polygon = None

        debug: dict[str, Any] = {"reason": "no_matching_detection"}

        if hide_unconfirmed_projection:
            debug = _unsafe_missing_projection_debug()
        elif homography is not None:
            projected = _safe_project(
                item.reference_polygon,
                homography,
                projection_data=projection_data,
            )
            if len(projected) >= 3:
                bbox = bbox_from_polygon(projected)
                effective_frame_size = frame_size
                if effective_frame_size is None and projection_data is not None:
                    effective_frame_size = projection_data.frame_size
                visible = (
                    bbox is not None
                    and (
                        effective_frame_size is None
                        or is_visible_in_frame(
                            bbox,
                            effective_frame_size,
                            min_visible_fraction=0.05,
                        )
                    )
                )
                if visible:
                    expected_polygon = projected
                else:
                    debug = _unsafe_missing_projection_debug()

        matches.append(
            _expected_segment_match(
                item,
                status="missing",
                expected_polygon=expected_polygon,
                debug=debug,
            )
        )

    return matches


def match_segments(
    expected: list[ExpectedSegment],
    detections: list[YoloDetection],
    homography: np.ndarray | None,
    *,
    frame_size: tuple[int, int] | None = None,
    projection_data: LocalProjectionData | None = None,
    yolo_anchor_rescue: bool = False,
) -> list[SegmentMatch]:
    projected_result = _build_projected_expected_segments(
        expected,
        homography,
        frame_size=frame_size,
        projection_data=projection_data,
    )
    projected_expected = projected_result.projected
    unprojected_expected = projected_result.unprojected

    if not projected_expected:
        return build_missing_matches(
            expected,
            homography,
            frame_size=frame_size,
            projection_data=projection_data,
            hide_unconfirmed_projection=True,
        )

    detection_candidates = _build_detection_candidates(detections)
    projected_by_index = {item.index: item for item in projected_expected}
    detection_by_index = {item.index: item for item in detection_candidates}
    slot_by_index = _build_expected_slots(
        projected_expected,
        projection_data=projection_data,
    )
    runtime_anchor_traces = _build_runtime_anchor_traces(
        projected_expected,
        slot_by_index=slot_by_index,
        enabled=yolo_anchor_rescue,
    )
    trusted_anchors = _build_trusted_missing_anchors(
        projected_expected,
        slot_by_index=slot_by_index,
        projection_data=projection_data,
    )

    candidate_pairs, candidate_debug = _collect_candidate_pairs(
        projected_expected,
        detection_candidates,
        slot_by_index=slot_by_index,
        runtime_anchor_traces=runtime_anchor_traces,
        yolo_anchor_rescue=yolo_anchor_rescue,
    )
    chosen_pairs = _choose_candidate_pairs(candidate_pairs)
    matched_expected_indices = {expected_index for expected_index, _, _ in chosen_pairs}
    matched_detection_indices = {detection_index for _, detection_index, _ in chosen_pairs}

    _register_matched_detection_anchors(
        chosen_pairs,
        projected_by_index=projected_by_index,
        detection_by_index=detection_by_index,
        candidate_debug=candidate_debug,
        projection_data=projection_data,
        trusted_anchors=trusted_anchors,
    )

    matches: list[SegmentMatch] = []
    _append_matched_segment_matches(
        matches,
        chosen_pairs,
        projected_by_index=projected_by_index,
        detection_by_index=detection_by_index,
        candidate_debug=candidate_debug,
        runtime_anchor_traces=runtime_anchor_traces,
        yolo_anchor_rescue=yolo_anchor_rescue,
    )

    _append_missing_expected_matches(
        matches,
        projected_expected,
        matched_expected_indices=matched_expected_indices,
        slot_by_index=slot_by_index,
        projection_data=projection_data,
        trusted_anchors=trusted_anchors,
        runtime_anchor_traces=runtime_anchor_traces,
        yolo_anchor_rescue=yolo_anchor_rescue,
    )

    _append_unprojected_expected_matches(matches, unprojected_expected)

    expected_index_to_match = _build_expected_index_to_missing_match(
        matches,
        projected_expected,
        matched_expected_indices=matched_expected_indices,
    )

    _append_unmatched_detection_matches(
        matches,
        detection_candidates,
        projected_expected,
        projected_by_index=projected_by_index,
        detection_by_index=detection_by_index,
        slot_by_index=slot_by_index,
        expected_index_to_match=expected_index_to_match,
        matched_expected_indices=matched_expected_indices,
        matched_detection_indices=matched_detection_indices,
        yolo_anchor_rescue=yolo_anchor_rescue,
    )

    return matches


def _collect_candidate_pairs(
    projected_expected: list[ProjectedExpected],
    detection_candidates: list[DetectionCandidate],
    *,
    slot_by_index: dict[int, ExpectedSlot],
    runtime_anchor_traces: dict[int, dict[str, Any]],
    yolo_anchor_rescue: bool,
) -> tuple[list[tuple[float, float, int, int]], dict[tuple[int, int], dict[str, Any]]]:
    candidate_pairs: list[tuple[float, float, int, int]] = []
    candidate_debug: dict[tuple[int, int], dict[str, Any]] = {}

    for expected_item in projected_expected:
        for detection_item in detection_candidates:
            if expected_item.item.class_key != detection_item.detection.class_key:
                continue

            global_debug = _match_score_details(
                expected_item.polygon,
                detection_item.polygon,
                expected_item.bbox,
                detection_item.bbox,
            )
            slot_candidate = _build_detection_slot_candidate(
                expected_item,
                detection_item,
                slot=slot_by_index.get(expected_item.index),
                global_debug=global_debug,
                runtime_anchor_trace=runtime_anchor_traces.get(expected_item.index),
                yolo_anchor_rescue=yolo_anchor_rescue,
            )
            if slot_candidate is None:
                continue

            slot_score, slot_iou, slot_debug = slot_candidate
            candidate_debug[(expected_item.index, detection_item.index)] = slot_debug
            candidate_pairs.append(
                (slot_score, slot_iou, expected_item.index, detection_item.index)
            )

    return candidate_pairs, candidate_debug


def _build_detection_slot_candidate(
    expected_item: ProjectedExpected,
    detection_item: DetectionCandidate,
    *,
    slot: ExpectedSlot | None,
    global_debug: dict[str, Any],
    runtime_anchor_trace: dict[str, Any] | None,
    yolo_anchor_rescue: bool,
) -> tuple[float, float, dict[str, Any]] | None:
    if not yolo_anchor_rescue:
        return _try_slot_candidate(
            expected_item,
            detection_item,
            slot=slot,
            global_debug=global_debug,
        )

    slot_candidate = runtime_yolo_anchor_candidate_details(
        expected_item,
        detection_item,
        slot=slot,
        global_debug=global_debug,
        slot_debug=_slot_debug_payload(slot),
    )
    if slot_candidate is None:
        return None

    record_runtime_yolo_anchor_trace_candidate(
        runtime_anchor_trace,
        detection_index=detection_item.index,
        debug=slot_candidate[2],
    )
    if not slot_candidate[2].get("passed"):
        return None

    return slot_candidate


def _register_matched_detection_anchors(
    chosen_pairs: list[tuple[int, int, float]],
    *,
    projected_by_index: dict[int, ProjectedExpected],
    detection_by_index: dict[int, DetectionCandidate],
    candidate_debug: dict[tuple[int, int], dict[str, Any]],
    projection_data: LocalProjectionData | None,
    trusted_anchors: list[TrustedAnchor],
) -> None:
    for expected_index, detection_index, iou in chosen_pairs:
        expected_item = projected_by_index[expected_index]
        detection_item = detection_by_index[detection_index]
        match_debug = candidate_debug.get((expected_index, detection_index))
        matched_anchor = _trusted_anchor_from_matched_detection(
            expected_item,
            detection_item=detection_item,
            match_iou=iou,
            projection_data=projection_data,
            match_debug=match_debug,
        )
        _append_or_replace_trusted_anchor(trusted_anchors, matched_anchor)


def _append_matched_segment_matches(
    matches: list[SegmentMatch],
    chosen_pairs: list[tuple[int, int, float]],
    *,
    projected_by_index: dict[int, ProjectedExpected],
    detection_by_index: dict[int, DetectionCandidate],
    candidate_debug: dict[tuple[int, int], dict[str, Any]],
    runtime_anchor_traces: dict[int, dict[str, Any]],
    yolo_anchor_rescue: bool,
) -> None:
    for expected_index, detection_index, iou in chosen_pairs:
        expected_item = projected_by_index[expected_index]
        detection_item = detection_by_index[detection_index]
        detection = detection_item.detection

        match_debug = candidate_debug.get((expected_index, detection_index))
        if yolo_anchor_rescue:
            match_debug = merge_runtime_yolo_anchor_trace_debug(
                match_debug,
                runtime_anchor_traces.get(expected_index),
            )

        matches.append(
            _expected_segment_match(
                expected_item.item,
                status="ok",
                iou=round(float(iou), 4),
                confidence=detection.confidence,
                expected_polygon=expected_item.polygon,
                detected_polygon=_display_detected_polygon(detection_item),
                detected_bbox=detection.bbox,
                debug=match_debug,
            )
        )


def _append_missing_expected_matches(
    matches: list[SegmentMatch],
    projected_expected: list[ProjectedExpected],
    *,
    matched_expected_indices: set[int],
    slot_by_index: dict[int, ExpectedSlot],
    projection_data: LocalProjectionData | None,
    trusted_anchors: list[TrustedAnchor],
    runtime_anchor_traces: dict[int, dict[str, Any]],
    yolo_anchor_rescue: bool,
) -> None:
    for expected_item in projected_expected:
        if expected_item.index in matched_expected_indices:
            continue

        slot = slot_by_index.get(expected_item.index)
        missing_debug = _missing_slot_debug(slot)
        if yolo_anchor_rescue:
            missing_debug = merge_runtime_yolo_no_anchor_debug(
                missing_debug,
                runtime_anchor_traces.get(expected_item.index),
            )
        missing_polygon = expected_item.polygon

        missing_refinement = _try_missing_polygon_refinement(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            all_expected=projected_expected,
        )
        if missing_refinement is not None:
            demotion_reason = _missing_refinement_affine_demotion_reason(
                missing_refinement,
                all_expected=projected_expected,
            )
            if demotion_reason is not None:
                missing_debug = _merge_missing_refinement_demotion_debug(
                    missing_debug,
                    missing_refinement,
                    reason=demotion_reason,
                )
                missing_refinement = None

        if missing_refinement is not None:
            missing_polygon, missing_debug = _resolve_refined_missing_projection(
                expected_item,
                missing_refinement=missing_refinement,
                missing_debug=missing_debug,
                slot=slot,
                projection_data=projection_data,
                all_expected=projected_expected,
            )
        else:
            missing_polygon, missing_debug = _resolve_fallback_missing_projection(
                expected_item,
                missing_debug=missing_debug,
                slot=slot,
                projection_data=projection_data,
                all_expected=projected_expected,
                all_slots=slot_by_index,
                trusted_anchors=trusted_anchors,
            )

        resolved_anchor = _trusted_anchor_from_resolved_projection(
            expected_item,
            polygon=missing_polygon,
            bbox=bbox_from_polygon(missing_polygon) if missing_polygon is not None else None,
            projection_data=projection_data,
            debug=missing_debug,
        )
        _append_or_replace_trusted_anchor(trusted_anchors, resolved_anchor)
        if yolo_anchor_rescue:
            missing_debug = _merge_runtime_applied_geometry_source_debug(missing_debug)

        matches.append(
            _expected_segment_match(
                expected_item.item,
                status="missing",
                expected_polygon=missing_polygon,
                debug=missing_debug,
            )
        )


def _resolve_refined_missing_projection(
    expected_item: ProjectedExpected,
    *,
    missing_refinement: MissingPolygonRefinement,
    missing_debug: dict[str, Any],
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected],
) -> tuple[list[list[float]], dict[str, Any]]:
    translation_override = _try_missing_refinement_translation_override(
        expected_item,
        refinement=missing_refinement,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    if translation_override is None:
        return (
            missing_refinement.polygon,
            _merge_missing_refinement_debug(missing_debug, missing_refinement),
        )

    rescue, projection_name = translation_override
    return (
        rescue.polygon,
        _merge_missing_translation_rescue_debug(
            missing_debug,
            rescue,
            source_projection=projection_name,
            replaced_projection="context_feature_affine",
            replaced_median_error=missing_refinement.median_error,
            replaced_inlier_ratio=missing_refinement.inlier_ratio,
            replaced_area_score=missing_refinement.area_score,
        ),
    )


def _resolve_fallback_missing_projection(
    expected_item: ProjectedExpected,
    *,
    missing_debug: dict[str, Any],
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected],
    all_slots: dict[int, ExpectedSlot],
    trusted_anchors: list[TrustedAnchor],
) -> tuple[list[list[float]] | None, dict[str, Any]]:
    fallback = _resolve_missing_fallback_projection(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
        all_slots=all_slots,
        trusted_anchors=trusted_anchors,
    )

    if fallback.debug:
        missing_debug = {**missing_debug, **fallback.debug}
    if fallback.hidden:
        missing_debug = _merge_unsafe_missing_projection_debug(missing_debug)

    return fallback.polygon, missing_debug


def _append_unmatched_detection_matches(
    matches: list[SegmentMatch],
    detection_candidates: list[DetectionCandidate],
    projected_expected: list[ProjectedExpected],
    *,
    projected_by_index: dict[int, ProjectedExpected],
    detection_by_index: dict[int, DetectionCandidate],
    slot_by_index: dict[int, ExpectedSlot],
    expected_index_to_match: dict[int, SegmentMatch],
    matched_expected_indices: set[int],
    matched_detection_indices: set[int],
    yolo_anchor_rescue: bool,
) -> None:
    unmatched_expected_indices: set[int] = set()
    occupied_detection_indices = set(matched_detection_indices)

    for detection_item in detection_candidates:
        if detection_item.index in occupied_detection_indices:
            continue

        action, expected_index, debug = _classify_unmatched_detection(
            detection_item,
            projected_expected,
            matched_expected_indices | unmatched_expected_indices,
            slot_by_index=slot_by_index,
            occupied_detection_indices=occupied_detection_indices,
            detection_by_index=detection_by_index,
            yolo_anchor_rescue=yolo_anchor_rescue,
        )
        detection = detection_item.detection

        if action == "discard":
            _record_detected_class_in_zone(
                expected_index,
                detection.class_key,
                expected_index_to_match,
            )
            continue

        if action == "unmatched":
            _append_unmatched_segment_match(
                matches,
                detection_item,
                expected_index=expected_index,
                expected_item=(
                    projected_by_index.get(expected_index)
                    if expected_index is not None
                    else None
                ),
                expected_index_to_match=expected_index_to_match,
                debug=debug,
            )
            if expected_index is not None:
                unmatched_expected_indices.add(expected_index)
            occupied_detection_indices.add(detection_item.index)
            continue

        _append_extra_segment_match(matches, detection_item, debug=debug)


def _record_detected_class_in_zone(
    expected_index: int | None,
    class_key: str,
    expected_index_to_match: dict[int, SegmentMatch],
) -> None:
    if expected_index is None:
        return

    target_match = expected_index_to_match.get(expected_index)
    if target_match is not None and target_match.detected_class_in_zone is None:
        target_match.detected_class_in_zone = class_key


def _append_unmatched_segment_match(
    matches: list[SegmentMatch],
    detection_item: DetectionCandidate,
    *,
    expected_index: int | None,
    expected_item: ProjectedExpected | None,
    expected_index_to_match: dict[int, SegmentMatch],
    debug: dict[str, Any] | None,
) -> None:
    detection = detection_item.detection
    _record_detected_class_in_zone(
        expected_index,
        detection.class_key,
        expected_index_to_match,
    )
    matches.append(
        _detection_segment_match(
            detection,
            name=expected_item.item.name if expected_item else "Не сопоставлено",
            hue=expected_item.item.hue if expected_item else None,
            status="unmatched",
            detected_polygon=detection_item.polygon,
            debug=debug,
        )
    )


def _append_extra_segment_match(
    matches: list[SegmentMatch],
    detection_item: DetectionCandidate,
    *,
    debug: dict[str, Any] | None,
) -> None:
    detection = detection_item.detection
    matches.append(
        _detection_segment_match(
            detection,
            name="Лишнее",
            hue=None,
            status="extra",
            detected_polygon=detection_item.polygon,
            debug=debug,
        )
    )


def match_segments_by_count(
    expected: list[ExpectedSegment],
    detections: list[YoloDetection],
) -> list[SegmentMatch]:
    expected_by_class: dict[str, list[ExpectedSegment]] = {}
    for item in expected:
        expected_by_class.setdefault(item.class_key, []).append(item)

    detections_by_class: dict[str, list[YoloDetection]] = {}
    for detection in detections:
        detections_by_class.setdefault(detection.class_key, []).append(detection)

    class_keys = sorted(set(expected_by_class) | set(detections_by_class))
    matches: list[SegmentMatch] = []

    for class_key in class_keys:
        expected_items = expected_by_class.get(class_key, [])
        detected_items = detections_by_class.get(class_key, [])
        matched_count = min(len(expected_items), len(detected_items))
        fallback_name = expected_items[0].name if expected_items else "Лишнее"
        fallback_hue = expected_items[0].hue if expected_items else None

        for index in range(matched_count):
            expected_item = expected_items[index]
            detection = detected_items[index]
            detected_polygon = _detection_polygon(detection)

            matches.append(
                _expected_segment_match(
                    expected_item,
                    status="ok",
                    confidence=detection.confidence,
                    detected_polygon=detected_polygon,
                    detected_bbox=detection.bbox,
                )
            )

        for expected_item in expected_items[matched_count:]:
            matches.append(
                _expected_segment_match(
                    expected_item,
                    status="missing",
                )
            )

        for detection in detected_items[matched_count:]:
            detected_polygon = _detection_polygon(detection)
            matches.append(
                _detection_segment_match(
                    detection,
                    name=fallback_name,
                    hue=fallback_hue,
                    status="extra",
                    detected_polygon=detected_polygon,
                )
            )

    return matches


def summarize(matches: list[SegmentMatch]) -> tuple[int, int, list[str]]:
    expected_matches = [match for match in matches if match.status in ("ok", "missing")]
    total = len(expected_matches)
    matched = sum(1 for match in expected_matches if match.status == "ok")
    missing_names = [match.name for match in expected_matches if match.status == "missing"]
    return total, matched, missing_names


def all_ok(matches: list[SegmentMatch], *, expected_total: int | None = None) -> bool:
    expected_matches = [match for match in matches if match.status in ("ok", "missing")]
    total = expected_total if expected_total is not None else len(expected_matches)
    if total <= 0:
        return False
    if any(match.status in ("extra", "unmatched") for match in matches):
        return False
    return sum(1 for match in expected_matches if match.status == "ok") == total


def _build_projected_expected_segments(
    expected: list[ExpectedSegment],
    homography: np.ndarray | None,
    *,
    frame_size: tuple[int, int] | None,
    projection_data: LocalProjectionData | None,
) -> ProjectedExpectedBuildResult:
    projected_expected: list[ProjectedExpected] = []
    unprojected_expected: list[tuple[int, ExpectedSegment, dict[str, Any]]] = []

    for index, item in enumerate(expected):
        projected = _safe_project(
            item.reference_polygon,
            homography,
            projection_data=projection_data,
        )
        if len(projected) < 3:
            unprojected_expected.append(
                (
                    index,
                    item,
                    _missing_none_projection_debug(
                        reason="projection_failed_invalid_polygon",
                        homography=homography,
                        projected_point_count=len(projected),
                    ),
                )
            )
            continue

        bbox = bbox_from_polygon(projected)
        if bbox is None:
            unprojected_expected.append(
                (
                    index,
                    item,
                    _missing_none_projection_debug(
                        reason="projection_failed_invalid_bbox",
                        homography=homography,
                        projected_point_count=len(projected),
                    ),
                )
            )
            continue

        if frame_size is not None and not is_visible_in_frame(bbox, frame_size):
            unprojected_expected.append(
                (
                    index,
                    item,
                    _missing_none_projection_debug(
                        reason="projection_failed_outside_frame",
                        homography=homography,
                        projected_point_count=len(projected),
                        bbox=bbox,
                    ),
                )
            )
            continue

        projected_expected.append(
            ProjectedExpected(
                index=index,
                item=item,
                polygon=projected,
                bbox=bbox,
            )
        )

    return ProjectedExpectedBuildResult(
        projected=projected_expected,
        unprojected=unprojected_expected,
    )


def _build_detection_candidates(
    detections: list[YoloDetection],
) -> list[DetectionCandidate]:
    candidates: list[DetectionCandidate] = []

    for index, detection in enumerate(detections):
        polygon = _detection_polygon(detection)
        bbox = _bbox_from_detection(detection, polygon)
        if polygon is None or bbox is None:
            continue

        candidates.append(
            DetectionCandidate(
                index=index,
                detection=detection,
                polygon=polygon,
                bbox=bbox,
            )
        )

    return candidates


def _build_runtime_anchor_traces(
    projected_expected: list[ProjectedExpected],
    *,
    slot_by_index: dict[int, ExpectedSlot],
    enabled: bool,
) -> dict[int, dict[str, Any]]:
    if not enabled:
        return {}

    return {
        item.index: build_runtime_yolo_anchor_trace(
            item,
            slot_by_index.get(item.index),
        )
        for item in projected_expected
    }


def _append_unprojected_expected_matches(
    matches: list[SegmentMatch],
    unprojected_expected: list[tuple[int, ExpectedSegment, dict[str, Any]]],
) -> None:
    for _, item, debug in unprojected_expected:
        matches.append(
            _expected_segment_match(
                item,
                status="missing",
                debug=debug,
            )
        )


def _build_expected_index_to_missing_match(
    matches: list[SegmentMatch],
    projected_expected: list[ProjectedExpected],
    *,
    matched_expected_indices: set[int],
) -> dict[int, SegmentMatch]:
    result: dict[int, SegmentMatch] = {}

    for expected_item in projected_expected:
        if expected_item.index in matched_expected_indices:
            continue
        for match in matches:
            if (
                match.status == "missing"
                and match.annotation_id == expected_item.item.annotation_id
                and match.segment_class_id == expected_item.item.segment_class_id
            ):
                result[expected_item.index] = match
                break

    return result


def _safe_project(
    polygon: list[list[float]],
    homography: np.ndarray | None,
    *,
    projection_data: LocalProjectionData | None = None,
) -> list[list[float]]:
    try:
        if projection_data is not None:
            projected = project_polygon_adaptive(
                polygon,
                data=projection_data,
            )
        elif homography is not None:
            projected = project_polygon(polygon, homography)
        else:
            return []
    except Exception:
        return []

    if not projected or len(projected) < 3:
        return []

    if not all(len(point) >= 2 for point in projected):
        return []

    return [[float(point[0]), float(point[1])] for point in projected]


def _detection_polygon(detection: YoloDetection) -> list[list[float]] | None:
    if detection.polygon is not None and len(detection.polygon) >= 3:
        return [[float(x), float(y)] for x, y in detection.polygon]
    bbox = _bbox_from_detection(detection, None)
    if bbox is not None:
        return polygon_from_bbox(bbox)
    return None


def _bbox_from_detection(
    detection: YoloDetection,
    polygon: list[list[float]] | None,
) -> BBox | None:
    bbox = detection.bbox or {}
    try:
        x = float(bbox.get("x", 0))
        y = float(bbox.get("y", 0))
        w = float(bbox.get("w", 0))
        h = float(bbox.get("h", 0))
    except (TypeError, ValueError):
        x = y = w = h = 0.0

    if w > 0 and h > 0:
        return (x, y, x + w, y + h)

    if polygon is not None:
        return bbox_from_polygon(polygon)

    return None

def _build_expected_slots(
    projected_expected: list[ProjectedExpected],
    *,
    projection_data: LocalProjectionData | None,
) -> dict[int, ExpectedSlot]:
    slots: dict[int, ExpectedSlot] = {}

    for expected_item in projected_expected:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
        search_bbox = expand_bbox(
            expected_item.bbox,
            factor=_thresholds.slot_search_expansion,
            frame_size=(projection_data.frame_size if projection_data is not None else None),
        )
        feature_support, feature_total = _slot_feature_support(
            expected_item,
            search_bbox=search_bbox,
            projection_data=projection_data,
            all_expected=projected_expected,
        )

        slots[expected_item.index] = ExpectedSlot(
            projected_bbox=expected_item.bbox,
            search_bbox=search_bbox,
            reference_bbox=reference_bbox,
            feature_support=feature_support,
            feature_total=feature_total,
        )

    return slots


def _slot_candidate_details(
    expected_item: ProjectedExpected,
    detection_item: DetectionCandidate,
    *,
    slot: ExpectedSlot | None,
    global_debug: dict[str, Any],
) -> tuple[float, float, dict[str, Any]] | None:
    """Return slot score details for same-class YOLO detection.

    A slot candidate exists only when YOLO detection is actually inside the
    expanded expected slot. The exact projected polygon IoU is kept as a debug
    metric only and must not be used as proof that a slot candidate is absent.
    """

    if slot is None:
        return None

    if (
        detection_item.detection.confidence
        < _thresholds.slot_min_yolo_confidence
    ):
        return None

    containment = bbox_containment(detection_item.bbox, slot.search_bbox)
    if containment < _thresholds.slot_min_detection_containment:
        return None

    expected_center = bbox_center(slot.projected_bbox)
    detection_center = bbox_center(detection_item.bbox)
    search_diag = max(1.0, bbox_diag(slot.search_bbox))
    center_distance = float(
        np.hypot(
            expected_center[0] - detection_center[0],
            expected_center[1] - detection_center[1],
        )
    )
    center_score = max(0.0, 1.0 - center_distance / max(1.0, search_diag * 0.55))

    expected_area = max(1.0, bbox_area(slot.projected_bbox))
    detection_area = max(1.0, bbox_area(detection_item.bbox))
    area_ratio = detection_area / expected_area
    area_score = min(area_ratio, 1.0 / area_ratio)
    area_score = max(0.0, min(1.0, area_score))

    slot_iou = bbox_iou(slot.search_bbox, detection_item.bbox)
    projected_iou = polygon_iou(expected_item.polygon, detection_item.polygon)
    feature_score = min(
        1.0,
        slot.feature_support / max(1, _thresholds.slot_min_feature_support),
    )

    confidence_score = max(0.0, min(1.0, float(detection_item.detection.confidence)))

    score = (
        center_score * 0.30
        + containment * 0.26
        + area_score * 0.14
        + min(1.0, slot_iou * 5.0) * 0.08
        + min(1.0, projected_iou * 3.0) * 0.05
        + feature_score * 0.05
        + confidence_score * 0.12
    )
    passed = score >= _thresholds.slot_min_score

    debug = {
        "reason": "slot_matched" if passed else "slot_candidate_rejected",
        "projection": "expected_slot",
        "candidate_source": "yolo_slot",
        "score": _round_debug(score),
        "threshold": _round_debug(_thresholds.slot_min_score),
        "passed": passed,
        "reject_reason": None if passed else "slot_score_below_threshold",
        "iou": _round_debug(projected_iou),
        "global_score": global_debug.get("score"),
        "global_iou": global_debug.get("iou"),
        "global_reject_reason": global_debug.get("reject_reason"),
        "slot_center_score": _round_debug(center_score),
        "slot_detection_containment": _round_debug(containment),
        "slot_bbox_iou": _round_debug(slot_iou),
        "slot_area_score": _round_debug(area_score),
        "slot_feature_score": _round_debug(feature_score),
        "slot_yolo_confidence": _round_debug(detection_item.detection.confidence),
        "slot_yolo_min_confidence": _round_debug(
            _thresholds.slot_min_yolo_confidence
        ),
        "slot": _slot_debug_payload(slot),
    }

    return float(score), float(projected_iou), debug


def _try_slot_candidate(
    expected_item: ProjectedExpected,
    detection_item: DetectionCandidate,
    *,
    slot: ExpectedSlot | None,
    global_debug: dict[str, Any],
) -> tuple[float, float, dict[str, Any]] | None:
    """Match detection inside an expanded expected slot instead of exact projection.

    The projected polygon is only a navigation hint. A same-class YOLO object that
    lands inside the expected slot should be accepted even when polygon IoU is low,
    which is common with 3D perspective drift and thin/round details.
    """

    details = _slot_candidate_details(
        expected_item,
        detection_item,
        slot=slot,
        global_debug=global_debug,
    )
    if details is None:
        return None

    score, projected_iou, debug = details
    if not debug.get("passed"):
        return None

    return score, projected_iou, debug


def _merge_runtime_applied_geometry_source_debug(
    debug: dict[str, Any],
) -> dict[str, Any]:
    projection = _missing_debug_projection_name(debug)
    if projection == "expected_slot_anchor_release":
        return {
            **debug,
            "applied_geometry_source": APPLIED_GEOMETRY_SOURCE_TRUSTED_ANCHOR_TRANSFORM,
            "runtime_anchor_transform_applied": True,
        }
    return {
        **debug,
        "applied_geometry_source": (
            debug.get("applied_geometry_source")
            or APPLIED_GEOMETRY_SOURCE_BASELINE
        ),
        "runtime_anchor_transform_applied": False,
    }


def _missing_slot_debug(slot: ExpectedSlot | None) -> dict[str, Any]:
    if slot is None:
        return {"reason": "slot_no_evidence", "projection": "expected_slot"}

    debug: dict[str, Any] = {
        "reason": "Модель YOLO не обнаружила деталь в ожидаемой области",
        "reason_code": "slot_no_yolo_confirmation",
        "projection": "expected_slot",
        "candidate_source": "none",
        "slot": _slot_debug_payload(slot),
    }

    if slot.feature_support >= _thresholds.slot_min_feature_support:
        debug.update(
            {
                "reason": "Ожидаемая область найдена, но YOLO не обнаружила деталь",
                "reason_code": "feature_slot_unconfirmed_without_yolo",
                "candidate_source": "context_feature_support_only",
                "slot_feature_source": "context_ring",
                "slot_feature_support": slot.feature_support,
                "slot_feature_total": slot.feature_total,
                "slot_feature_min_support": (
                    _thresholds.slot_min_feature_support
                ),
            }
        )

    return debug


def _try_missing_polygon_refinement(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None = None,
) -> MissingPolygonRefinement | None:
    if not _thresholds.missing_polygon_refinement:
        return None
    if slot is None or projection_data is None:
        return None

    min_support = max(3, _thresholds.missing_polygon_min_feature_support)
    if slot.feature_support < min_support:
        return None

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return None

    local_reference, local_frame = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    if len(local_reference) < min_support:
        return None

    ransac_threshold = max(
        1.0,
        min(25.0, _thresholds.missing_polygon_max_reprojection_error),
    )

    try:
        affine, inliers = cv2.estimateAffinePartial2D(
            local_reference,
            local_frame,
            method=cv2.RANSAC,
            ransacReprojThreshold=ransac_threshold,
            maxIters=1200,
            confidence=0.98,
            refineIters=10,
        )
    except cv2.error:
        return None

    if affine is None or inliers is None:
        return None
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        return None
    if not _missing_affine_scale_in_range(affine):
        return None

    inlier_mask = np.asarray(inliers, dtype=np.uint8).reshape(-1) > 0
    inlier_count = int(np.count_nonzero(inlier_mask))
    if inlier_count < min_support:
        return None

    candidate_count = int(len(local_reference))
    inlier_ratio = inlier_count / max(1, candidate_count)
    if inlier_ratio < _thresholds.missing_polygon_context_min_inlier_ratio:
        return None
    is_multi_object_context = all_expected is not None and len(all_expected) > 1
    if is_multi_object_context and inlier_ratio < 0.50:
        return None

    inlier_reference = local_reference[inlier_mask]
    inlier_frame = local_frame[inlier_mask]
    reference_spread = _missing_context_spread_score(inlier_reference, reference_bbox)
    frame_spread = _missing_context_spread_score(inlier_frame, slot.projected_bbox)
    min_spread = _thresholds.missing_polygon_context_min_spread
    if min(reference_spread, frame_spread) < min_spread:
        return None

    quadrant_count = _missing_context_quadrant_count(inlier_reference, reference_bbox)
    min_quadrants = _thresholds.missing_polygon_context_min_quadrants
    if is_multi_object_context:
        min_quadrants = max(min_quadrants, 3)
    if quadrant_count < min_quadrants:
        return None

    median_error = affine_reprojection_median_error(
        affine,
        source=inlier_reference,
        target=inlier_frame,
    )
    if median_error is None:
        return None
    if median_error > _thresholds.missing_polygon_max_reprojection_error:
        return None

    polygon = project_polygon_by_affine(
        expected_item.item.reference_polygon,
        np.asarray(affine, dtype=np.float32),
    )
    if len(polygon) < 3:
        return None

    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return None

    if (
        projection_data.frame_size is not None
        and not is_visible_in_frame(
            bbox,
            projection_data.frame_size,
            min_visible_fraction=0.05,
        )
    ):
        return None

    containment = bbox_containment(bbox, slot.search_bbox)
    if containment < _MISSING_POLYGON_MIN_CONTAINMENT:
        return None

    area_score = bbox_area_similarity(bbox, expected_item.bbox)
    min_area_score = max(
        _MISSING_POLYGON_MIN_AREA_SCORE,
        _thresholds.missing_polygon_context_min_area_score,
    )
    if area_score < min_area_score:
        return None

    center_drift_factor = bbox_center_distance_factor(bbox, expected_item.bbox)
    if (
        center_drift_factor
        > _thresholds.missing_polygon_context_max_center_drift_factor
    ):
        return None

    if is_multi_object_context:
        global_projection = _missing_global_projection(
            expected_item,
            projection_data=projection_data,
        )
        if global_projection is None:
            return None
        _, global_bbox = global_projection
        global_area_score = bbox_area_similarity(bbox, global_bbox)
        global_center_factor = bbox_center_distance_factor(bbox, global_bbox)
        if global_area_score < 0.42 or global_center_factor > 0.58:
            return None

    return MissingPolygonRefinement(
        polygon=polygon,
        bbox=bbox,
        feature_support=slot.feature_support,
        feature_total=slot.feature_total,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        reference_spread=float(reference_spread),
        frame_spread=float(frame_spread),
        quadrant_count=int(quadrant_count),
        center_drift_factor=float(center_drift_factor),
        area_score=float(area_score),
    )


def _merge_missing_refinement_debug(
    missing_debug: dict[str, Any],
    refinement: MissingPolygonRefinement,
) -> dict[str, Any]:
    return {
        **missing_debug,
        "missing_polygon_refined": True,
        "missing_polygon_projection": "context_feature_affine",
        "missing_polygon_bbox": _bbox_debug(refinement.bbox),
        "missing_polygon_feature_support": refinement.feature_support,
        "missing_polygon_feature_total": refinement.feature_total,
        "missing_polygon_candidate_count": refinement.candidate_count,
        "missing_polygon_inliers": refinement.inlier_count,
        "missing_polygon_inlier_ratio": _round_debug(refinement.inlier_ratio),
        "missing_polygon_median_error": _round_debug(refinement.median_error),
        "missing_polygon_area_score": _round_debug(refinement.area_score),
    }



def _missing_refinement_affine_demotion_reason(
    refinement: MissingPolygonRefinement,
    *,
    all_expected: list[ProjectedExpected] | None = None,
) -> str | None:
    if all_expected is None or len(all_expected) <= 1:
        return None

    feature_support = int(refinement.feature_support)
    feature_total = int(refinement.feature_total)
    candidate_count = int(refinement.candidate_count)
    quadrant_count = int(refinement.quadrant_count)
    median_error = float(refinement.median_error)
    inlier_ratio = float(refinement.inlier_ratio)
    center_factor = float(refinement.center_drift_factor)
    area_score = float(refinement.area_score)

    max_support = _WEAK_CONTEXT_AFFINE_DEMOTION_MAX_SUPPORT
    max_total = _WEAK_CONTEXT_AFFINE_DEMOTION_MAX_TOTAL
    min_error = _WEAK_CONTEXT_AFFINE_DEMOTION_MIN_MEDIAN_ERROR
    max_ratio = _WEAK_CONTEXT_AFFINE_DEMOTION_MAX_INLIER_RATIO
    min_center = _WEAK_CONTEXT_AFFINE_DEMOTION_MIN_CENTER_FACTOR
    min_area = _WEAK_CONTEXT_AFFINE_DEMOTION_MIN_AREA_SCORE

    weak_evidence = (
        feature_support <= max_support
        or feature_total <= max_total
        or candidate_count <= max(4, max_support + 2)
    )
    sparse_evidence = (
        feature_support <= _WEAK_CONTEXT_AFFINE_DEMOTION_SPARSE_SUPPORT
        or candidate_count <= _WEAK_CONTEXT_AFFINE_DEMOTION_SPARSE_CANDIDATES
    )
    weak_geometry = quadrant_count <= 3

    if (
        weak_evidence
        and weak_geometry
        and median_error >= min_error
        and (inlier_ratio <= max_ratio or center_factor >= min_center or area_score <= min_area)
    ):
        return "weak_context_affine_unstable"

    if sparse_evidence and weak_geometry and center_factor >= min_center * 0.85:
        return "sparse_context_affine_center_drift"

    if weak_evidence and weak_geometry and area_score <= min_area * 0.94 and median_error >= min_error * 0.85:
        return "weak_context_affine_bad_area"

    return None


def _merge_missing_refinement_demotion_debug(
    missing_debug: dict[str, Any],
    refinement: MissingPolygonRefinement,
    *,
    reason: str,
) -> dict[str, Any]:
    return {
        **missing_debug,
        "missing_polygon_affine_demoted": True,
        "missing_polygon_affine_demoted_reason": reason,
        "missing_polygon_demoted_projection": "context_feature_affine",
        "missing_polygon_demoted_median_error": _round_debug(refinement.median_error),
        "missing_polygon_demoted_area_score": _round_debug(refinement.area_score),
    }


def _try_missing_refinement_translation_override(
    expected_item: ProjectedExpected,
    *,
    refinement: MissingPolygonRefinement,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None = None,
) -> tuple[MissingTranslationRescue, str] | None:
    if slot is None or projection_data is None:
        return None
    if not _missing_refinement_should_try_translation_override(refinement):
        return None

    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return None

    global_polygon, global_bbox = global_projection
    rescue = _try_missing_translation_rescue(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        all_expected=all_expected,
    )
    projection_name = "context_feature_affine_translation_rescue"

    if rescue is None:
        rescue = _try_missing_scene_translation_rescue(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            global_polygon=global_polygon,
            global_bbox=global_bbox,
            all_expected=all_expected,
        )
        projection_name = "context_feature_affine_scene_translation_rescue"

    if rescue is None:
        return None
    if not _missing_translation_override_is_safer(
        refinement,
        rescue,
        expected_item=expected_item,
        all_expected=all_expected,
    ):
        return None

    return rescue, projection_name


def _missing_refinement_should_try_translation_override(
    refinement: MissingPolygonRefinement,
) -> bool:
    if refinement.candidate_count <= 0:
        return False

    weak_ratio = refinement.inlier_ratio < 0.58
    noisy_error = refinement.median_error > 5.75
    weak_area = refinement.area_score < 0.78
    wide_drift = refinement.center_drift_factor > 0.30
    weak_spread = min(refinement.reference_spread, refinement.frame_spread) < 0.24
    weak_quadrants = refinement.quadrant_count < 3

    return any(
        (
            weak_ratio,
            noisy_error,
            weak_area,
            wide_drift,
            weak_spread,
            weak_quadrants,
        )
    )


def _missing_translation_override_is_safer(
    refinement: MissingPolygonRefinement,
    rescue: MissingTranslationRescue,
    *,
    expected_item: ProjectedExpected,
    all_expected: list[ProjectedExpected] | None,
) -> bool:
    rescue_area_score = bbox_area_similarity(rescue.bbox, expected_item.bbox)
    if rescue_area_score < 0.32:
        return False
    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=rescue.bbox,
        all_expected=all_expected,
    ):
        return False

    affine_is_deformed = (
        refinement.area_score < 0.72
        or refinement.center_drift_factor > 0.42
        or refinement.inlier_ratio < 0.48
    )
    rescue_error_better = rescue.median_error <= max(
        3.25,
        refinement.median_error * 0.82,
    )
    rescue_ratio_better = rescue.inlier_ratio >= refinement.inlier_ratio + 0.08

    return affine_is_deformed or rescue_error_better or rescue_ratio_better


def _merge_missing_translation_rescue_debug(
    missing_debug: dict[str, Any],
    rescue: MissingTranslationRescue,
    *,
    source_projection: str,
    replaced_projection: str,
    replaced_median_error: float,
    replaced_inlier_ratio: float,
    replaced_area_score: float,
) -> dict[str, Any]:
    return {
        **missing_debug,
        "missing_polygon_refined": True,
        **_missing_projection_debug_payload(
            projection=source_projection,
            safety="translation_override",
        ),
        "missing_polygon_bbox": _bbox_debug(rescue.bbox),
        "missing_polygon_candidate_count": rescue.candidate_count,
        "missing_polygon_inliers": rescue.inlier_count,
        "missing_polygon_inlier_ratio": _round_debug(rescue.inlier_ratio),
        "missing_polygon_median_error": _round_debug(rescue.median_error),
        "missing_polygon_translation_shift_x": _round_debug(rescue.shift_x),
        "missing_polygon_translation_shift_y": _round_debug(rescue.shift_y),
        "missing_polygon_translation_shift_factor": _round_debug(
            rescue.shift_factor
        ),
        "missing_polygon_replaced_projection": replaced_projection,
        "missing_polygon_replaced_median_error": _round_debug(
            replaced_median_error
        ),
        "missing_polygon_replaced_inlier_ratio": _round_debug(
            replaced_inlier_ratio
        ),
        "missing_polygon_replaced_area_score": _round_debug(
            replaced_area_score
        ),
    }


def _missing_none_projection_debug(
    *,
    reason: str,
    homography: np.ndarray | None,
    projected_point_count: int,
    bbox: BBox | None = None,
) -> dict[str, Any]:
    debug = _missing_projection_debug_payload(
        projection="none",
        safety="none",
        reason="projection_failed",
        reason_code="missing_projection_failed",
        missing_polygon_none_reason=reason,
        missing_polygon_none_has_homography=homography is not None,
        missing_polygon_none_projected_point_count=int(projected_point_count),
    )
    if bbox is not None:
        debug["missing_polygon_none_bbox"] = _bbox_debug(bbox)
    return debug


def _record_missing_translation_rescue_reject(
    reject_debug: dict[str, Any] | None,
    reason: str,
) -> None:
    if reject_debug is not None:
        reject_debug["reject_reason"] = reason


def _merge_global_translation_rescue_debug(
    debug: dict[str, Any] | None,
    rescue_debug: dict[str, Any],
    *,
    accepted: bool,
    reject_reason: str | None = None,
) -> None:
    if debug is None:
        return

    reason = reject_reason or str(rescue_debug.get("reject_reason") or "accepted")
    debug.update(
        {
            "missing_polygon_global_translation_rescue_attempted": True,
            "missing_polygon_global_translation_rescue_accepted": bool(accepted),
            "missing_polygon_global_translation_rescue_reject_reason": (
                None if accepted else reason
            ),
        }
    )


def _use_missing_translation_projection(
    state: MissingFallbackState,
    rescue: MissingTranslationRescue,
    *,
    projection: str,
    safety: str,
    source: str,
    **payload_kwargs: Any,
) -> None:
    state.debug.update(
        _missing_translation_rescue_debug_payload(
            projection,
            safety,
            rescue,
            **payload_kwargs,
        )
    )
    state.polygon = rescue.polygon
    state.bbox = rescue.bbox
    state.context_rescue_used = True
    state.source = source


def _local_global_debug_payload(
    local_global_area_score: float,
    local_global_center_factor: float,
) -> dict[str, Any]:
    return {
        "missing_polygon_local_global_area_score": _round_debug(
            local_global_area_score
        ),
        "missing_polygon_local_global_center_factor": _round_debug(
            local_global_center_factor
        ),
    }


def _use_missing_local_displacement_projection(
    state: MissingFallbackState,
    displacement: MissingLocalDisplacement,
    *,
    extra_debug: dict[str, Any] | None = None,
) -> None:
    state.debug = _merge_missing_local_displacement_debug(state.debug, displacement)
    state.debug.update(
        {
            "missing_polygon_fallback_source": "local_displacement",
            "missing_polygon_fallback_reason": "sparse_local_displacement",
        }
    )
    if extra_debug:
        state.debug.update(extra_debug)
    state.polygon = displacement.polygon
    state.bbox = displacement.bbox
    state.context_rescue_used = True
    state.source = "local_displacement"


def _use_missing_global_fallback_projection(
    state: MissingFallbackState,
    *,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    local_global_area_score: float,
    local_global_center_factor: float,
) -> None:
    state.debug.update(
        _missing_projection_debug_payload(
            projection="expected_slot_global_fallback",
            safety="global_fallback",
            include_projection_alias=True,
            missing_polygon_fallback_source="global_fallback",
            missing_polygon_fallback_reason="local_global_disagrees",
            **_local_global_debug_payload(
                local_global_area_score,
                local_global_center_factor,
            ),
        )
    )
    state.polygon = global_polygon
    state.bbox = global_bbox
    state.source = "global_fallback"

def _merge_unsafe_missing_projection_debug(
    missing_debug: dict[str, Any],
) -> dict[str, Any]:
    return {
        **missing_debug,
        "missing_polygon_refined": False,
        **_missing_projection_debug_payload(
            projection="unsafe_hidden",
            safety="unsafe_hidden",
            reason_code="missing_projection_low_confidence",
        ),
    }


def _unsafe_hidden_shadow_debug(
    debug: dict[str, Any],
    *,
    polygon: list[list[float]] | None,
    bbox: BBox | None,
    hidden_reason: str,
    fallback_source: str,
    context_rescue_used: bool,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "missing_polygon_hidden_shadow_enabled": True,
        "missing_polygon_hidden_shadow_available": polygon is not None and bbox is not None,
        "missing_polygon_hidden_shadow_reason": hidden_reason,
        "missing_polygon_hidden_shadow_fallback_source": fallback_source,
        "missing_polygon_hidden_shadow_context_rescue_used": bool(context_rescue_used),
        "missing_polygon_hidden_shadow_projection": (
            _missing_debug_projection_name(debug) or fallback_source or "expected_slot"
        ),
        "missing_polygon_hidden_shadow_projection_safety": str(
            debug.get("missing_polygon_projection_safety") or ""
        ),
    }
    return payload



def _selective_hidden_reject(reason: str) -> dict[str, Any]:
    return {
        "missing_polygon_selective_hidden_release_rejected": True,
        "missing_polygon_selective_hidden_release_reject_reason": reason,
    }

def _try_selective_hidden_global_fallback_release(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None,
    fallback_polygon: list[list[float]] | None,
    fallback_bbox: BBox | None,
    hidden_reason: str,
    fallback_source: str,
    debug: dict[str, Any],
) -> dict[str, Any] | None:
    if hidden_reason != "rich_context_but_no_safe_transform":
        return None
    if fallback_source != "global_fallback":
        return None
    if fallback_polygon is None or fallback_bbox is None or len(fallback_polygon) < 3:
        return None
    if _missing_debug_projection_name(debug) != "expected_slot_global_fallback":
        return None

    slot_support = int(slot.feature_support) if slot is not None else 0
    slot_total = int(slot.feature_total) if slot is not None else 0
    if slot_support < _SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_SUPPORT:
        return _selective_hidden_reject("too_few_slot_features")

    support_ratio = slot_support / max(1, slot_total)
    if (
        slot_total > 0
        and support_ratio
        < _SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_SUPPORT_RATIO
    ):
        return _selective_hidden_reject("low_slot_feature_ratio")

    reference_area_score = reference_area_score_for_expected(
        expected_item,
        slot=slot,
        bbox=fallback_bbox,
    )
    if (
        reference_area_score is not None
        and reference_area_score
        < _SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_REFERENCE_AREA_SCORE
    ):
        return _selective_hidden_reject("bad_reference_area_score")

    local_center_factor = bbox_center_distance_factor(fallback_bbox, expected_item.bbox)
    if local_center_factor > _SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MAX_CENTER_FACTOR:
        return _selective_hidden_reject("large_local_global_center_shift")

    max_overlap = max_overlap_with_other_expected(
        expected_item,
        bbox=fallback_bbox,
        all_expected=all_expected,
    )
    if max_overlap > _SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MAX_OTHER_OVERLAP:
        return _selective_hidden_reject("overlaps_other_expected")

    visible = True
    if projection_data is not None and projection_data.frame_size is not None:
        visible = is_visible_in_frame(
            fallback_bbox,
            projection_data.frame_size,
            min_visible_fraction=float(
                _SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_VISIBLE_FRACTION
            ),
        )
    if not visible:
        return _selective_hidden_reject("outside_frame")

    return {
        "missing_polygon_selective_hidden_release": True,
        "missing_polygon_selective_hidden_release_source": "global_fallback",
        "missing_polygon_selective_hidden_release_hidden_reason": hidden_reason,
    }


def _try_selective_hidden_expected_slot_agreement_release(
    expected_item: ProjectedExpected,
    *,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None,
    fallback_polygon: list[list[float]] | None,
    fallback_bbox: BBox | None,
    global_bbox: BBox | None,
    hidden_reason: str,
    fallback_source: str,
) -> dict[str, Any] | None:
    if hidden_reason != "multi_expected_slot_without_global_consensus":
        return None
    if fallback_source != "expected_slot":
        return None
    if fallback_polygon is None or fallback_bbox is None or len(fallback_polygon) < 3:
        return None
    if global_bbox is None:
        return _selective_hidden_reject("agreement_no_global_candidate")

    area_score = bbox_area_similarity(fallback_bbox, global_bbox)
    center_factor = bbox_center_distance_factor(global_bbox, fallback_bbox)
    if center_factor < _SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_CENTER_FACTOR:
        return _selective_hidden_reject("agreement_shift_too_small")
    if center_factor > _SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MAX_CENTER_FACTOR:
        return _selective_hidden_reject("agreement_shift_too_large")
    if area_score < _SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_AREA_SCORE:
        return _selective_hidden_reject("agreement_bad_area_score")

    max_overlap = max_overlap_with_other_expected(
        expected_item,
        bbox=fallback_bbox,
        all_expected=all_expected,
    )
    if max_overlap > _SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MAX_OTHER_OVERLAP:
        return _selective_hidden_reject("agreement_overlaps_other_expected")

    visible = True
    if projection_data is not None and projection_data.frame_size is not None:
        visible = is_visible_in_frame(
            fallback_bbox,
            projection_data.frame_size,
            min_visible_fraction=float(
                _SELECTIVE_HIDDEN_EXPECTED_SLOT_AGREEMENT_MIN_VISIBLE_FRACTION
            ),
        )
    if not visible:
        return _selective_hidden_reject("agreement_outside_frame")

    return {
        "missing_polygon_selective_hidden_release": True,
        "missing_polygon_selective_hidden_release_source": "expected_slot_agreement",
        "missing_polygon_selective_hidden_release_hidden_reason": hidden_reason,
    }


def _missing_fallback_projection_result(
    *,
    polygon: list[list[float]] | None,
    bbox: BBox | None,
    hidden: bool,
    debug: dict[str, Any],
    final_projection: str | None = None,
) -> MissingFallbackProjection:
    if final_projection is not None:
        debug = finalize_missing_candidate_agreement_debug(
            debug,
            final_projection=final_projection,
        )
    return MissingFallbackProjection(
        polygon=polygon,
        bbox=bbox,
        hidden=hidden,
        debug=debug,
    )


def _resolve_missing_fallback_projection(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None = None,
    all_slots: dict[int, ExpectedSlot] | None = None,
    trusted_anchors: list[TrustedAnchor] | None = None,
) -> MissingFallbackProjection:
    state = MissingFallbackState(
        polygon=expected_item.polygon,
        bbox=expected_item.bbox,
        debug={},
    )

    append_missing_projection_candidate(
        state.debug,
        name="expected_slot",
        expected_item=expected_item,
        polygon=state.polygon,
        bbox=state.bbox,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    state.debug.update(
        {
            "missing_polygon_fallback_source": state.source,
            "missing_polygon_fallback_reason": "fallback_started",
        }
    )

    _apply_missing_global_projection_pipeline(
        state,
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
        all_slots=all_slots,
        trusted_anchors=trusted_anchors,
    )
    _update_missing_fallback_reason(state.debug, state.source)

    _try_prevent_expected_slot_hidden_projection(
        state,
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )

    append_missing_projection_candidate(
        state.debug,
        name=f"selected:{state.source}",
        expected_item=expected_item,
        polygon=state.polygon,
        bbox=state.bbox,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
        selected=True,
    )

    hidden_reason = _missing_fallback_projection_unsafe_reason(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        fallback_bbox=state.bbox,
        context_rescue_used=state.context_rescue_used,
        fallback_source=state.source,
        all_expected=all_expected,
    )
    if hidden_reason is not None:
        return _resolve_hidden_missing_fallback_projection(
            state,
            expected_item,
            slot=slot,
            projection_data=projection_data,
            all_expected=all_expected,
            hidden_reason=hidden_reason,
        )

    return _missing_fallback_projection_result(
        polygon=state.polygon,
        bbox=state.bbox,
        hidden=False,
        debug=state.debug,
        final_projection=_missing_debug_projection_name(state.debug) or state.source,
    )


def _apply_missing_global_projection_pipeline(
    state: MissingFallbackState,
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None,
    all_slots: dict[int, ExpectedSlot] | None,
    trusted_anchors: list[TrustedAnchor] | None,
) -> None:
    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        state.debug.update(
            {
                "missing_polygon_fallback_global_available": False,
                "missing_polygon_fallback_reason": "no_global_projection",
            }
        )
        return

    global_polygon, global_bbox = global_projection
    state.global_candidate_bbox = global_bbox
    append_missing_projection_candidate(
        state.debug,
        name="global_fallback",
        expected_item=expected_item,
        polygon=global_polygon,
        bbox=global_bbox,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    local_global_area_score = bbox_area_similarity(expected_item.bbox, global_bbox)
    local_global_center_factor = bbox_center_distance_factor(
        expected_item.bbox,
        global_bbox,
    )
    local_projection_disagrees = (
        local_global_area_score
        < _thresholds.missing_fallback_min_local_global_area_score
        or local_global_center_factor
        > _thresholds.missing_fallback_max_local_global_center_factor
    )
    state.debug.update(
        {
            "missing_polygon_fallback_global_available": True,
            "missing_polygon_fallback_local_global_disagrees": bool(
                local_projection_disagrees
            ),
        }
    )

    if _apply_primary_missing_rescue_pipeline(
        state,
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        all_expected=all_expected,
        all_slots=all_slots,
        trusted_anchors=trusted_anchors,
    ):
        return

    if not local_projection_disagrees:
        return

    _apply_global_disagreement_missing_rescue_pipeline(
        state,
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        all_expected=all_expected,
        local_global_area_score=local_global_area_score,
        local_global_center_factor=local_global_center_factor,
    )


def _apply_primary_missing_rescue_pipeline(
    state: MissingFallbackState,
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    all_expected: list[ProjectedExpected] | None,
    all_slots: dict[int, ExpectedSlot] | None,
    trusted_anchors: list[TrustedAnchor] | None,
) -> bool:
    rescue = _try_missing_translation_rescue(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        all_expected=all_expected,
    )
    if rescue is not None:
        _use_missing_translation_projection(
            state,
            rescue,
            projection="expected_slot_context_translation_rescue",
            safety="context_translation_rescue",
            source="context_translation_rescue",
        )
        return True

    scene_rescue = _try_missing_scene_translation_rescue(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        all_expected=all_expected,
    )
    if scene_rescue is not None:
        _use_missing_translation_projection(
            state,
            scene_rescue,
            projection="expected_slot_scene_translation_rescue",
            safety="scene_translation_rescue",
            source="scene_translation_rescue",
        )
        return True

    anchor_release_debug: dict[str, Any] = {}
    anchor_release = _try_missing_anchor_release(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        fallback_bbox=state.bbox,
        all_expected=all_expected,
        all_slots=all_slots,
        trusted_anchors=trusted_anchors,
        reject_debug=anchor_release_debug,
    )
    if anchor_release is None:
        state.debug.update(anchor_release_debug)
        return False

    _use_missing_translation_projection(
        state,
        anchor_release,
        projection="expected_slot_anchor_release",
        safety="trusted_anchor_release",
        source="anchor_release",
    )
    return True


def _apply_global_disagreement_missing_rescue_pipeline(
    state: MissingFallbackState,
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    all_expected: list[ProjectedExpected] | None,
    local_global_area_score: float,
    local_global_center_factor: float,
) -> None:
    global_translation_rescue = _try_missing_global_fallback_translation_rescue(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        all_expected=all_expected,
        debug=state.debug,
    )
    if global_translation_rescue is not None:
        _use_missing_translation_projection(
            state,
            global_translation_rescue,
            projection="expected_slot_global_translation_rescue",
            safety="global_fallback_translation_rescue",
            source="global_fallback_translation_rescue",
            fallback_source="global_fallback_translation_rescue",
            fallback_reason="global_fallback_translation_rescue",
            local_global_area_score=local_global_area_score,
            local_global_center_factor=local_global_center_factor,
        )
        return

    local_displacement = _try_missing_local_displacement_field(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    if local_displacement is not None:
        _use_missing_local_displacement_projection(
            state,
            local_displacement,
            extra_debug=_local_global_debug_payload(
                local_global_area_score,
                local_global_center_factor,
            ),
        )
        return

    _use_missing_global_fallback_projection(
        state,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        local_global_area_score=local_global_area_score,
        local_global_center_factor=local_global_center_factor,
    )


def _try_prevent_expected_slot_hidden_projection(
    state: MissingFallbackState,
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None,
) -> None:
    preliminary_hidden_reason = _missing_fallback_projection_unsafe_reason(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        fallback_bbox=state.bbox,
        context_rescue_used=state.context_rescue_used,
        fallback_source=state.source,
        all_expected=all_expected,
    )
    if (
        preliminary_hidden_reason is None
        or state.source != "expected_slot"
        or state.context_rescue_used
    ):
        return

    local_displacement = _try_missing_local_displacement_field(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    if local_displacement is None:
        return

    _use_missing_local_displacement_projection(
        state,
        local_displacement,
        extra_debug={
            "missing_polygon_hidden_prevented_reason": preliminary_hidden_reason,
        },
    )


def _resolve_hidden_missing_fallback_projection(
    state: MissingFallbackState,
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None,
    hidden_reason: str,
) -> MissingFallbackProjection:
    selective_release_debug = _try_selective_hidden_global_fallback_release(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
        fallback_polygon=state.polygon,
        fallback_bbox=state.bbox,
        hidden_reason=hidden_reason,
        fallback_source=state.source,
        debug=state.debug,
    )
    global_release = _build_selective_hidden_release_result(
        state.debug,
        release_debug=selective_release_debug,
        expected_item=expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
        fallback_polygon=state.polygon,
        fallback_bbox=state.bbox,
        hidden_reason=hidden_reason,
        candidate_name="selective_hidden_release:global_fallback",
        projection="expected_slot_global_fallback_hidden_release",
        safety="selective_hidden_global_fallback_release",
        fallback_source="global_fallback",
        fallback_reason="selective_hidden_global_fallback_release",
    )
    if global_release is not None:
        return global_release

    expected_slot_agreement_debug = _try_selective_hidden_expected_slot_agreement_release(
        expected_item,
        projection_data=projection_data,
        all_expected=all_expected,
        fallback_polygon=state.polygon,
        fallback_bbox=state.bbox,
        global_bbox=state.global_candidate_bbox,
        hidden_reason=hidden_reason,
        fallback_source=state.source,
    )
    expected_slot_release = _build_selective_hidden_release_result(
        state.debug,
        release_debug=expected_slot_agreement_debug,
        expected_item=expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
        fallback_polygon=state.polygon,
        fallback_bbox=state.bbox,
        hidden_reason=hidden_reason,
        candidate_name="selective_hidden_release:expected_slot_agreement",
        projection="expected_slot_agreement_hidden_release",
        safety="selective_hidden_expected_slot_agreement_release",
        fallback_source="expected_slot",
        fallback_reason="selective_hidden_expected_slot_agreement_release",
    )
    if expected_slot_release is not None:
        return expected_slot_release

    hidden_debug = {
        **state.debug,
        **_unsafe_hidden_shadow_debug(
            state.debug,
            polygon=state.polygon,
            bbox=state.bbox,
            hidden_reason=hidden_reason,
            fallback_source=state.source,
            context_rescue_used=state.context_rescue_used,
        ),
        "missing_polygon_hidden_reason": hidden_reason,
    }
    if selective_release_debug is not None:
        hidden_debug.update(selective_release_debug)
    if expected_slot_agreement_debug is not None:
        hidden_debug.update(expected_slot_agreement_debug)
    return _missing_fallback_projection_result(
        polygon=None,
        bbox=None,
        hidden=True,
        debug=hidden_debug,
        final_projection="unsafe_hidden",
    )


def _build_selective_hidden_release_result(
    debug: dict[str, Any],
    *,
    release_debug: dict[str, Any] | None,
    expected_item: ProjectedExpected,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None,
    fallback_polygon: list[list[float]] | None,
    fallback_bbox: BBox | None,
    hidden_reason: str,
    candidate_name: str,
    projection: str,
    safety: str,
    fallback_source: str,
    fallback_reason: str,
) -> MissingFallbackProjection | None:
    if not release_debug or not release_debug.get("missing_polygon_selective_hidden_release"):
        return None

    append_missing_projection_candidate(
        debug,
        name=candidate_name,
        expected_item=expected_item,
        polygon=fallback_polygon,
        bbox=fallback_bbox,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
        selected=True,
        hidden_release=True,
        reason=hidden_reason,
    )
    return _missing_fallback_projection_result(
        polygon=fallback_polygon,
        bbox=fallback_bbox,
        hidden=False,
        debug={
            **debug,
            **release_debug,
            **_missing_projection_debug_payload(
                projection=projection,
                safety=safety,
                include_projection_alias=True,
                missing_polygon_fallback_source=fallback_source,
                missing_polygon_fallback_reason=fallback_reason,
                reason_code="feature_slot_unconfirmed_without_yolo",
            ),
        },
        final_projection=projection,
    )


def _update_missing_fallback_reason(
    debug: dict[str, Any],
    fallback_source: str,
) -> None:
    debug["missing_polygon_fallback_source"] = fallback_source
    if fallback_source == "expected_slot":
        current_reason = debug.get("missing_polygon_fallback_reason")
        if current_reason in {None, "fallback_started"}:
            debug["missing_polygon_fallback_reason"] = (
                "kept_expected_slot_no_safe_rescue"
            )
        return

    fallback_reasons = {
        "context_translation_rescue": "context_translation_rescue",
        "scene_translation_rescue": "scene_translation_rescue",
        "anchor_release": "anchor_release",
        "global_fallback": "local_global_disagrees",
        "global_fallback_translation_rescue": "global_fallback_translation_rescue",
        "local_displacement": "sparse_local_displacement",
    }
    reason = fallback_reasons.get(fallback_source)
    if reason is not None:
        debug["missing_polygon_fallback_reason"] = reason

def _missing_translation_rescue_debug_payload(
    projection: str,
    safety: str,
    rescue: MissingTranslationRescue,
    *,
    fallback_source: str | None = None,
    fallback_reason: str | None = None,
    local_global_area_score: float | None = None,
    local_global_center_factor: float | None = None,
) -> dict[str, Any]:
    payload = _missing_projection_debug_payload(
        projection=projection,
        safety=safety,
        include_projection_alias=True,
        missing_polygon_inliers=rescue.inlier_count,
        missing_polygon_inlier_ratio=_round_debug(rescue.inlier_ratio),
        missing_polygon_translation_shift_factor=_round_debug(rescue.shift_factor),
    )
    if local_global_area_score is not None:
        payload["missing_polygon_local_global_area_score"] = _round_debug(
            local_global_area_score
        )
    if local_global_center_factor is not None:
        payload["missing_polygon_local_global_center_factor"] = _round_debug(
            local_global_center_factor
        )
    if rescue.source_counts is not None:
        payload["missing_polygon_anchor_sources"] = rescue.source_counts
    if fallback_source is not None:
        payload["missing_polygon_fallback_source"] = fallback_source
    if fallback_reason is not None:
        payload["missing_polygon_fallback_reason"] = fallback_reason
    return payload


def _try_missing_translation_rescue(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    all_expected: list[ProjectedExpected] | None = None,
    min_support_override: int | None = None,
    min_inlier_ratio_override: float | None = None,
    max_residual_error_override: float | None = None,
    max_shift_factor_override: float | None = None,
    min_context_spread_override: float | None = None,
    min_search_containment_override: float | None = None,
    max_other_overlap: float | None = None,
    reject_debug: dict[str, Any] | None = None,
) -> MissingTranslationRescue | None:
    if slot is None or projection_data is None:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "no_slot_or_projection_data",
        )
        return None
    if projection_data.global_homography is None:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "no_global_homography",
        )
        return None

    if min_support_override is None:
        min_support = max(
            _thresholds.missing_polygon_min_feature_support,
            _thresholds.missing_rescue_translation_min_support,
            3,
        )
    else:
        min_support = max(2, int(min_support_override))
    _record_missing_translation_rescue_reject(
        reject_debug,
        "started",
    )

    local_reference, local_frame = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    local_point_count = int(len(local_reference))
    if local_point_count < min_support or len(local_reference) != len(local_frame):
        _record_missing_translation_rescue_reject(
            reject_debug,
            "too_few_local_points",
        )
        return None

    projected_reference = project_points_with_homography(
        local_reference,
        projection_data.global_homography,
    )
    if projected_reference is None or len(projected_reference) != len(local_frame):
        _record_missing_translation_rescue_reject(
            reject_debug,
            "projection_failed",
        )
        return None

    residuals = local_frame.astype(np.float32) - projected_reference.astype(np.float32)
    if residuals.ndim != 2 or residuals.shape[1] < 2 or not np.isfinite(residuals).all():
        _record_missing_translation_rescue_reject(
            reject_debug,
            "invalid_residuals",
        )
        return None

    median_residual = np.median(residuals[:, :2], axis=0).astype(np.float32)
    if not np.isfinite(median_residual).all():
        _record_missing_translation_rescue_reject(
            reject_debug,
            "invalid_median_residual",
        )
        return None

    residual_errors = np.linalg.norm(residuals[:, :2] - median_residual[None, :], axis=1)
    if len(residual_errors) == 0 or not np.isfinite(residual_errors).all():
        _record_missing_translation_rescue_reject(
            reject_debug,
            "invalid_residual_errors",
        )
        return None

    threshold = float(
        _thresholds.missing_rescue_translation_max_residual_error
        if max_residual_error_override is None
        else max_residual_error_override
    )
    inlier_mask = residual_errors <= threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    candidate_count = int(len(residual_errors))
    if inlier_count < min_support:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "too_few_inliers",
        )
        return None

    inlier_ratio = inlier_count / max(1, candidate_count)
    min_inlier_ratio = (
        _thresholds.missing_rescue_translation_min_inlier_ratio
        if min_inlier_ratio_override is None
        else float(min_inlier_ratio_override)
    )
    if inlier_ratio < min_inlier_ratio:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "low_inlier_ratio",
        )
        return None

    inlier_reference = local_reference[inlier_mask]
    reference_bbox = slot.reference_bbox or bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is not None:
        spread = _missing_context_spread_score(inlier_reference, reference_bbox)
        min_spread = (
            max(0.07, _thresholds.missing_polygon_context_min_spread * 0.55)
            if min_context_spread_override is None
            else float(min_context_spread_override)
        )
        if spread < min_spread:
            _record_missing_translation_rescue_reject(
                reject_debug,
                "low_context_spread",
            )
            return None
    else:
        spread = 0.0

    median_error = float(np.median(residual_errors[inlier_mask]))
    if median_error > threshold:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "high_median_error",
        )
        return None

    shift_x = float(median_residual[0])
    shift_y = float(median_residual[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / max(1.0, bbox_diag(global_bbox))
    max_shift_factor = (
        _thresholds.missing_rescue_translation_max_shift_factor
        if max_shift_factor_override is None
        else float(max_shift_factor_override)
    )
    if shift_factor > max_shift_factor:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "large_shift",
        )
        return None

    polygon = translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    if len(polygon) < 3:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "invalid_polygon",
        )
        return None

    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "invalid_bbox",
        )
        return None
    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
    ):
        _record_missing_translation_rescue_reject(
            reject_debug,
            "outside_frame",
        )
        return None

    area_score = bbox_area_similarity(bbox, global_bbox)
    if area_score < 0.92:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "area_changed",
        )
        return None
    center_factor = bbox_center_distance_factor(bbox, global_bbox)
    if center_factor > max(0.05, max_shift_factor):
        _record_missing_translation_rescue_reject(
            reject_debug,
            "center_far_from_global",
        )
        return None

    search_containment = bbox_containment(bbox, slot.search_bbox)
    min_search_containment = (
        0.05
        if min_search_containment_override is None
        else float(min_search_containment_override)
    )
    if search_containment < min_search_containment:
        _record_missing_translation_rescue_reject(
            reject_debug,
            "low_search_containment",
        )
        return None

    if max_other_overlap is not None and _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=bbox,
        all_expected=all_expected,
        max_overlap=float(max_other_overlap),
    ):
        _record_missing_translation_rescue_reject(
            reject_debug,
            "bad_overlap",
        )
        return None

    _record_missing_translation_rescue_reject(
        reject_debug,
        "accepted",
    )
    return MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
    )



def _try_missing_local_displacement_field(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None = None,
) -> MissingLocalDisplacement | None:
    if slot is None or projection_data is None:
        return None
    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return None

    min_support = max(3, int(_thresholds.missing_local_displacement_min_support))
    local_reference, local_frame = _missing_polygon_refinement_points(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        all_expected=all_expected,
    )
    local_point_count = int(len(local_reference))
    if local_point_count < min_support:
        return None

    base_affine = _reference_to_expected_affine(expected_item)
    if base_affine is None:
        return None
    base_reference_projected = _project_points_by_affine(
        local_reference,
        base_affine,
    )
    if base_reference_projected is None or len(base_reference_projected) != local_point_count:
        return None

    residuals = (local_frame - base_reference_projected).astype(np.float32)
    if not np.isfinite(residuals).all():
        return None

    local_diag = max(1.0, bbox_diag(slot.projected_bbox))
    max_shift = max(
        1.0,
        local_diag * float(_thresholds.missing_local_displacement_max_shift_factor),
    )
    residual_lengths = np.linalg.norm(residuals, axis=1)
    finite_mask = np.isfinite(residual_lengths) & (residual_lengths <= max_shift)
    if int(np.count_nonzero(finite_mask)) < min_support:
        return None

    local_reference = local_reference[finite_mask]
    residuals = residuals[finite_mask]
    kept_mask = _local_displacement_residual_consensus_mask(
        local_reference,
        residuals,
        max_residual_error=float(
            _thresholds.missing_local_displacement_max_residual_error
        ),
    )
    if kept_mask is None or int(np.count_nonzero(kept_mask)) < min_support:
        return None

    kept_reference = local_reference[kept_mask]
    kept_residuals = residuals[kept_mask]
    kept_count = int(len(kept_reference))
    reference_spread = _missing_context_spread_score(kept_reference, reference_bbox)
    if reference_spread < _thresholds.missing_local_displacement_min_spread:
        return None

    quadrant_count = _missing_context_quadrant_count(kept_reference, reference_bbox)
    if quadrant_count < _thresholds.missing_local_displacement_min_quadrants:
        return None

    base_points = np.asarray(expected_item.polygon, dtype=np.float32)
    reference_vertices = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )
    if (
        len(base_points) < 3
        or len(reference_vertices) != len(base_points)
        or reference_vertices.ndim != 2
        or reference_vertices.shape[1] < 2
    ):
        return None

    nearest_count = max(
        3,
        min(int(_thresholds.missing_local_displacement_nearest_points), kept_count),
    )
    vertex_residuals = _interpolate_local_displacements(
        reference_vertices[:, :2],
        kept_reference,
        kept_residuals,
        nearest_count=nearest_count,
    )
    if vertex_residuals is None:
        return None

    displaced_points = base_points[:, :2] + vertex_residuals
    if not np.isfinite(displaced_points).all():
        return None

    vertex_shifts = np.linalg.norm(displaced_points - base_points[:, :2], axis=1)
    max_vertex_shift = float(np.max(vertex_shifts)) if len(vertex_shifts) else 0.0
    max_vertex_shift_factor = max_vertex_shift / local_diag
    if max_vertex_shift_factor > _thresholds.missing_local_displacement_max_shift_factor:
        return None

    polygon = [[float(x), float(y)] for x, y in displaced_points.tolist()]
    if len(polygon) < 3 or not polygon_has_usable_area(polygon):
        return None

    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return None
    if (
        projection_data.frame_size is not None
        and not is_visible_in_frame(
            bbox,
            projection_data.frame_size,
            min_visible_fraction=0.05,
        )
    ):
        return None

    containment = bbox_containment(bbox, slot.search_bbox)
    if containment < _thresholds.missing_local_displacement_min_search_containment:
        return None

    area_score = bbox_area_similarity(bbox, expected_item.bbox)
    if area_score < _thresholds.missing_local_displacement_min_local_area_score:
        return None

    center_drift_factor = bbox_center_distance_factor(bbox, expected_item.bbox)
    if center_drift_factor > _thresholds.missing_local_displacement_max_local_center_factor:
        return None

    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=bbox,
        all_expected=all_expected,
        max_overlap=float(_thresholds.missing_local_displacement_max_other_overlap),
    ):
        return None

    axis_angle_delta, major_length_ratio = polygon_axis_delta(polygon, expected_item.polygon)
    if axis_angle_delta is not None and axis_angle_delta > 22.0:
        return None
    if major_length_ratio is not None and not (0.62 <= major_length_ratio <= 1.62):
        return None

    return MissingLocalDisplacement(
        polygon=polygon,
        bbox=bbox,
        area_score=float(area_score),
        center_drift_factor=float(center_drift_factor),
    )



def _reference_to_expected_affine(
    expected_item: ProjectedExpected,
) -> np.ndarray | None:
    source = np.asarray(expected_item.item.reference_polygon, dtype=np.float32)
    target = np.asarray(expected_item.polygon, dtype=np.float32)
    if (
        source.ndim != 2
        or target.ndim != 2
        or source.shape[1] < 2
        or target.shape[1] < 2
        or len(source) != len(target)
        or len(source) < 3
        or not np.isfinite(source).all()
        or not np.isfinite(target).all()
    ):
        return None
    try:
        affine, _inliers = cv2.estimateAffinePartial2D(
            source[:, :2],
            target[:, :2],
            method=cv2.LMEDS,
            refineIters=10,
        )
    except cv2.error:
        return None
    if affine is None or affine.shape != (2, 3) or not np.isfinite(affine).all():
        return None
    return affine.astype(np.float32)


def _project_points_by_affine(
    points: np.ndarray,
    affine: np.ndarray,
) -> np.ndarray | None:
    if len(points) == 0:
        return None
    source = np.asarray(points, dtype=np.float32)
    if source.ndim != 2 or source.shape[1] < 2 or not np.isfinite(source).all():
        return None
    try:
        projected = cv2.transform(
            source[:, :2].reshape(-1, 1, 2),
            affine.astype(np.float32),
        ).reshape(-1, 2)
    except cv2.error:
        return None
    if not np.isfinite(projected).all():
        return None
    return projected.astype(np.float32)


def _local_displacement_residual_consensus_mask(
    reference_points: np.ndarray,
    residuals: np.ndarray,
    *,
    max_residual_error: float,
) -> np.ndarray | None:
    count = int(len(reference_points))
    if count == 0 or count != len(residuals):
        return None
    if count < 3:
        return np.ones(count, dtype=bool)

    neighbor_count = max(3, min(7, count - 1))
    keep = np.zeros(count, dtype=bool)
    for idx in range(count):
        distances = np.linalg.norm(reference_points - reference_points[idx], axis=1)
        order = np.argsort(distances)
        neighbors = [int(i) for i in order if int(i) != idx][:neighbor_count]
        if not neighbors:
            continue
        local_median = np.median(residuals[neighbors], axis=0)
        error = float(np.linalg.norm(residuals[idx] - local_median))
        if error <= max_residual_error:
            keep[idx] = True

    if int(np.count_nonzero(keep)) < 3:
        return None
    return keep


def _interpolate_local_displacements(
    target_reference_points: np.ndarray,
    source_reference_points: np.ndarray,
    source_residuals: np.ndarray,
    *,
    nearest_count: int,
) -> np.ndarray | None:
    if (
        len(target_reference_points) == 0
        or len(source_reference_points) == 0
        or len(source_reference_points) != len(source_residuals)
    ):
        return None

    interpolated: list[np.ndarray] = []
    power = 1.65
    eps = 1e-3
    for target in target_reference_points:
        distances = np.linalg.norm(source_reference_points - target, axis=1)
        order = np.argsort(distances)[:nearest_count]
        if len(order) == 0:
            return None
        nearest_distances = distances[order]
        nearest_residuals = source_residuals[order]
        if float(np.min(nearest_distances)) <= eps:
            interpolated.append(nearest_residuals[int(np.argmin(nearest_distances))])
            continue
        weights = 1.0 / np.power(nearest_distances + eps, power)
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0 or not np.isfinite(weight_sum):
            return None
        residual = np.sum(nearest_residuals * weights[:, None], axis=0) / weight_sum
        interpolated.append(residual.astype(np.float32))

    result = np.asarray(interpolated, dtype=np.float32)
    if result.ndim != 2 or result.shape[1] != 2 or not np.isfinite(result).all():
        return None
    return result


def _merge_missing_local_displacement_debug(
    missing_debug: dict[str, Any],
    displacement: MissingLocalDisplacement,
) -> dict[str, Any]:
    return {
        **missing_debug,
        "projection": "expected_slot_local_displacement",
        "missing_polygon_projection": "expected_slot_local_displacement",
        "missing_polygon_projection_safety": "sparse_local_displacement",
        "missing_polygon_area_score": _round_debug(displacement.area_score),
        "missing_polygon_center_drift_factor": _round_debug(
            displacement.center_drift_factor
        ),
    }


def _try_missing_global_fallback_translation_rescue(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    all_expected: list[ProjectedExpected] | None = None,
    debug: dict[str, Any] | None = None,
) -> MissingTranslationRescue | None:
    diagnostics_enabled = debug is not None
    rescue_debug: dict[str, Any] = {}

    if slot is None or projection_data is None:
        _record_missing_translation_rescue_reject(
            rescue_debug,
            "no_slot_or_projection_data",
        )
        if diagnostics_enabled:
            _merge_global_translation_rescue_debug(
                debug,
                rescue_debug,
                accepted=False,
            )
        return None

    rescue = _try_missing_translation_rescue(
        expected_item,
        slot=slot,
        projection_data=projection_data,
        global_polygon=global_polygon,
        global_bbox=global_bbox,
        all_expected=all_expected,
        min_support_override=_thresholds.missing_global_fallback_translation_rescue_min_support,
        min_inlier_ratio_override=_thresholds.missing_global_fallback_translation_rescue_min_inlier_ratio,
        max_residual_error_override=_thresholds.missing_global_fallback_translation_rescue_max_residual_error,
        max_shift_factor_override=_thresholds.missing_global_fallback_translation_rescue_max_shift_factor,
        min_context_spread_override=0.055,
        min_search_containment_override=_thresholds.missing_global_fallback_translation_rescue_min_search_containment,
        max_other_overlap=_thresholds.missing_global_fallback_translation_rescue_max_other_overlap,
        reject_debug=rescue_debug if diagnostics_enabled else None,
    )
    if rescue is None:
        if diagnostics_enabled:
            _merge_global_translation_rescue_debug(
                debug,
                rescue_debug or {"reject_reason": "translation_rescue_rejected"},
                accepted=False,
            )
        return None

    local_area_score = bbox_area_similarity(rescue.bbox, expected_item.bbox)
    rescue_debug["local_area_score"] = _round_debug(local_area_score)
    if (
        local_area_score
        < _thresholds.missing_global_fallback_translation_rescue_min_local_area_score
    ):
        _record_missing_translation_rescue_reject(
            rescue_debug,
            "local_area_score_low",
        )
        if diagnostics_enabled:
            _merge_global_translation_rescue_debug(
                debug,
                rescue_debug,
                accepted=False,
            )
        return None

    local_center_factor = bbox_center_distance_factor(rescue.bbox, expected_item.bbox)
    rescue_debug["local_center_factor"] = _round_debug(local_center_factor)
    if (
        local_center_factor
        > _thresholds.missing_global_fallback_translation_rescue_max_local_center_factor
    ):
        _record_missing_translation_rescue_reject(
            rescue_debug,
            "local_center_factor_high",
        )
        if diagnostics_enabled:
            _merge_global_translation_rescue_debug(
                debug,
                rescue_debug,
                accepted=False,
            )
        return None

    rescue.source_counts = {
        "global_fallback_translation_rescue": rescue.inlier_count,
        "global_fallback_translation_candidates": rescue.candidate_count,
    }
    if diagnostics_enabled:
        _merge_global_translation_rescue_debug(
            debug,
            rescue_debug,
            accepted=True,
        )
    return rescue


def _try_missing_scene_translation_rescue(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    all_expected: list[ProjectedExpected] | None = None,
) -> MissingTranslationRescue | None:
    if projection_data is None or projection_data.global_homography is None:
        return None

    reference_points = as_match_points(projection_data.reference_points)
    frame_points = as_match_points(projection_data.frame_points)
    if reference_points is None or frame_points is None:
        return None
    if len(reference_points) != len(frame_points):
        return None

    reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return None

    min_support = max(
        _thresholds.missing_scene_rescue_min_support,
        _thresholds.missing_polygon_min_feature_support,
        4,
    )

    reference_window = expand_bbox(
        reference_bbox,
        factor=_thresholds.missing_scene_rescue_context_expansion,
    )
    frame_window = expand_bbox(
        global_bbox,
        factor=_thresholds.missing_scene_rescue_context_expansion,
        frame_size=projection_data.frame_size,
    )
    reference_exclusion_zones, frame_exclusion_zones = _all_expected_context_exclusion_zones(
        expected_item,
        all_expected=all_expected,
        projection_data=projection_data,
    )

    selected_reference: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []
    projected_reference = project_points_with_homography(
        reference_points,
        projection_data.global_homography,
    )
    if projected_reference is None or len(projected_reference) != len(frame_points):
        return None

    residual_prefilter = max(
        _thresholds.missing_scene_rescue_max_residual_error * 5.0,
        bbox_diag(global_bbox) * 0.85,
        32.0,
    )

    for reference_point, frame_point, projected_point in zip(
        reference_points,
        frame_points,
        projected_reference,
        strict=True,
    ):
        if not point_in_bbox(reference_point, reference_window):
            continue
        if point_in_any_bbox(reference_point, reference_exclusion_zones):
            continue
        if not point_in_bbox(frame_point, frame_window):
            continue
        if point_in_any_bbox(frame_point, frame_exclusion_zones):
            continue
        if float(np.linalg.norm(frame_point[:2] - projected_point[:2])) > residual_prefilter:
            continue

        selected_reference.append(reference_point)
        selected_frame.append(frame_point)

    if len(selected_reference) < min_support:
        return None

    local_reference = np.asarray(selected_reference, dtype=np.float32)
    local_frame = np.asarray(selected_frame, dtype=np.float32)

    max_points = max(_MISSING_POLYGON_MAX_LOCAL_POINTS, 160)
    if len(local_reference) > max_points:
        reference_center = np.asarray(bbox_center(reference_bbox), dtype=np.float32)
        projected_center = np.asarray(bbox_center(global_bbox), dtype=np.float32)
        reference_distance = np.linalg.norm(local_reference - reference_center, axis=1)
        frame_distance = np.linalg.norm(local_frame - projected_center, axis=1)
        indices = np.argsort(reference_distance + frame_distance)[:max_points]
        local_reference = local_reference[indices]
        local_frame = local_frame[indices]

    projected_local = project_points_with_homography(
        local_reference,
        projection_data.global_homography,
    )
    if projected_local is None or len(projected_local) != len(local_frame):
        return None

    residuals = local_frame.astype(np.float32) - projected_local.astype(np.float32)
    if residuals.ndim != 2 or residuals.shape[1] < 2 or not np.isfinite(residuals).all():
        return None

    median_residual = np.median(residuals[:, :2], axis=0).astype(np.float32)
    if not np.isfinite(median_residual).all():
        return None

    residual_errors = np.linalg.norm(residuals[:, :2] - median_residual[None, :], axis=1)
    if len(residual_errors) == 0 or not np.isfinite(residual_errors).all():
        return None

    threshold = float(_thresholds.missing_scene_rescue_max_residual_error)
    inlier_mask = residual_errors <= threshold
    inlier_count = int(np.count_nonzero(inlier_mask))
    candidate_count = int(len(residual_errors))
    if inlier_count < min_support:
        return None

    inlier_ratio = inlier_count / max(1, candidate_count)
    if inlier_ratio < _thresholds.missing_scene_rescue_min_inlier_ratio:
        return None

    inlier_reference = local_reference[inlier_mask]
    spread = _missing_context_spread_score(inlier_reference, reference_bbox)
    if spread < _thresholds.missing_scene_rescue_min_spread:
        return None

    inlier_residuals = residuals[inlier_mask, :2]
    median_residual = np.median(inlier_residuals, axis=0).astype(np.float32)
    residual_errors = np.linalg.norm(inlier_residuals - median_residual[None, :], axis=1)
    if len(residual_errors) == 0 or not np.isfinite(residual_errors).all():
        return None

    median_error = float(np.median(residual_errors))
    if median_error > threshold:
        return None

    shift_x = float(median_residual[0])
    shift_y = float(median_residual[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / max(1.0, bbox_diag(global_bbox))
    if shift_factor > _thresholds.missing_scene_rescue_max_shift_factor:
        return None

    polygon = translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    if len(polygon) < 3:
        return None

    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return None
    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
    ):
        return None

    if bbox_area_similarity(bbox, global_bbox) < 0.92:
        return None
    if bbox_center_distance_factor(bbox, global_bbox) > max(
        0.05,
        _thresholds.missing_scene_rescue_max_shift_factor,
    ):
        return None
    if slot is not None and bbox_containment(bbox, slot.search_bbox) < 0.03:
        return None
    if _missing_rescue_overlaps_other_expected(
        expected_item,
        bbox=bbox,
        all_expected=all_expected,
    ):
        return None

    return MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
    )


def _try_missing_anchor_release(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    fallback_bbox: BBox | None,
    all_expected: list[ProjectedExpected] | None = None,
    all_slots: dict[int, ExpectedSlot] | None = None,
    trusted_anchors: list[TrustedAnchor] | None = None,
    reject_debug: dict[str, Any] | None = None,
) -> MissingTranslationRescue | None:
    update_anchor_release_debug(
        reject_debug,
        attempted=True,
        total_anchors=len(trusted_anchors or []),
    )

    def reject(reason: str, **fields: Any) -> None:
        set_anchor_release_reject(reject_debug, reason, **fields)
        return None

    if slot is None or projection_data is None or projection_data.global_homography is None:
        return reject("anchor_release_rejected_unavailable_inputs")
    if not all_expected or len(all_expected) <= 1 or not all_slots:
        return reject("anchor_release_rejected_not_multi_object")
    if not trusted_anchors:
        return reject("anchor_release_rejected_no_trusted_anchors")

    min_anchors = max(1, int(_thresholds.missing_anchor_release_min_anchors))
    max_anchor_error = float(_thresholds.missing_anchor_release_max_anchor_error)
    current_center = np.asarray(bbox_center(global_bbox), dtype=np.float32)
    current_diag = max(1.0, bbox_diag(global_bbox))
    max_neighbor_distance = (
        _thresholds.missing_anchor_release_max_neighbor_distance_factor
        * current_diag
    )

    all_candidates: list[tuple[TrustedAnchor, np.ndarray, float, float]] = []
    rejected_self = 0
    rejected_bad_residual = 0
    rejected_shift = 0
    rejected_overlap = 0
    candidate_sources: dict[str, int] = {}

    for anchor in trusted_anchors:
        if anchor.expected_index == expected_item.index:
            rejected_self += 1
            continue

        residual = np.asarray(anchor.residual, dtype=np.float32)
        if residual.shape != (2,) or not np.isfinite(residual).all():
            rejected_bad_residual += 1
            continue

        residual_shift_factor = float(np.linalg.norm(residual) / current_diag)
        if (
            residual_shift_factor
            > _thresholds.missing_anchor_release_max_shift_factor
        ):
            rejected_shift += 1
            continue

        overlap_with_current = max(
            bbox_iou(anchor.local_bbox, global_bbox),
            bbox_containment(anchor.local_bbox, global_bbox),
            bbox_containment(global_bbox, anchor.local_bbox),
        )
        if (
            overlap_with_current
            > _thresholds.missing_anchor_release_max_other_overlap
        ):
            rejected_overlap += 1
            continue

        anchor_center = np.asarray(bbox_center(anchor.global_bbox), dtype=np.float32)
        distance = float(np.linalg.norm(anchor_center - current_center))
        distance_weight = 1.0 / max(1.0, distance)
        source_weight = _trusted_anchor_source_weight(anchor.source)
        weight = max(0.01, float(anchor.weight) * distance_weight * source_weight)
        all_candidates.append((anchor, residual, weight, distance))
        candidate_sources[anchor.source] = candidate_sources.get(anchor.source, 0) + 1

    update_anchor_release_debug(
        reject_debug,
        candidate_count=len(all_candidates),
        rejected_self=rejected_self,
        rejected_bad_residual=rejected_bad_residual,
        rejected_shift=rejected_shift,
        rejected_overlap=rejected_overlap,
        candidate_sources=candidate_sources,
    )

    if not all_candidates:
        return reject("anchor_release_rejected_no_candidates")

    strong_candidates = [
        item
        for item in all_candidates
        if item[0].source != "weak_local_global_slot_hint"
    ]
    candidate_pool = (
        strong_candidates
        if len(strong_candidates) >= min_anchors
        else all_candidates
    )
    update_anchor_release_debug(
        reject_debug,
        strong_candidate_count=len(strong_candidates),
        weak_candidate_count=len(all_candidates) - len(strong_candidates),
        weak_candidate_pool_used=(candidate_pool is all_candidates),
    )

    neighbor_candidates = [
        (anchor, residual, weight)
        for anchor, residual, weight, distance in candidate_pool
        if distance <= max_neighbor_distance
    ]
    update_anchor_release_debug(
        reject_debug,
        neighbor_candidate_count=len(neighbor_candidates),
        max_neighbor_distance=_round_debug(max_neighbor_distance),
    )
    if len(neighbor_candidates) >= min_anchors:
        candidates = neighbor_candidates
    else:
        candidates = [
            (anchor, residual, weight)
            for anchor, residual, weight, _ in candidate_pool
        ]

    update_anchor_release_debug(
        reject_debug,
        selected_candidate_count=len(candidates),
        min_anchors=min_anchors,
    )

    if len(candidates) < min_anchors:
        if not _anchor_release_allows_single_anchor(
            candidates,
            slot=slot,
            global_bbox=global_bbox,
            fallback_bbox=fallback_bbox,
        ):
            return reject(
                "anchor_release_rejected_single_anchor_guard",
                candidates=len(candidates),
                min_anchors=min_anchors,
            )
        min_anchors = 1

    consensus = _select_anchor_release_consensus(
        candidates,
        min_anchors=min_anchors,
        max_anchor_error=max_anchor_error,
        global_bbox=global_bbox,
    )
    if consensus is None:
        return reject(
            "anchor_release_rejected_high_dispersion",
            candidates=len(candidates),
            min_anchors=min_anchors,
        )

    update_anchor_release_debug(
        reject_debug,
        inliers=consensus.inlier_count,
        inlier_ratio=_round_debug(consensus.inlier_ratio),
        median_error=_round_debug(consensus.median_error),
        dispersion=_round_debug(consensus.dispersion),
    )

    inlier_anchors = [
        candidate[0]
        for candidate, is_inlier in zip(candidates, consensus.inlier_mask, strict=True)
        if bool(is_inlier)
    ]
    source_counts = trusted_anchor_source_counts(inlier_anchors)
    weak_slot_anchor_count = sum(
        1 for anchor in inlier_anchors if anchor.source == "weak_local_global_slot_hint"
    )
    weak_slot_only = weak_slot_anchor_count > 0 and weak_slot_anchor_count == len(inlier_anchors)
    if weak_slot_only:
        if (
            weak_slot_anchor_count
            < _thresholds.missing_anchor_release_weak_slot_min_inliers
        ):
            return reject(
                "anchor_release_rejected_weak_slot_guard",
                weak_slot_anchors=weak_slot_anchor_count,
            )
        if (
            consensus.inlier_ratio
            < _thresholds.missing_anchor_release_weak_slot_min_inlier_ratio
        ):
            return reject(
                "anchor_release_rejected_weak_slot_guard",
                weak_slot_inlier_ratio=_round_debug(consensus.inlier_ratio),
            )
        if (
            consensus.dispersion
            > _thresholds.missing_anchor_release_weak_slot_max_dispersion_factor
        ):
            return reject(
                "anchor_release_rejected_weak_slot_guard",
                weak_slot_dispersion=_round_debug(consensus.dispersion),
            )
    update_anchor_release_debug(
        reject_debug,
        weak_slot_anchor_count=weak_slot_anchor_count,
        weak_slot_only=weak_slot_only,
    )
    source_counts = trusted_anchor_source_counts(inlier_anchors)
    single_anchor = consensus.inlier_count == 1
    median_error = consensus.median_error
    if single_anchor:
        anchor_median_error = max(anchor.median_error for anchor in inlier_anchors)
        median_error = max(median_error, float(anchor_median_error))
    if median_error > max_anchor_error:
        return reject(
            "anchor_release_rejected_high_dispersion",
            median_error=_round_debug(median_error),
            max_anchor_error=_round_debug(max_anchor_error),
        )

    shift = consensus.shift
    shift_x = float(shift[0])
    shift_y = float(shift[1])
    shift_length = float(np.hypot(shift_x, shift_y))
    shift_factor = shift_length / current_diag
    update_anchor_release_debug(
        reject_debug,
        shift_x=_round_debug(shift_x),
        shift_y=_round_debug(shift_y),
        shift_factor=_round_debug(shift_factor),
    )
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return reject(
            "anchor_release_rejected_large_shift",
            shift_factor=_round_debug(shift_factor),
        )

    polygon = translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    if len(polygon) < 3:
        return reject("anchor_release_rejected_bad_polygon")

    bbox = bbox_from_polygon(polygon)
    if bbox is None:
        return reject("anchor_release_rejected_bad_bbox")

    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=_thresholds.missing_anchor_release_min_visible_fraction,
    ):
        return reject("anchor_release_rejected_bad_overlap")

    if fallback_bbox is not None:
        local_area_score = bbox_area_similarity(bbox, fallback_bbox)
        local_center_factor = bbox_center_distance_factor(bbox, fallback_bbox)
        update_anchor_release_debug(
            reject_debug,
            local_area_score=_round_debug(local_area_score),
            local_center_factor=_round_debug(local_center_factor),
        )
        if (
            local_area_score
            < _thresholds.missing_anchor_release_min_local_area_score
        ):
            return reject(
                "anchor_release_rejected_bad_overlap",
                local_area_score=_round_debug(local_area_score),
            )
        if (
            local_center_factor
            > _thresholds.missing_anchor_release_max_local_center_factor
        ):
            return reject(
                "anchor_release_rejected_bad_overlap",
                local_center_factor=_round_debug(local_center_factor),
            )

        if single_anchor:
            single_min_area = max(
                _thresholds.missing_anchor_release_min_local_area_score,
                _thresholds.missing_anchor_release_single_min_local_area_score,
            )
            single_max_center = min(
                _thresholds.missing_anchor_release_max_local_center_factor,
                _thresholds.missing_anchor_release_single_max_local_center_factor,
            )
            if local_area_score < single_min_area:
                return reject(
                    "anchor_release_rejected_single_anchor_guard",
                    local_area_score=_round_debug(local_area_score),
                )
            if local_center_factor > single_max_center:
                return reject(
                    "anchor_release_rejected_single_anchor_guard",
                    local_center_factor=_round_debug(local_center_factor),
                )

    if (
        single_anchor
        and shift_factor > _thresholds.missing_anchor_release_single_max_shift_factor
    ):
        return reject(
            "anchor_release_rejected_single_anchor_guard",
            shift_factor=_round_debug(shift_factor),
        )

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is not None:
        reference_area_score = bbox_area_similarity(bbox, reference_bbox)
        update_anchor_release_debug(
            reject_debug,
            reference_area_score=_round_debug(reference_area_score),
        )
        if reference_area_score < 0.12:
            return reject(
                "anchor_release_rejected_bad_overlap",
                reference_area_score=_round_debug(reference_area_score),
            )

    overlap_decision = _anchor_release_overlap_decision(
        expected_item,
        bbox=bbox,
        global_bbox=global_bbox,
        fallback_bbox=fallback_bbox,
        all_expected=all_expected,
        trusted_anchors=trusted_anchors,
        consensus=consensus,
        inlier_anchors=inlier_anchors,
        single_anchor=single_anchor,
        shift_factor=shift_factor,
    )
    update_anchor_release_debug(
        reject_debug,
        overlap_count=overlap_decision.overlap_count,
        overlap_unresolved_count=overlap_decision.unresolved_count,
        overlap_strong_anchor_count=overlap_decision.strong_anchor_count,
        overlap_max=_round_debug(overlap_decision.max_overlap),
        overlap_resolved_max=_round_debug(overlap_decision.max_resolved_overlap),
    )
    if not overlap_decision.allowed:
        return reject(
            overlap_decision.reason or "anchor_release_rejected_bad_overlap",
        )

    return MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=consensus.candidate_count,
        inlier_count=consensus.inlier_count,
        inlier_ratio=float(consensus.inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
        source_counts=source_counts,
    )


def _select_anchor_release_consensus(
    candidates: list[tuple[TrustedAnchor, np.ndarray, float]],
    *,
    min_anchors: int,
    max_anchor_error: float,
    global_bbox: BBox,
) -> AnchorReleaseConsensus | None:
    residual_array = np.asarray([item[1] for item in candidates], dtype=np.float32)
    weights_array = np.asarray([item[2] for item in candidates], dtype=np.float32)
    if residual_array.ndim != 2 or residual_array.shape[1] != 2:
        return None
    if not np.isfinite(residual_array).all():
        return None
    if not np.isfinite(weights_array).all() or float(np.sum(weights_array)) <= 0.0:
        return None

    min_local_ratio = float(
        _thresholds.missing_anchor_release_local_min_inlier_ratio
    )
    best: AnchorReleaseConsensus | None = None
    best_score = float("-inf")

    for seed in residual_array:
        seed_errors = np.linalg.norm(residual_array - seed[None, :], axis=1)
        seed_mask = seed_errors <= max_anchor_error
        consensus = _build_anchor_release_consensus(
            residual_array,
            weights_array,
            initial_mask=seed_mask,
            min_anchors=min_anchors,
            max_anchor_error=max_anchor_error,
            global_bbox=global_bbox,
        )
        if consensus is None:
            continue
        if consensus.inlier_count > 1 and consensus.inlier_ratio < min_local_ratio:
            continue

        inlier_weight_sum = float(np.sum(weights_array[consensus.inlier_mask]))
        score = (
            consensus.inlier_count * 100.0
            + consensus.inlier_ratio * 10.0
            + inlier_weight_sum
            - consensus.median_error
            - consensus.dispersion * 10.0
        )
        if score > best_score:
            best = consensus
            best_score = score

    return best


def _build_anchor_release_consensus(
    residual_array: np.ndarray,
    weights_array: np.ndarray,
    *,
    initial_mask: np.ndarray,
    min_anchors: int,
    max_anchor_error: float,
    global_bbox: BBox,
) -> AnchorReleaseConsensus | None:
    if initial_mask.dtype != np.bool_:
        initial_mask = initial_mask.astype(bool)
    if int(np.count_nonzero(initial_mask)) < min_anchors:
        return None

    initial_weights = weights_array[initial_mask]
    if float(np.sum(initial_weights)) <= 0.0:
        return None

    initial_shift = np.average(
        residual_array[initial_mask],
        axis=0,
        weights=initial_weights,
    ).astype(np.float32)
    if not np.isfinite(initial_shift).all():
        return None

    errors = np.linalg.norm(residual_array - initial_shift[None, :], axis=1)
    if len(errors) == 0 or not np.isfinite(errors).all():
        return None

    inlier_mask = errors <= max_anchor_error
    inlier_count = int(np.count_nonzero(inlier_mask))
    candidate_count = int(len(errors))
    if inlier_count < min_anchors:
        return None

    inlier_weights = weights_array[inlier_mask]
    if float(np.sum(inlier_weights)) <= 0.0:
        return None

    shift = np.average(
        residual_array[inlier_mask],
        axis=0,
        weights=inlier_weights,
    ).astype(np.float32)
    if not np.isfinite(shift).all():
        return None

    inlier_errors = np.linalg.norm(residual_array[inlier_mask] - shift[None, :], axis=1)
    if len(inlier_errors) == 0 or not np.isfinite(inlier_errors).all():
        return None

    median_error = float(np.median(inlier_errors))
    dispersion = float(
        np.sqrt(np.average(np.square(inlier_errors), weights=inlier_weights))
        / max(1.0, bbox_diag(global_bbox))
    )
    if dispersion > _thresholds.missing_anchor_release_max_dispersion_factor:
        return None

    return AnchorReleaseConsensus(
        shift=shift,
        inlier_mask=inlier_mask,
        candidate_count=candidate_count,
        inlier_count=inlier_count,
        inlier_ratio=inlier_count / max(1, candidate_count),
        median_error=median_error,
        dispersion=dispersion,
    )



def _missing_debug_projection_name(debug: dict[str, Any] | None) -> str:
    if not debug:
        return ""
    value = debug.get("missing_polygon_projection") or debug.get("projection")
    return str(value or "")


def _build_trusted_missing_anchors(
    all_expected: list[ProjectedExpected],
    *,
    slot_by_index: dict[int, ExpectedSlot],
    projection_data: LocalProjectionData | None,
) -> list[TrustedAnchor]:
    if projection_data is None or projection_data.global_homography is None:
        return []
    if len(all_expected) <= 1:
        return []

    anchors: list[TrustedAnchor] = []

    def add_anchor(anchor: TrustedAnchor | None) -> bool:
        if anchor is None:
            return False
        anchors.append(anchor)
        return True

    for expected_item in all_expected:
        slot = slot_by_index.get(expected_item.index)
        if slot is None:
            continue

        global_projection = _missing_global_projection(
            expected_item,
            projection_data=projection_data,
        )
        if global_projection is None:
            continue
        global_polygon, global_bbox = global_projection

        rescue = _try_missing_translation_rescue(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            global_polygon=global_polygon,
            global_bbox=global_bbox,
            all_expected=all_expected,
        )
        if rescue is not None and _trusted_anchor_translation_reject_reason(rescue) is None:
            if add_anchor(
                _trusted_anchor_from_translation_rescue(
                    expected_item,
                    rescue=rescue,
                    global_bbox=global_bbox,
                )
            ):
                continue

        refinement = _try_missing_polygon_refinement(
            expected_item,
            slot=slot,
            projection_data=projection_data,
            all_expected=all_expected,
        )
        if (
            refinement is not None
            and _trusted_anchor_affine_reject_reason(refinement, global_bbox) is None
        ):
            if add_anchor(
                _trusted_anchor_from_affine_refinement(
                    expected_item,
                    refinement=refinement,
                    global_bbox=global_bbox,
                )
            ):
                continue

        if _trusted_anchor_local_global_reject_reason(
            expected_item,
            slot=slot,
            global_bbox=global_bbox,
        ) is None:
            add_anchor(
                _trusted_anchor_from_local_global_slot(
                    expected_item,
                    slot=slot,
                    global_bbox=global_bbox,
                )
            )
            continue

        if _trusted_anchor_weak_local_global_reject_reason(
            expected_item,
            slot=slot,
            global_bbox=global_bbox,
        ) is None:
            add_anchor(
                _trusted_anchor_from_weak_local_global_slot(
                    expected_item,
                    slot=slot,
                    global_bbox=global_bbox,
                )
            )

    return anchors


def _trusted_anchor_translation_reject_reason(
    rescue: MissingTranslationRescue,
) -> str | None:
    max_anchor_error = float(_thresholds.missing_anchor_release_max_anchor_error)
    min_anchor_ratio = max(
        _thresholds.missing_rescue_translation_min_inlier_ratio,
        _thresholds.missing_anchor_release_min_inlier_ratio,
    )

    if rescue.inlier_count < _thresholds.missing_rescue_translation_min_support:
        return "low_support"
    if rescue.inlier_ratio < min_anchor_ratio:
        return "low_inlier_ratio"
    if rescue.median_error > max_anchor_error:
        return "high_median_error"
    if rescue.shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return "large_shift"

    residual = np.asarray([rescue.shift_x, rescue.shift_y], dtype=np.float32)
    if not np.isfinite(residual).all():
        return "bad_residual"
    return None


def _trusted_anchor_affine_reject_reason(
    refinement: MissingPolygonRefinement,
    global_bbox: BBox,
) -> str | None:
    max_anchor_error = float(_thresholds.missing_anchor_release_max_anchor_error)
    min_support = max(6, _thresholds.missing_polygon_min_feature_support)
    min_ratio = max(0.55, _thresholds.missing_anchor_release_min_inlier_ratio)

    if refinement.inlier_count < min_support:
        return "low_support"
    if refinement.inlier_ratio < min_ratio:
        return "low_inlier_ratio"
    if refinement.median_error > max_anchor_error:
        return "high_median_error"
    if refinement.area_score < 0.46:
        return "low_area_score"
    if refinement.center_drift_factor > 0.70:
        return "large_center_drift"

    residual = np.asarray(
        [
            bbox_center(refinement.bbox)[0] - bbox_center(global_bbox)[0],
            bbox_center(refinement.bbox)[1] - bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(residual).all():
        return "bad_residual"

    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return "large_shift"

    local_global_area_score = bbox_area_similarity(refinement.bbox, global_bbox)
    if local_global_area_score < 0.30:
        return "bad_local_global_area"
    return None


def _trusted_anchor_local_global_reject_reason(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
) -> str | None:
    min_support = max(4, _thresholds.missing_polygon_min_feature_support)
    if slot.feature_support < min_support:
        return "low_support"

    feature_ratio = slot.feature_support / max(1, slot.feature_total)
    if feature_ratio < 0.10:
        return "low_feature_ratio"

    local_global_area_score = bbox_area_similarity(expected_item.bbox, global_bbox)
    local_global_center_factor = bbox_center_distance_factor(
        expected_item.bbox,
        global_bbox,
    )
    if local_global_area_score < 0.55:
        return "bad_area"
    if local_global_center_factor > 0.65:
        return "large_center_drift"

    residual = np.asarray(
        [
            bbox_center(expected_item.bbox)[0] - bbox_center(global_bbox)[0],
            bbox_center(expected_item.bbox)[1] - bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(residual).all():
        return "bad_residual"

    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return "large_shift"
    return None




def _trusted_anchor_weak_local_global_reject_reason(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
) -> str | None:
    min_support = max(
        1,
        int(_thresholds.missing_anchor_release_weak_slot_min_feature_support),
    )
    if slot.feature_support < min_support:
        return "low_support"

    feature_ratio = slot.feature_support / max(1, slot.feature_total)
    if feature_ratio < _thresholds.missing_anchor_release_weak_slot_min_feature_ratio:
        return "low_feature_ratio"

    local_global_area_score = bbox_area_similarity(expected_item.bbox, global_bbox)
    local_global_center_factor = bbox_center_distance_factor(
        expected_item.bbox,
        global_bbox,
    )
    if (
        local_global_area_score
        < _thresholds.missing_anchor_release_weak_slot_min_area_score
    ):
        return "bad_area"
    if (
        local_global_center_factor
        > _thresholds.missing_anchor_release_weak_slot_max_center_factor
    ):
        return "large_center_drift"

    residual = np.asarray(
        [
            bbox_center(expected_item.bbox)[0] - bbox_center(global_bbox)[0],
            bbox_center(expected_item.bbox)[1] - bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    if not np.isfinite(residual).all():
        return "bad_residual"

    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    if shift_factor > _thresholds.missing_anchor_release_weak_slot_max_shift_factor:
        return "large_shift"
    return None


def _trusted_anchor_from_translation_rescue(
    expected_item: ProjectedExpected,
    *,
    rescue: MissingTranslationRescue,
    global_bbox: BBox,
) -> TrustedAnchor | None:
    if _trusted_anchor_translation_reject_reason(rescue) is not None:
        return None

    residual = np.asarray([rescue.shift_x, rescue.shift_y], dtype=np.float32)

    support_weight = float(rescue.inlier_count) / max(1.0, float(rescue.candidate_count))
    error_weight = 1.0 / max(1.0, float(rescue.median_error))

    return TrustedAnchor(
        expected_index=expected_item.index,
        source="translation_rescue",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=rescue.bbox,
        weight=max(0.01, support_weight * error_weight),
        candidate_count=rescue.candidate_count,
        inlier_count=rescue.inlier_count,
        inlier_ratio=rescue.inlier_ratio,
        median_error=rescue.median_error,
        shift_factor=rescue.shift_factor,
    )


def _trusted_anchor_from_affine_refinement(
    expected_item: ProjectedExpected,
    *,
    refinement: MissingPolygonRefinement,
    global_bbox: BBox,
) -> TrustedAnchor | None:
    if _trusted_anchor_affine_reject_reason(refinement, global_bbox) is not None:
        return None

    residual = np.asarray(
        [
            bbox_center(refinement.bbox)[0] - bbox_center(global_bbox)[0],
            bbox_center(refinement.bbox)[1] - bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    local_global_area_score = bbox_area_similarity(refinement.bbox, global_bbox)

    support_weight = refinement.inlier_count / max(1.0, float(refinement.candidate_count))
    geometry_weight = refinement.area_score * max(0.20, local_global_area_score)
    error_weight = 1.0 / max(1.0, float(refinement.median_error))

    return TrustedAnchor(
        expected_index=expected_item.index,
        source="context_feature_affine",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=refinement.bbox,
        weight=max(0.01, support_weight * geometry_weight * error_weight),
        candidate_count=refinement.candidate_count,
        inlier_count=refinement.inlier_count,
        inlier_ratio=refinement.inlier_ratio,
        median_error=refinement.median_error,
        shift_factor=shift_factor,
    )


def _trusted_anchor_from_local_global_slot(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
) -> TrustedAnchor | None:
    if _trusted_anchor_local_global_reject_reason(
        expected_item,
        slot=slot,
        global_bbox=global_bbox,
    ) is not None:
        return None

    feature_ratio = slot.feature_support / max(1, slot.feature_total)
    local_global_area_score = bbox_area_similarity(expected_item.bbox, global_bbox)
    local_global_center_factor = bbox_center_distance_factor(
        expected_item.bbox,
        global_bbox,
    )
    residual = np.asarray(
        [
            bbox_center(expected_item.bbox)[0] - bbox_center(global_bbox)[0],
            bbox_center(expected_item.bbox)[1] - bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))

    error_weight = 1.0 / max(
        1.0,
        local_global_center_factor * bbox_diag(global_bbox),
    )

    return TrustedAnchor(
        expected_index=expected_item.index,
        source="local_global_slot_hint",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=expected_item.bbox,
        weight=max(0.01, feature_ratio * local_global_area_score * error_weight),
        candidate_count=max(1, slot.feature_total),
        inlier_count=slot.feature_support,
        inlier_ratio=feature_ratio,
        median_error=float(local_global_center_factor * bbox_diag(global_bbox)),
        shift_factor=shift_factor,
    )




def _trusted_anchor_from_weak_local_global_slot(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
) -> TrustedAnchor | None:
    if _trusted_anchor_weak_local_global_reject_reason(
        expected_item,
        slot=slot,
        global_bbox=global_bbox,
    ) is not None:
        return None

    feature_ratio = slot.feature_support / max(1, slot.feature_total)
    local_global_area_score = bbox_area_similarity(expected_item.bbox, global_bbox)
    local_global_center_factor = bbox_center_distance_factor(
        expected_item.bbox,
        global_bbox,
    )
    residual = np.asarray(
        [
            bbox_center(expected_item.bbox)[0] - bbox_center(global_bbox)[0],
            bbox_center(expected_item.bbox)[1] - bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    median_error = float(local_global_center_factor * bbox_diag(global_bbox))

    support_weight = min(1.0, slot.feature_support / max(1.0, float(slot.feature_total)))
    geometry_weight = max(0.02, local_global_area_score)
    error_weight = 1.0 / max(1.0, median_error)

    return TrustedAnchor(
        expected_index=expected_item.index,
        source="weak_local_global_slot_hint",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=expected_item.bbox,
        weight=max(0.005, support_weight * geometry_weight * error_weight),
        candidate_count=max(1, slot.feature_total),
        inlier_count=slot.feature_support,
        inlier_ratio=feature_ratio,
        median_error=median_error,
        shift_factor=shift_factor,
    )




def _trusted_anchor_from_matched_detection(
    expected_item: ProjectedExpected,
    *,
    detection_item: DetectionCandidate,
    match_iou: float,
    projection_data: LocalProjectionData | None,
    match_debug: dict[str, Any] | None = None,
) -> TrustedAnchor | None:
    if projection_data is None:
        return None

    runtime_trusted = bool(
        match_debug is not None
        and match_debug.get("runtime_yolo_anchor_trusted") is True
    )
    if not runtime_trusted and match_iou < max(0.55, IOU_MATCH_THRESHOLD):
        return None

    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return None
    _, global_bbox = global_projection

    local_bbox = detection_item.bbox
    residual = np.asarray(
        [
            bbox_center(local_bbox)[0] - bbox_center(global_bbox)[0],
            bbox_center(local_bbox)[1] - bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    if residual.shape != (2,) or not np.isfinite(residual).all():
        return None

    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return None

    area_score = bbox_area_similarity(local_bbox, global_bbox)
    if area_score < 0.24:
        return None

    confidence = max(0.0, min(1.0, float(detection_item.detection.confidence or 0.0)))
    if runtime_trusted and match_debug is not None:
        anchor_iou = _debug_float(match_debug.get("runtime_anchor_slot_iou"), default=match_iou)
        coverage = _debug_float(match_debug.get("runtime_anchor_slot_coverage"), default=match_iou)
        containment = _debug_float(
            match_debug.get("runtime_anchor_detection_containment"),
            default=match_iou,
        )
        quality = max(0.01, min(1.0, (anchor_iou + coverage + containment) / 3.0))
    else:
        quality = max(0.01, min(1.0, float(match_iou)))
    return TrustedAnchor(
        expected_index=expected_item.index,
        source="matched_detection",
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=local_bbox,
        weight=max(0.01, quality * max(0.10, confidence) * max(0.10, area_score)),
        candidate_count=1,
        inlier_count=1,
        inlier_ratio=quality,
        median_error=max(0.0, (1.0 - quality) * bbox_diag(global_bbox)),
        shift_factor=shift_factor,
    )


def _trusted_anchor_from_resolved_projection(
    expected_item: ProjectedExpected,
    *,
    polygon: list[list[float]] | None,
    bbox: BBox | None,
    projection_data: LocalProjectionData | None,
    debug: dict[str, Any] | None,
) -> TrustedAnchor | None:
    if polygon is None or bbox is None or debug is None or projection_data is None:
        return None

    projection_name = str(
        debug.get("missing_polygon_projection")
        or debug.get("projection")
        or ""
    )
    source_by_projection = {
        "context_feature_affine_translation_rescue": "resolved_context_translation",
        "context_feature_affine_scene_translation_rescue": "resolved_context_translation",
        "expected_slot_context_translation_rescue": "resolved_context_translation",
        "expected_slot_global_translation_rescue": "resolved_context_translation",
        "expected_slot_scene_translation_rescue": "resolved_scene_translation",
        "expected_slot_anchor_release": "resolved_anchor_release",
        "context_feature_affine": "resolved_context_affine",
    }
    source = source_by_projection.get(projection_name)
    if source is None:
        return None

    global_projection = _missing_global_projection(
        expected_item,
        projection_data=projection_data,
    )
    if global_projection is None:
        return None
    _, global_bbox = global_projection

    residual = np.asarray(
        [
            bbox_center(bbox)[0] - bbox_center(global_bbox)[0],
            bbox_center(bbox)[1] - bbox_center(global_bbox)[1],
        ],
        dtype=np.float32,
    )
    if residual.shape != (2,) or not np.isfinite(residual).all():
        return None

    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    if shift_factor > _thresholds.missing_anchor_release_max_shift_factor:
        return None

    area_score = bbox_area_similarity(bbox, global_bbox)
    if area_score < 0.24:
        return None

    candidate_count = _debug_int(debug.get("missing_polygon_candidate_count"), default=1)
    inlier_count = _debug_int(debug.get("missing_polygon_inliers"), default=1)
    inlier_ratio = _debug_float(debug.get("missing_polygon_inlier_ratio"), default=1.0)
    median_error = _debug_float(debug.get("missing_polygon_median_error"), default=0.0)

    if source == "resolved_context_affine":
        if inlier_count < max(6, _thresholds.missing_polygon_min_feature_support):
            return None
        if inlier_ratio < 0.55:
            return None
        if median_error > _thresholds.missing_anchor_release_max_anchor_error:
            return None
        if _debug_float(debug.get("missing_polygon_area_score"), default=1.0) < 0.46:
            return None
    elif source == "resolved_anchor_release":
        if inlier_count < 2:
            return None
        if inlier_ratio < 0.70:
            return None
    else:
        if inlier_count < 3:
            return None
        if inlier_ratio < 0.48:
            return None
        if median_error > _thresholds.missing_anchor_release_max_anchor_error:
            return None

    support_weight = inlier_count / max(1.0, float(candidate_count))
    error_weight = 1.0 / max(1.0, float(median_error))
    return TrustedAnchor(
        expected_index=expected_item.index,
        source=source,
        residual=residual,
        global_bbox=global_bbox,
        local_bbox=bbox,
        weight=max(0.01, support_weight * max(0.10, area_score) * error_weight),
        candidate_count=max(1, candidate_count),
        inlier_count=max(1, inlier_count),
        inlier_ratio=max(0.0, min(1.0, float(inlier_ratio))),
        median_error=max(0.0, float(median_error)),
        shift_factor=shift_factor,
    )


def _append_or_replace_trusted_anchor(
    anchors: list[TrustedAnchor],
    anchor: TrustedAnchor | None,
) -> None:
    if anchor is None:
        return
    for index, existing in enumerate(list(anchors)):
        if existing.expected_index != anchor.expected_index:
            continue
        if _trusted_anchor_source_weight(existing.source) >= _trusted_anchor_source_weight(anchor.source):
            return
        anchors[index] = anchor
        return
    anchors.append(anchor)


def _debug_int(value: Any, *, default: int = 0) -> int:
    try:
        if value is None:
            return default
        return int(value)
    except (TypeError, ValueError):
        return default


def _debug_float(value: Any, *, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        result = float(value)
    except (TypeError, ValueError):
        return default
    return result if np.isfinite(result) else default


def _trusted_anchor_source_weight(source: str) -> float:
    if source == "matched_detection":
        return 1.15
    if source == "translation_rescue":
        return 1.0
    if source == "resolved_context_translation":
        return 0.92
    if source == "context_feature_affine":
        return 0.78
    if source == "resolved_anchor_release":
        return 0.72
    if source == "resolved_context_affine":
        return 0.66
    if source == "resolved_scene_translation":
        return 0.58
    if source == "local_global_slot_hint":
        return 0.42
    if source == "weak_local_global_slot_hint":
        return 0.18
    return 0.42


def _anchor_release_allows_single_anchor(
    candidates: list[tuple[TrustedAnchor, np.ndarray, float]],
    *,
    slot: ExpectedSlot,
    global_bbox: BBox,
    fallback_bbox: BBox | None,
) -> bool:
    if len(candidates) != 1:
        return False

    anchor, residual, _ = candidates[0]
    if anchor.source in {"local_global_slot_hint", "weak_local_global_slot_hint"}:
        return False
    if anchor.inlier_count < _thresholds.missing_anchor_release_single_min_inliers:
        return False
    if anchor.inlier_ratio < _thresholds.missing_anchor_release_single_min_inlier_ratio:
        return False
    if anchor.median_error > _thresholds.missing_anchor_release_single_max_anchor_error:
        return False

    shift_factor = float(np.linalg.norm(residual) / max(1.0, bbox_diag(global_bbox)))
    if shift_factor > _thresholds.missing_anchor_release_single_max_shift_factor:
        return False

    polygon_bbox = translate_bbox(global_bbox, dx=float(residual[0]), dy=float(residual[1]))
    local_bbox = fallback_bbox or slot.projected_bbox
    local_area_score = bbox_area_similarity(polygon_bbox, local_bbox)
    local_center_factor = bbox_center_distance_factor(polygon_bbox, local_bbox)

    if local_area_score < _thresholds.missing_anchor_release_single_min_local_area_score:
        return False
    if local_center_factor > _thresholds.missing_anchor_release_single_max_local_center_factor:
        return False

    return True




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
        if (
            global_area_score
            < _thresholds.missing_fallback_min_global_area_score
        ):
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



def _missing_polygon_refinement_points(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot,
    projection_data: LocalProjectionData,
    all_expected: list[ProjectedExpected] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    reference_points = as_match_points(projection_data.reference_points)
    frame_points = as_match_points(projection_data.frame_points)
    if reference_points is None or frame_points is None:
        return empty_match_points(), empty_match_points()
    if len(reference_points) != len(frame_points):
        return empty_match_points(), empty_match_points()

    reference_bbox = slot.reference_bbox
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return empty_match_points(), empty_match_points()

    context_expansion = _thresholds.missing_polygon_context_expansion
    reference_window = expand_bbox(reference_bbox, factor=context_expansion)
    frame_window = expand_bbox(
        slot.projected_bbox,
        factor=context_expansion,
        frame_size=projection_data.frame_size,
    )

    reference_polygon = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )
    reference_margin = _missing_context_exclusion_margin(reference_bbox)
    reference_exclusion_zones, frame_exclusion_zones = _multi_object_context_exclusion_zones(
        expected_item,
        all_expected=all_expected,
        projection_data=projection_data,
    )

    selected_reference: list[np.ndarray] = []
    selected_frame: list[np.ndarray] = []

    for reference_point, frame_point in zip(reference_points, frame_points, strict=True):
        if not point_in_bbox(reference_point, reference_window):
            continue
        if not point_outside_np_polygon_margin(
            reference_point,
            reference_polygon,
            margin=reference_margin,
        ):
            continue
        if point_in_any_bbox(reference_point, reference_exclusion_zones):
            continue
        if not point_in_bbox(frame_point, frame_window):
            continue
        if point_in_any_bbox(frame_point, frame_exclusion_zones):
            continue

        selected_reference.append(reference_point)
        selected_frame.append(frame_point)

    if not selected_reference:
        return empty_match_points(), empty_match_points()

    local_reference = np.asarray(selected_reference, dtype=np.float32)
    local_frame = np.asarray(selected_frame, dtype=np.float32)
    if len(local_reference) <= _MISSING_POLYGON_MAX_LOCAL_POINTS:
        return local_reference, local_frame

    reference_center = np.asarray(bbox_center(reference_bbox), dtype=np.float32)
    frame_center = np.asarray(bbox_center(slot.projected_bbox), dtype=np.float32)
    reference_distance = np.linalg.norm(local_reference - reference_center, axis=1)
    frame_distance = np.linalg.norm(local_frame - frame_center, axis=1)
    indices = np.argsort(reference_distance + frame_distance)[
        :_MISSING_POLYGON_MAX_LOCAL_POINTS
    ]
    return local_reference[indices], local_frame[indices]




def _missing_affine_scale_in_range(affine: np.ndarray) -> bool:
    transform = np.asarray(affine, dtype=np.float32)
    if transform.shape != (2, 3):
        return False

    scale_x = float(np.linalg.norm(transform[:, 0]))
    scale_y = float(np.linalg.norm(transform[:, 1]))
    return (
        _MISSING_POLYGON_MIN_AFFINE_SCALE
        <= scale_x
        <= _MISSING_POLYGON_MAX_AFFINE_SCALE
        and _MISSING_POLYGON_MIN_AFFINE_SCALE
        <= scale_y
        <= _MISSING_POLYGON_MAX_AFFINE_SCALE
    )

def _missing_context_spread_score(points: np.ndarray, bbox: BBox) -> float:
    if len(points) < 3:
        return 0.0
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array).all():
        return 0.0

    x1, y1, x2, y2 = bbox
    bbox_w = max(1.0, float(x2 - x1))
    bbox_h = max(1.0, float(y2 - y1))
    point_w = float(np.max(array[:, 0]) - np.min(array[:, 0]))
    point_h = float(np.max(array[:, 1]) - np.min(array[:, 1]))
    x_spread = max(0.0, min(1.0, point_w / bbox_w))
    y_spread = max(0.0, min(1.0, point_h / bbox_h))

    return float(np.sqrt(x_spread * y_spread))


def _missing_context_quadrant_count(points: np.ndarray, bbox: BBox) -> int:
    if len(points) == 0:
        return 0
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or not np.isfinite(array).all():
        return 0

    cx, cy = bbox_center(bbox)
    quadrants: set[tuple[int, int]] = set()
    for x, y in array[:, :2]:
        quadrants.add((1 if float(x) >= cx else 0, 1 if float(y) >= cy else 0))
    return len(quadrants)

def _slot_feature_support(
    expected_item: ProjectedExpected,
    *,
    search_bbox: BBox,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None = None,
) -> tuple[int, int]:
    if projection_data is None:
        return 0, 0

    reference_points = as_match_points(projection_data.reference_points)
    frame_points = as_match_points(projection_data.frame_points)
    if reference_points is None or frame_points is None:
        return 0, 0
    if len(reference_points) != len(frame_points):
        return 0, 0

    reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return 0, 0

    reference_window = expand_bbox(
        reference_bbox,
        factor=_thresholds.slot_feature_search_expansion,
    )
    reference_polygon = np.asarray(
        expected_item.item.reference_polygon,
        dtype=np.float32,
    )
    reference_margin = _missing_context_exclusion_margin(reference_bbox)
    reference_exclusion_zones, frame_exclusion_zones = _multi_object_context_exclusion_zones(
        expected_item,
        all_expected=all_expected,
        projection_data=projection_data,
    )

    total = 0
    support = 0
    for reference_point, frame_point in zip(reference_points, frame_points, strict=True):
        if not point_in_bbox(reference_point, reference_window):
            continue
        if not point_outside_np_polygon_margin(
            reference_point,
            reference_polygon,
            margin=reference_margin,
        ):
            continue
        if point_in_any_bbox(reference_point, reference_exclusion_zones):
            continue
        if point_in_any_bbox(frame_point, frame_exclusion_zones):
            continue
        total += 1
        if point_in_bbox(frame_point, search_bbox):
            support += 1

    return support, total


def _slot_debug_payload(slot: ExpectedSlot | None) -> dict[str, Any] | None:
    if slot is None:
        return None
    return {
        "projected_bbox": _bbox_debug(slot.projected_bbox),
        "search_bbox": _bbox_debug(slot.search_bbox),
        "reference_bbox": _bbox_debug(slot.reference_bbox),
        "feature_support": slot.feature_support,
        "feature_total": slot.feature_total,
        "search_expansion": _thresholds.slot_search_expansion,
        "feature_search_expansion": _thresholds.slot_feature_search_expansion,
        "feature_source": "context_ring",
    }


def _bbox_debug(bbox: BBox | None) -> dict[str, float] | None:
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return {
        "x": _round_debug(x1),
        "y": _round_debug(y1),
        "w": _round_debug(max(0.0, x2 - x1)),
        "h": _round_debug(max(0.0, y2 - y1)),
    }

def _all_expected_context_exclusion_zones(
    expected_item: ProjectedExpected,
    *,
    all_expected: list[ProjectedExpected] | None,
    projection_data: LocalProjectionData | None,
) -> tuple[list[BBox], list[BBox]]:
    expected_items = all_expected if all_expected else [expected_item]
    expansion = float(
        _thresholds.missing_polygon_multi_context_exclusion_expansion
    )
    reference_zones: list[BBox] = []
    frame_zones: list[BBox] = []
    frame_size = projection_data.frame_size if projection_data is not None else None

    for item in expected_items:
        reference_bbox = bbox_from_polygon(item.item.reference_polygon)
        if reference_bbox is not None:
            reference_zones.append(expand_bbox(reference_bbox, factor=expansion))

        frame_zones.append(
            expand_bbox(
                item.bbox,
                factor=expansion,
                frame_size=frame_size,
            )
        )

    return reference_zones, frame_zones


def _anchor_release_overlap_decision(
    expected_item: ProjectedExpected,
    *,
    bbox: BBox,
    global_bbox: BBox,
    fallback_bbox: BBox | None,
    all_expected: list[ProjectedExpected] | None,
    trusted_anchors: list[TrustedAnchor] | None,
    consensus: AnchorReleaseConsensus,
    inlier_anchors: list[TrustedAnchor],
    single_anchor: bool,
    shift_factor: float,
) -> AnchorOverlapDecision:
    """Decide whether overlap with neighboring expected slots is still unsafe.

    The old guard compared the released bbox with every projected expected slot.
    In multi-object scenes those slots can be stale: a neighboring object can have
    already produced a trusted local correction, while its original expected bbox
    still intersects the current released slot.  This arbitration keeps the hard
    safety check against trusted resolved neighbors, but allows overlaps with
    stale/unresolved slots only when the anchor consensus is very compact and the
    released center is still owned by the current slot.
    """

    if not all_expected or len(all_expected) <= 1:
        return AnchorOverlapDecision(
            allowed=True,
            reason=None,
            overlap_count=0,
            unresolved_count=0,
            strong_anchor_count=0,
            max_overlap=0.0,
            max_resolved_overlap=0.0,
        )

    overlap_limit = float(_thresholds.missing_anchor_release_max_other_overlap)
    strong_anchor_by_index = _best_strong_anchor_by_expected_index(trusted_anchors or [])
    target_owner_bbox = fallback_bbox or global_bbox
    target_center = np.asarray(bbox_center(target_owner_bbox), dtype=np.float32)
    released_center = np.asarray(bbox_center(bbox), dtype=np.float32)
    current_diag = max(1.0, bbox_diag(global_bbox))

    overlap_count = 0
    unresolved_count = 0
    strong_anchor_count = 0
    max_overlap = 0.0
    max_resolved_overlap = 0.0

    for other in all_expected:
        if other.index == expected_item.index:
            continue

        overlap = max(
            bbox_iou(bbox, other.bbox),
            bbox_containment(bbox, other.bbox),
            bbox_containment(other.bbox, bbox),
        )
        max_overlap = max(max_overlap, float(overlap))
        if overlap <= overlap_limit:
            continue

        overlap_count += 1
        strong_anchor = strong_anchor_by_index.get(other.index)
        if strong_anchor is not None:
            strong_anchor_count += 1
            resolved_overlap = max(
                bbox_iou(bbox, strong_anchor.local_bbox),
                bbox_containment(bbox, strong_anchor.local_bbox),
                bbox_containment(strong_anchor.local_bbox, bbox),
            )
            max_resolved_overlap = max(max_resolved_overlap, float(resolved_overlap))
            if resolved_overlap > overlap_limit:
                return AnchorOverlapDecision(
                    allowed=False,
                    reason="anchor_release_rejected_overlap_with_strong_anchor",
                    overlap_count=overlap_count,
                    unresolved_count=unresolved_count,
                    strong_anchor_count=strong_anchor_count,
                    max_overlap=max_overlap,
                    max_resolved_overlap=max_resolved_overlap,
                )
            continue

        unresolved_count += 1
        other_center = np.asarray(bbox_center(other.bbox), dtype=np.float32)
        target_distance = float(np.linalg.norm(released_center - target_center))
        other_distance = float(np.linalg.norm(released_center - other_center))
        ownership_margin = float(
            _thresholds.missing_anchor_release_overlap_center_margin
        )
        if other_distance * ownership_margin < target_distance:
            return AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_owned_by_other_slot",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )

    if overlap_count == 0:
        return AnchorOverlapDecision(
            allowed=True,
            reason=None,
            overlap_count=0,
            unresolved_count=0,
            strong_anchor_count=0,
            max_overlap=max_overlap,
            max_resolved_overlap=max_resolved_overlap,
        )

    if unresolved_count > 0:
        if single_anchor:
            target_drift = float(np.linalg.norm(released_center - target_center)) / current_diag
            if _anchor_release_allows_strict_single_overlap(
                inlier_anchors,
                consensus=consensus,
                shift_factor=shift_factor,
                unresolved_count=unresolved_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
                target_drift=target_drift,
            ):
                return AnchorOverlapDecision(
                    allowed=True,
                    reason=None,
                    overlap_count=overlap_count,
                    unresolved_count=unresolved_count,
                    strong_anchor_count=strong_anchor_count,
                    max_overlap=max_overlap,
                    max_resolved_overlap=max_resolved_overlap,
                )
            return AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_single_anchor_guard",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )
        if (
            consensus.inlier_count
            < _thresholds.missing_anchor_release_overlap_min_inliers
        ):
            return AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_weak_consensus",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )
        if (
            consensus.inlier_ratio
            < _thresholds.missing_anchor_release_overlap_min_inlier_ratio
        ):
            return AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_weak_consensus",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )
        if (
            consensus.dispersion
            > _thresholds.missing_anchor_release_overlap_max_dispersion_factor
        ):
            return AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_high_dispersion",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )
        if (
            shift_factor
            > _thresholds.missing_anchor_release_overlap_max_shift_factor
        ):
            return AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_large_shift",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )

        target_drift = float(np.linalg.norm(released_center - target_center)) / current_diag
        max_target_drift = (
            _thresholds.missing_anchor_release_max_local_center_factor
        )
        if target_drift > max_target_drift:
            return AnchorOverlapDecision(
                allowed=False,
                reason="anchor_release_rejected_overlap_target_drift",
                overlap_count=overlap_count,
                unresolved_count=unresolved_count,
                strong_anchor_count=strong_anchor_count,
                max_overlap=max_overlap,
                max_resolved_overlap=max_resolved_overlap,
            )

    return AnchorOverlapDecision(
        allowed=True,
        reason=None,
        overlap_count=overlap_count,
        unresolved_count=unresolved_count,
        strong_anchor_count=strong_anchor_count,
        max_overlap=max_overlap,
        max_resolved_overlap=max_resolved_overlap,
    )


def _anchor_release_allows_strict_single_overlap(
    inlier_anchors: list[TrustedAnchor],
    *,
    consensus: AnchorReleaseConsensus,
    shift_factor: float,
    unresolved_count: int,
    max_overlap: float,
    max_resolved_overlap: float,
    target_drift: float,
) -> bool:
    if len(inlier_anchors) != 1:
        return False
    if unresolved_count > _thresholds.missing_anchor_release_strict_single_overlap_max_unresolved:
        return False
    if max_resolved_overlap > _thresholds.missing_anchor_release_max_other_overlap:
        return False
    if not np.isfinite(max_overlap):
        return False
    if target_drift > _thresholds.missing_anchor_release_single_max_local_center_factor:
        return False

    anchor = inlier_anchors[0]
    if anchor.source not in {"translation_rescue", "context_feature_affine"}:
        return False
    if anchor.inlier_count < _thresholds.missing_anchor_release_strict_single_overlap_min_inliers:
        return False
    if anchor.inlier_ratio < _thresholds.missing_anchor_release_strict_single_overlap_min_inlier_ratio:
        return False
    if anchor.median_error > _thresholds.missing_anchor_release_strict_single_overlap_max_anchor_error:
        return False
    if consensus.dispersion > _thresholds.missing_anchor_release_strict_single_overlap_max_dispersion_factor:
        return False
    if shift_factor > _thresholds.missing_anchor_release_strict_single_overlap_max_shift_factor:
        return False

    return True


def _best_strong_anchor_by_expected_index(
    trusted_anchors: list[TrustedAnchor],
) -> dict[int, TrustedAnchor]:
    result: dict[int, TrustedAnchor] = {}

    for anchor in trusted_anchors:
        if anchor.source not in {"translation_rescue", "context_feature_affine"}:
            continue
        if anchor.inlier_count < 2:
            continue
        min_ratio = _thresholds.missing_anchor_release_min_inlier_ratio
        if anchor.inlier_ratio < min_ratio:
            continue
        current = result.get(anchor.expected_index)
        if current is None or anchor.weight > current.weight:
            result[anchor.expected_index] = anchor

    return result


def _missing_rescue_overlaps_other_expected(
    expected_item: ProjectedExpected,
    *,
    bbox: BBox,
    all_expected: list[ProjectedExpected] | None,
    max_overlap: float | None = None,
) -> bool:
    if not all_expected or len(all_expected) <= 1:
        return False

    overlap_limit = (
        _thresholds.missing_scene_rescue_max_other_overlap
        if max_overlap is None
        else max_overlap
    )
    for other in all_expected:
        if other.index == expected_item.index:
            continue
        overlap = max(
            bbox_iou(bbox, other.bbox),
            bbox_containment(bbox, other.bbox),
            bbox_containment(other.bbox, bbox),
        )
        if overlap > overlap_limit:
            return True
    return False


def _multi_object_context_exclusion_zones(
    expected_item: ProjectedExpected,
    *,
    all_expected: list[ProjectedExpected] | None,
    projection_data: LocalProjectionData | None,
) -> tuple[list[BBox], list[BBox]]:
    if not all_expected or len(all_expected) <= 1:
        return [], []

    expansion = float(
        _thresholds.missing_polygon_multi_context_exclusion_expansion
    )
    reference_zones: list[BBox] = []
    frame_zones: list[BBox] = []
    frame_size = projection_data.frame_size if projection_data is not None else None

    for other in all_expected:
        if other.index == expected_item.index:
            continue

        reference_bbox = bbox_from_polygon(other.item.reference_polygon)
        if reference_bbox is not None:
            reference_zones.append(expand_bbox(reference_bbox, factor=expansion))

        frame_zones.append(
            expand_bbox(
                other.bbox,
                factor=expansion,
                frame_size=frame_size,
            )
        )

    return reference_zones, frame_zones

def _missing_context_exclusion_margin(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    min_side = max(1.0, min(float(x2 - x1), float(y2 - y1)))
    configured = float(_thresholds.missing_polygon_context_exclusion_margin)
    return max(0.0, min(configured, min_side * 0.25))


def _display_detected_polygon(
    detection_item: DetectionCandidate,
) -> list[list[float]] | None:
    return detection_item.polygon

def _match_score_details(
    expected_polygon: list[list[float]],
    detection_polygon: list[list[float]] | None,
    expected_bbox: BBox,
    detection_bbox: BBox,
) -> dict[str, Any]:
    bbox_iou_value = bbox_iou(expected_bbox, detection_bbox)

    if detection_polygon is None or len(detection_polygon) < 3:
        polygon_iou_value = 0.0
    else:
        polygon_iou_value = polygon_iou(expected_polygon, detection_polygon)

    iou = max(bbox_iou_value, polygon_iou_value)
    threshold = _match_threshold(expected_bbox)
    debug: dict[str, Any] = {
        "bbox_iou": _round_debug(bbox_iou_value),
        "polygon_iou": _round_debug(polygon_iou_value),
        "iou": _round_debug(iou),
        "score": 0.0,
        "threshold": _round_debug(threshold),
        "passed": False,
    }

    if iou < IOU_MATCH_THRESHOLD:
        debug["reject_reason"] = "iou_below_min"
        return debug

    expected_area = bbox_area(expected_bbox)
    detection_area = bbox_area(detection_bbox)
    if expected_area <= 0 or detection_area <= 0:
        debug["reject_reason"] = "empty_bbox"
        return debug

    area_ratio = detection_area / expected_area
    debug["area_ratio"] = _round_debug(area_ratio)
    if area_ratio < _MIN_AREA_RATIO or area_ratio > _MAX_AREA_RATIO:
        debug["reject_reason"] = "area_ratio_out_of_range"
        return debug

    expected_center = bbox_center(expected_bbox)
    detection_center = bbox_center(detection_bbox)
    center_distance = float(
        np.hypot(
            expected_center[0] - detection_center[0],
            expected_center[1] - detection_center[1],
        )
    )
    expected_diag = max(1.0, bbox_diag(expected_bbox))
    center_limit = expected_diag * _MAX_CENTER_DISTANCE_FACTOR
    center_score = max(0.0, 1.0 - center_distance / expected_diag)
    debug["center_distance"] = _round_debug(center_distance)
    debug["center_limit"] = _round_debug(center_limit)

    if center_distance > center_limit:
        debug["reject_reason"] = "center_too_far"
        return debug

    area_score = min(area_ratio, 1.0 / area_ratio)
    score = (
        polygon_iou_value * 0.50
        + bbox_iou_value * 0.25
        + center_score * 0.20
        + area_score * 0.05
    )
    debug["score"] = _round_debug(score)

    if score < threshold:
        debug["reject_reason"] = "score_below_threshold"
        return debug

    debug["passed"] = True
    debug["reject_reason"] = None
    return debug


def _round_debug(value: float) -> float:
    return round(float(value), 4)


def _match_threshold(expected_bbox: BBox) -> float:
    area = bbox_area(expected_bbox)
    if area <= 32 * 32:
        return min(_MIN_MATCH_SCORE, 0.34)
    if area >= 180 * 180:
        return max(_MIN_MATCH_SCORE, 0.46)
    return _MIN_MATCH_SCORE


def _choose_candidate_pairs(
    candidate_pairs: list[tuple[float, float, int, int]],
) -> list[tuple[int, int, float]]:
    if not candidate_pairs:
        return []

    expected_indices = sorted({item[2] for item in candidate_pairs})
    detection_indices = sorted({item[3] for item in candidate_pairs})

    if (
        len(candidate_pairs) > _MAX_OPTIMAL_ASSIGNMENT_CANDIDATES
        or len(expected_indices) > _MAX_OPTIMAL_ASSIGNMENT_ITEMS
        or len(detection_indices) > _MAX_OPTIMAL_ASSIGNMENT_ITEMS
    ):
        return _choose_candidate_pairs_greedy(candidate_pairs)

    expected_pos = {value: index for index, value in enumerate(expected_indices)}
    detection_pos = {value: index for index, value in enumerate(detection_indices)}

    by_expected: list[list[tuple[int, float, float, int]]] = [
        [] for _ in expected_indices
    ]
    for score, iou, expected_index, detection_index in candidate_pairs:
        by_expected[expected_pos[expected_index]].append(
            (detection_pos[detection_index], score, iou, detection_index)
        )

    states: dict[int, tuple[float, list[tuple[int, int, float]]]] = {0: (0.0, [])}
    for expected_index, options in zip(expected_indices, by_expected, strict=True):
        next_states = dict(states)

        for used_mask, (total_score, chosen) in states.items():
            for detection_bit, score, iou, detection_index in options:
                bit = 1 << detection_bit
                if used_mask & bit:
                    continue

                next_mask = used_mask | bit
                candidate_score = total_score + score
                candidate_chosen = [*chosen, (expected_index, detection_index, iou)]
                previous = next_states.get(next_mask)

                if previous is None or _assignment_better(
                    candidate_score,
                    candidate_chosen,
                    previous[0],
                    previous[1],
                ):
                    next_states[next_mask] = (candidate_score, candidate_chosen)

        states = next_states

    return max(
        (value[1] for value in states.values()),
        key=lambda chosen: (
            len(chosen),
            sum(iou for _, _, iou in chosen),
        ),
    )


def _assignment_better(
    candidate_score: float,
    candidate_chosen: list[tuple[int, int, float]],
    previous_score: float,
    previous_chosen: list[tuple[int, int, float]],
) -> bool:
    if len(candidate_chosen) != len(previous_chosen):
        return len(candidate_chosen) > len(previous_chosen)
    return candidate_score > previous_score


def _choose_candidate_pairs_greedy(
    candidate_pairs: list[tuple[float, float, int, int]],
) -> list[tuple[int, int, float]]:
    candidate_pairs = sorted(candidate_pairs, key=lambda item: item[0], reverse=True)
    matched_expected_indices: set[int] = set()
    matched_detection_indices: set[int] = set()
    chosen_pairs: list[tuple[int, int, float]] = []

    for _score, iou, expected_index, detection_index in candidate_pairs:
        if expected_index in matched_expected_indices:
            continue
        if detection_index in matched_detection_indices:
            continue

        matched_expected_indices.add(expected_index)
        matched_detection_indices.add(detection_index)
        chosen_pairs.append((expected_index, detection_index, iou))

    return chosen_pairs

def _classify_unmatched_detection(
    detection: DetectionCandidate,
    projected_expected: list[ProjectedExpected],
    occupied_expected_indices: set[int],
    *,
    slot_by_index: dict[int, ExpectedSlot],
    occupied_detection_indices: set[int],
    detection_by_index: dict[int, DetectionCandidate],
    yolo_anchor_rescue: bool = False,
    iou_threshold: float = 0.10,
) -> tuple[str, int | None, dict[str, Any] | None]:
    if detection.detection.confidence < settings.YOLO_EXTRA_CONF_THRESHOLD:
        return (
            "discard",
            None,
            {
                "reason": "confidence_below_extra_threshold",
                "confidence": _round_debug(detection.detection.confidence),
                "threshold": _round_debug(settings.YOLO_EXTRA_CONF_THRESHOLD),
            },
        )

    duplicate_debug = _occupied_detection_duplicate_debug(
        detection,
        occupied_detection_indices=occupied_detection_indices,
        detection_by_index=detection_by_index,
    )
    if duplicate_debug is not None:
        return "discard", None, duplicate_debug

    best_same_class: tuple[float, float, ProjectedExpected, dict[str, Any]] | None = (
        None
    )

    for expected_item in projected_expected:
        if expected_item.index in occupied_expected_indices:
            continue
        if expected_item.item.class_key != detection.detection.class_key:
            continue

        global_debug = _match_score_details(
            expected_item.polygon,
            detection.polygon,
            expected_item.bbox,
            detection.bbox,
        )
        slot = slot_by_index.get(expected_item.index)
        if yolo_anchor_rescue:
            slot_details = try_runtime_yolo_anchor_candidate(
                expected_item,
                detection,
                slot=slot,
                global_debug=global_debug,
                slot_debug=_slot_debug_payload(slot),
            )
        else:
            slot_details = _slot_candidate_details(
                expected_item,
                detection,
                slot=slot,
                global_debug=global_debug,
            )
        if slot_details is None:
            continue

        score, _projected_iou, debug = slot_details
        containment = float(debug.get("slot_detection_containment") or 0.0)
        candidate = (score, containment, expected_item, debug)

        if best_same_class is None or (score, containment) > (
            best_same_class[0],
            best_same_class[1],
        ):
            best_same_class = candidate

    if best_same_class is not None:
        _score, _containment, expected_item, debug = best_same_class
        debug = {
            **debug,
            "reason": "same_class_slot_detection_not_matched",
            "expected_name": expected_item.item.name,
        }
        debug.pop("iou", None)
        return "unmatched", expected_item.index, debug

    for expected_item in projected_expected:
        iou = bbox_iou(expected_item.bbox, detection.bbox)

        if iou < iou_threshold:
            continue

        return (
            "discard",
            expected_item.index,
            {
                "reason": "inside_expected_zone",
                "bbox_iou": _round_debug(iou),
                "expected_name": expected_item.item.name,
            },
        )

    return (
        "extra",
        None,
        {
            "reason": "no_unmatched_expected_slot_of_same_class",
            "confidence": _round_debug(detection.detection.confidence),
        },
    )


def _occupied_detection_duplicate_debug(
    detection: DetectionCandidate,
    *,
    occupied_detection_indices: set[int],
    detection_by_index: dict[int, DetectionCandidate],
) -> dict[str, Any] | None:
    for occupied_index in occupied_detection_indices:
        occupied_detection = detection_by_index.get(occupied_index)
        if occupied_detection is None:
            continue

        bbox_iou = bbox_iou(detection.bbox, occupied_detection.bbox)
        containment = max(
            bbox_containment(detection.bbox, occupied_detection.bbox),
            bbox_containment(occupied_detection.bbox, detection.bbox),
        )

        if (
            bbox_iou < _DUPLICATE_MATCHED_BBOX_IOU
            and containment < _DUPLICATE_MATCHED_CONTAINMENT
        ):
            continue

        return {
            "reason": "duplicate_of_accepted_detection",
            "accepted_detection_index": occupied_index,
            "accepted_class_key": occupied_detection.detection.class_key,
            "bbox_iou": _round_debug(bbox_iou),
            "containment": _round_debug(containment),
        }

    return None
