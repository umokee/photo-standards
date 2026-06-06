from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from modules.yolo.inspection.domain.alignment import (
    LocalProjectionData,
    project_polygon,
    project_polygon_adaptive,
)
from modules.yolo.inspection.domain.matcher_anchor_release import (
    _try_missing_anchor_release,
)
from modules.yolo.inspection.domain.matcher_context import (
    _all_expected_context_exclusion_zones,
    _missing_context_quadrant_count,
    _missing_context_spread_score,
    _missing_polygon_refinement_points,
    _missing_rescue_overlaps_other_expected,
    _slot_feature_support,
)
from modules.yolo.inspection.domain.matcher_debug import (
    _bbox_debug,
    _round_debug,
    _slot_debug_payload,
)
from modules.yolo.inspection.domain.matcher_geometry import (
    affine_reprojection_median_error,
    as_match_points,
    bbox_area_similarity,
    bbox_center,
    bbox_center_distance_factor,
    bbox_containment,
    bbox_diag,
    bbox_from_polygon,
    expand_bbox,
    is_visible_in_frame,
    point_in_any_bbox,
    point_in_bbox,
    polygon_from_bbox,
    project_points_with_homography,
    project_polygon_by_affine,
    translate_polygon,
)
from modules.yolo.inspection.domain.matcher_local_displacement import (
    _merge_missing_local_displacement_debug,
    _try_missing_local_displacement_field,
)
from modules.yolo.inspection.domain.matcher_missing_candidates import (
    append_missing_projection_candidate,
    finalize_missing_candidate_agreement_debug,
    max_overlap_with_other_expected,
    reference_area_score_for_expected,
)
from modules.yolo.inspection.domain.matcher_projection_candidates import (
    apply_projection_candidate,
    global_fallback_candidate,
    local_displacement_candidate,
    translation_rescue_candidate,
)
from modules.yolo.inspection.domain.matcher_projection_validation import (
    _missing_fallback_projection_unsafe_reason,
    _missing_global_projection,
    validate_translated_global_candidate,
)
from modules.yolo.inspection.domain.matcher_runtime_fusion import (
    build_runtime_yolo_anchor_trace,
    merge_runtime_yolo_anchor_trace_debug,
    merge_runtime_yolo_no_anchor_debug,
    record_runtime_yolo_anchor_trace_candidate,
    runtime_yolo_anchor_candidate_details,
)
from modules.yolo.inspection.domain.matcher_scoring import (
    _choose_candidate_pairs,
    _classify_unmatched_detection,
    _match_score_details,
    _try_slot_candidate,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
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
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds
from modules.yolo.inspection.domain.matcher_translation_solver import (
    solve_translation_from_projected_points,
)
from modules.yolo.inspection.domain.matcher_trusted_anchors import (
    _append_or_replace_trusted_anchor,
    _build_trusted_missing_anchors,
    _trusted_anchor_from_matched_detection,
    _trusted_anchor_from_resolved_projection,
)
from modules.yolo.inspection.domain.runtime_fusion_contract import (
    APPLIED_GEOMETRY_SOURCE_BASELINE,
    APPLIED_GEOMETRY_SOURCE_TRUSTED_ANCHOR_TRANSFORM,
)
from modules.yolo.inspection.domain.types import (
    ExpectedSegment,
    SegmentMatch,
    YoloDetection,
)

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
                visible = bbox is not None and (
                    effective_frame_size is None
                    or is_visible_in_frame(
                        bbox,
                        effective_frame_size,
                        min_visible_fraction=0.05,
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
        translation_rescue_fn=_try_missing_translation_rescue,
        polygon_refinement_fn=_try_missing_polygon_refinement,
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
    matched_detection_indices = {
        detection_index for _, detection_index, _ in chosen_pairs
    }

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
            bbox=bbox_from_polygon(missing_polygon)
            if missing_polygon is not None
            else None,
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
    missing_names = [
        match.name for match in expected_matches if match.status == "missing"
    ]
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
            frame_size=(
                projection_data.frame_size if projection_data is not None else None
            ),
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
            debug.get("applied_geometry_source") or APPLIED_GEOMETRY_SOURCE_BASELINE
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
                "slot_feature_min_support": (_thresholds.slot_min_feature_support),
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

    if projection_data.frame_size is not None and not is_visible_in_frame(
        bbox,
        projection_data.frame_size,
        min_visible_fraction=0.05,
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
        and (
            inlier_ratio <= max_ratio
            or center_factor >= min_center
            or area_score <= min_area
        )
    ):
        return "weak_context_affine_unstable"

    if sparse_evidence and weak_geometry and center_factor >= min_center * 0.85:
        return "sparse_context_affine_center_drift"

    if (
        weak_evidence
        and weak_geometry
        and area_score <= min_area * 0.94
        and median_error >= min_error * 0.85
    ):
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
        "missing_polygon_translation_shift_factor": _round_debug(rescue.shift_factor),
        "missing_polygon_replaced_projection": replaced_projection,
        "missing_polygon_replaced_median_error": _round_debug(replaced_median_error),
        "missing_polygon_replaced_inlier_ratio": _round_debug(replaced_inlier_ratio),
        "missing_polygon_replaced_area_score": _round_debug(replaced_area_score),
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


def _use_missing_translation_projection(
    state: MissingFallbackState,
    rescue: MissingTranslationRescue,
    *,
    projection: str,
    safety: str,
    source: str,
    **payload_kwargs: Any,
) -> None:
    apply_projection_candidate(
        state,
        translation_rescue_candidate(
            rescue,
            source=source,
            debug=_missing_translation_rescue_debug_payload(
                projection,
                safety,
                rescue,
                **payload_kwargs,
            ),
        ),
    )


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
    debug = _merge_missing_local_displacement_debug(state.debug, displacement)
    debug.update(
        {
            "missing_polygon_fallback_source": "local_displacement",
            "missing_polygon_fallback_reason": "sparse_local_displacement",
        }
    )
    if extra_debug:
        debug.update(extra_debug)
    apply_projection_candidate(
        state,
        local_displacement_candidate(
            displacement,
            debug=debug,
        ),
    )


def _use_missing_global_fallback_projection(
    state: MissingFallbackState,
    *,
    global_polygon: list[list[float]],
    global_bbox: BBox,
    local_global_area_score: float,
    local_global_center_factor: float,
) -> None:
    apply_projection_candidate(
        state,
        global_fallback_candidate(
            polygon=global_polygon,
            bbox=global_bbox,
            debug=_missing_projection_debug_payload(
                projection="expected_slot_global_fallback",
                safety="global_fallback",
                include_projection_alias=True,
                missing_polygon_fallback_source="global_fallback",
                missing_polygon_fallback_reason="local_global_disagrees",
                **_local_global_debug_payload(
                    local_global_area_score,
                    local_global_center_factor,
                ),
            ),
        ),
    )


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
        and support_ratio < _SELECTIVE_HIDDEN_GLOBAL_FALLBACK_MIN_SUPPORT_RATIO
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

    expected_slot_agreement_debug = (
        _try_selective_hidden_expected_slot_agreement_release(
            expected_item,
            projection_data=projection_data,
            all_expected=all_expected,
            fallback_polygon=state.polygon,
            fallback_bbox=state.bbox,
            global_bbox=state.global_candidate_bbox,
            hidden_reason=hidden_reason,
            fallback_source=state.source,
        )
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
    if not release_debug or not release_debug.get(
        "missing_polygon_selective_hidden_release"
    ):
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
    def reject(reason: str) -> None:
        _record_missing_translation_rescue_reject(reject_debug, reason)
        return None

    if slot is None or projection_data is None:
        return reject("no_slot_or_projection_data")
    if projection_data.global_homography is None:
        return reject("no_global_homography")

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
        return reject("too_few_local_points")

    projected_reference = project_points_with_homography(
        local_reference,
        projection_data.global_homography,
    )
    if projected_reference is None or len(projected_reference) != len(local_frame):
        return reject("projection_failed")

    threshold = float(
        _thresholds.missing_rescue_translation_max_residual_error
        if max_residual_error_override is None
        else max_residual_error_override
    )
    solve, solve_reject_reason = solve_translation_from_projected_points(
        local_frame=local_frame,
        projected_reference=projected_reference,
        min_support=min_support,
        threshold=threshold,
        global_bbox=global_bbox,
    )
    if solve is None:
        return reject(solve_reject_reason or "invalid_residuals")

    inlier_ratio = solve.inlier_ratio
    min_inlier_ratio = (
        _thresholds.missing_rescue_translation_min_inlier_ratio
        if min_inlier_ratio_override is None
        else float(min_inlier_ratio_override)
    )
    if inlier_ratio < min_inlier_ratio:
        return reject("low_inlier_ratio")

    inlier_reference = local_reference[solve.inlier_mask]
    reference_bbox = slot.reference_bbox or bbox_from_polygon(
        expected_item.item.reference_polygon
    )
    if reference_bbox is not None:
        spread = _missing_context_spread_score(inlier_reference, reference_bbox)
        min_spread = (
            max(0.07, _thresholds.missing_polygon_context_min_spread * 0.55)
            if min_context_spread_override is None
            else float(min_context_spread_override)
        )
        if spread < min_spread:
            return reject("low_context_spread")
    else:
        spread = 0.0

    median_error = solve.median_error
    if median_error > threshold:
        return reject("high_median_error")

    shift_x = solve.shift_x
    shift_y = solve.shift_y
    shift_factor = solve.shift_factor
    max_shift_factor = (
        _thresholds.missing_rescue_translation_max_shift_factor
        if max_shift_factor_override is None
        else float(max_shift_factor_override)
    )
    if shift_factor > max_shift_factor:
        return reject("large_shift")

    polygon = translate_polygon(global_polygon, dx=shift_x, dy=shift_y)
    bbox = bbox_from_polygon(polygon)
    min_search_containment = (
        0.05
        if min_search_containment_override is None
        else float(min_search_containment_override)
    )
    validation = validate_translated_global_candidate(
        expected_item,
        polygon=polygon,
        bbox=bbox,
        global_bbox=global_bbox,
        slot=slot,
        projection_data=projection_data,
        max_center_factor=max(0.05, max_shift_factor),
        min_search_containment=min_search_containment,
        max_other_overlap=max_other_overlap,
        all_expected=all_expected,
    )
    if not validation.accepted:
        return reject(validation.reject_reason or "invalid_projection")

    _record_missing_translation_rescue_reject(
        reject_debug,
        "accepted",
    )
    return MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=solve.candidate_count,
        inlier_count=solve.inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
    )


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
    reference_exclusion_zones, frame_exclusion_zones = (
        _all_expected_context_exclusion_zones(
            expected_item,
            all_expected=all_expected,
            projection_data=projection_data,
        )
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
        if (
            float(np.linalg.norm(frame_point[:2] - projected_point[:2]))
            > residual_prefilter
        ):
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

    threshold = float(_thresholds.missing_scene_rescue_max_residual_error)
    solve, _ = solve_translation_from_projected_points(
        local_frame=local_frame,
        projected_reference=projected_local,
        min_support=min_support,
        threshold=threshold,
        global_bbox=global_bbox,
    )
    if solve is None:
        return None

    inlier_ratio = solve.inlier_ratio
    if inlier_ratio < _thresholds.missing_scene_rescue_min_inlier_ratio:
        return None

    inlier_reference = local_reference[solve.inlier_mask]
    spread = _missing_context_spread_score(inlier_reference, reference_bbox)
    if spread < _thresholds.missing_scene_rescue_min_spread:
        return None

    inlier_residuals = solve.residuals[solve.inlier_mask, :2]
    median_residual = np.median(inlier_residuals, axis=0).astype(np.float32)
    residual_errors = np.linalg.norm(
        inlier_residuals - median_residual[None, :], axis=1
    )
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
    bbox = bbox_from_polygon(polygon)
    validation = validate_translated_global_candidate(
        expected_item,
        polygon=polygon,
        bbox=bbox,
        global_bbox=global_bbox,
        slot=slot,
        projection_data=projection_data,
        max_center_factor=max(
            0.05,
            _thresholds.missing_scene_rescue_max_shift_factor,
        ),
        min_search_containment=0.03 if slot is not None else None,
        all_expected=all_expected,
        max_other_overlap=_thresholds.missing_scene_rescue_max_other_overlap,
    )
    if not validation.accepted:
        return None

    return MissingTranslationRescue(
        polygon=polygon,
        bbox=bbox,
        candidate_count=solve.candidate_count,
        inlier_count=solve.inlier_count,
        inlier_ratio=float(inlier_ratio),
        median_error=float(median_error),
        shift_x=shift_x,
        shift_y=shift_y,
        shift_factor=float(shift_factor),
    )


def _missing_debug_projection_name(debug: dict[str, Any] | None) -> str:
    if not debug:
        return ""
    value = debug.get("missing_polygon_projection") or debug.get("projection")
    return str(value or "")


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


def _display_detected_polygon(
    detection_item: DetectionCandidate,
) -> list[list[float]] | None:
    return detection_item.polygon
