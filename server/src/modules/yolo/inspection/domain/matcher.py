from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from modules.yolo.inspection.domain.alignment import (
    LocalProjectionData,
    project_polygon,
)
from modules.yolo.inspection.domain.types import (
    ExpectedSegment,
    SegmentMatch,
    YoloDetection,
)

_MIN_POLYGON_POINTS = 3
_MIN_MATCH_IOU = 0.10
_MAX_CENTER_DISTANCE_FACTOR = 0.60
_MIN_VISIBLE_FRACTION = 0.04
_UNMATCHED_NEARBY_CENTER_DISTANCE_FACTOR = 1.35
_UNMATCHED_RELAXED_IOU = 0.015
_DUPLICATE_DETECTION_IOU = 0.70
_DUPLICATE_DETECTION_CONTAINMENT = 0.85
_MIN_EXPECTED_SLOT_CONFIDENCE = 0.05
_MIN_EXTRA_CONFIDENCE = 0.25
_RELAXED_SAME_CLASS_CENTER_FACTOR = 1.10
_RELAXED_SAME_CLASS_MIN_IOU = 0.015
_AMBIGUOUS_SLOT_MIN_SAME_CLASS_COUNT = 2

BBox = tuple[float, float, float, float]


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
            if len(polygon) < _MIN_POLYGON_POINTS:
                continue
            items.append(
                ExpectedSegment(
                    annotation_id=annotation.id,
                    segment_class_id=segment_class.id,
                    class_key=str(segment_class.id),
                    name=segment_class.name,
                    hue=segment_class.hue,
                    reference_polygon=_clean_polygon(polygon),
                )
            )

    return items


def build_missing_matches(
    expected: list[ExpectedSegment],
    homography: np.ndarray | None = None,
    *,
    frame_size: tuple[int, int] | None = None,
    projection_data: LocalProjectionData | None = None,
    hide_unconfirmed_projection: bool = False,
) -> list[SegmentMatch]:
    matches: list[SegmentMatch] = []
    transform = _resolve_homography(homography, projection_data)

    for item in expected:
        projected, reason, projection_debug = _project_expected_polygon(
            item,
            transform,
            frame_size=_resolve_frame_size(frame_size, projection_data),
        )
        unconfirmed = hide_unconfirmed_projection or projected is None
        matches.append(
            _expected_match(
                item,
                status="missing",
                expected_polygon=None if unconfirmed else projected,
                debug=_with_projection_debug(
                        _missing_debug(
                            projection="feature_lightglue_homography"
                            if projected
                            else "unconfirmed_hidden",
                            safety="unsafe_hidden" if unconfirmed else "confirmed",
                            reason=reason or "no_matching_detection",
                            reason_code=(
                                "scene_pose_unconfirmed"
                                if unconfirmed
                                else "no_matching_detection"
                            ),
                        ),
                        projection_debug=projection_debug,
                    ),
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
) -> list[SegmentMatch]:
    """Assign detections to expected slots with one row per expected object.

    v17_status_conflict_fix:
    - expected-slot rows own the final status: ok / unmatched / missing;
    - a same-class YOLO detection is first consumed by an unresolved expected slot
      before it can become an extra row;
    - missing polygons are kept when the homography produced a finite polygon even
      if the polygon is partly outside the frame, so overlay can clip and draw it;
    - extra rows are only remaining detections after expected slots are satisfied.
    """
    transform = _resolve_homography(homography, projection_data)
    resolved_frame_size = _resolve_frame_size(frame_size, projection_data)

    projected: list[_ProjectedSlot] = []
    hidden_expected: list[
        tuple[int, ExpectedSegment, list[list[float]] | None, str, dict[str, Any]]
    ] = []
    for index, item in enumerate(expected):
        polygon, reason, projection_debug = _project_expected_polygon(
            item,
            transform,
            frame_size=resolved_frame_size,
        )
        if polygon is None:
            hidden_expected.append(
                (index, item, None, reason or "scene_pose_unconfirmed", projection_debug)
            )
            continue
        bbox = _bbox_from_polygon(polygon)
        if bbox is None:
            hidden_expected.append((index, item, polygon, "invalid_projected_polygon", projection_debug))
            continue
        projected.append(
            _ProjectedSlot(index=index, item=item, polygon=polygon, bbox=bbox, debug=projection_debug)
        )

    detection_items = [
        _DetectionSlot(
            index=index, detection=detection, polygon=_detection_polygon(detection)
        )
        for index, detection in enumerate(detections)
    ]
    detection_items = [item for item in detection_items if item.bbox is not None]
    detection_items = _dedupe_detection_items(detection_items)
    ambiguous_slots = _find_ambiguous_same_class_slots(projected, detection_items)

    if not projected and not hidden_expected:
        return []

    if not projected and not detection_items:
        return build_missing_matches(
            expected,
            transform,
            frame_size=resolved_frame_size,
            hide_unconfirmed_projection=transform is None,
        )

    candidates: list[tuple[float, float, int, int, dict[str, Any]]] = []
    for slot in projected:
        if slot.index in ambiguous_slots:
            continue
        for det in detection_items:
            if slot.item.class_key != det.detection.class_key:
                continue
            score, iou, center_factor, center_inside = _score_slot_detection(slot, det)
            acceptable, match_source = _candidate_is_acceptable(
                iou=iou,
                center_factor=center_factor,
                center_inside=center_inside,
                class_matches=True,
                confidence=det.detection.confidence,
            )
            if acceptable:
                candidates.append(
                    (
                        score,
                        iou,
                        slot.index,
                        det.index,
                        {
                            "match_source": match_source,
                            "projection": dict(slot.debug or {}),
                            "iou": _round(iou),
                            "center_distance_factor": _round(center_factor),
                            "center_inside_projected_slot": center_inside,
                            "expected_slot_low_conf_allowed": (
                                float(det.detection.confidence or 0.0)
                                >= _MIN_EXPECTED_SLOT_CONFIDENCE
                            ),
                        },
                    )
                )

    candidates.sort(key=lambda item: item[0], reverse=True)
    used_expected: set[int] = set()
    used_detections: set[int] = set()
    chosen: list[tuple[int, int, float, dict[str, Any]]] = []
    for _score, iou, expected_index, detection_index, debug in candidates:
        if expected_index in used_expected or detection_index in used_detections:
            continue
        used_expected.add(expected_index)
        used_detections.add(detection_index)
        chosen.append((expected_index, detection_index, iou, debug))

    projected_by_index = {slot.index: slot for slot in projected}
    detection_by_index = {item.index: item for item in detection_items}
    matches: list[SegmentMatch] = []

    for slot_index, detection_indices in ambiguous_slots.items():
        if slot_index in used_expected:
            continue
        slot = projected_by_index.get(slot_index)
        if slot is None:
            continue
        existing_indices = [index for index in detection_indices if index in detection_by_index]
        if len(existing_indices) < _AMBIGUOUS_SLOT_MIN_SAME_CLASS_COUNT:
            continue
        best_index = max(
            existing_indices,
            key=lambda index: float(detection_by_index[index].detection.confidence or 0.0),
        )
        best_det = detection_by_index[best_index]
        used_expected.add(slot.index)
        used_detections.update(existing_indices)
        debug = _with_projection_debug(
            {
                "reason": _reason_message("multiple_same_class_detections_inside_projected_slot"),
                "reason_code": "multiple_same_class_detections_inside_projected_slot",
                "expected_class_key": slot.item.class_key,
                "same_class_detection_count": len(existing_indices),
                "detection_indices": existing_indices,
                "assignment_policy": "slot_marked_ambiguous_no_silent_ok",
            },
            projection_debug=slot.debug,
        )
        match = _expected_match(
            slot.item,
            status="unmatched",
            confidence=best_det.detection.confidence,
            expected_polygon=slot.polygon,
            detected_polygon=best_det.polygon,
            detected_bbox=best_det.detection.bbox,
            debug=debug,
        )
        match.detected_class_in_zone = best_det.detection.class_key
        matches.append(match)

    for expected_index, detection_index, iou, debug in chosen:
        slot = projected_by_index[expected_index]
        det = detection_by_index[detection_index]
        matches.append(
            _expected_match(
                slot.item,
                status="ok",
                iou=iou,
                confidence=det.detection.confidence,
                expected_polygon=slot.polygon,
                detected_polygon=det.polygon,
                detected_bbox=det.detection.bbox,
                debug=_with_projection_debug(debug, projection_debug=slot.debug),
            )
        )

    unmatched_assignments = _choose_unmatched_assignments(
        projected,
        detection_items,
        used_expected=used_expected,
        used_detections=used_detections,
    )
    used_unmatched_detections = {
        det_index for det_index, _debug in unmatched_assignments.values()
    }

    unresolved: list[
        tuple[int, ExpectedSegment, list[list[float]] | None, BBox | None, str, dict[str, Any]]
    ] = []
    for slot in projected:
        if slot.index in used_expected:
            continue
        unresolved.append(
            (
                slot.index,
                slot.item,
                slot.polygon,
                slot.bbox,
                "no_detection_in_projected_slot",
                dict(slot.debug or {}),
            )
        )
    for index, item, polygon, reason, projection_debug in hidden_expected:
        if index in used_expected:
            continue
        bbox = _bbox_from_polygon(polygon) if polygon is not None else None
        unresolved.append((index, item, polygon, bbox, reason, projection_debug))

    same_class_assignments = _choose_same_class_unresolved_assignments(
        unresolved,
        detection_items,
        occupied_detection_indices=used_detections | used_unmatched_detections,
    )
    used_same_class_detections = {
        det_index for det_index, _debug in same_class_assignments.values()
    }

    for expected_index, item, polygon, _bbox, reason, projection_debug in unresolved:
        if expected_index in used_expected:
            continue

        unmatched = unmatched_assignments.get(expected_index)
        if unmatched is None:
            unmatched = same_class_assignments.get(expected_index)

        if unmatched is not None:
            detection_index, debug = unmatched
            det = detection_by_index[detection_index]
            reason_code = str(debug.get("reason_code") or "")

            # v24_1_wrong_class_inside_slot_missing:
            # If YOLO detects a different class inside the expected slot, the
            # expected object is still missing. Do not attach the wrong YOLO
            # mask/polygon/bbox to the expected row; consume the detection so it
            # also does not become a duplicate extra for the same physical zone.
            if reason_code == "different_class_detection_inside_projected_slot":
                wrong_debug = _with_projection_debug(dict(debug), projection_debug=projection_debug)
                wrong_debug["detected_mask_ignored"] = True
                wrong_debug["wrong_class_detection_consumed"] = True
                match = _expected_match(
                    item,
                    status="missing",
                    confidence=det.detection.confidence,
                    expected_polygon=polygon,
                    detected_polygon=None,
                    detected_bbox=None,
                    debug=wrong_debug,
                )
                match.detected_class_in_zone = det.detection.class_key
                matches.append(match)
                continue

            match = _expected_match(
                item,
                status="unmatched",
                confidence=det.detection.confidence,
                expected_polygon=polygon,
                detected_polygon=det.polygon,
                detected_bbox=det.detection.bbox,
                debug=_with_projection_debug(debug, projection_debug=projection_debug),
            )
            match.detected_class_in_zone = det.detection.class_key
            matches.append(match)
            continue

        confirmed_projection = polygon is not None and transform is not None
        matches.append(
            _expected_match(
                item,
                status="missing",
                expected_polygon=polygon,
                debug=_with_projection_debug(
                        _missing_debug(
                            projection="feature_lightglue_homography"
                            if confirmed_projection
                            else "unconfirmed_hidden",
                            safety="confirmed" if confirmed_projection else "unsafe_hidden",
                            reason=(
                                "no_detection_in_projected_slot"
                                if confirmed_projection
                                else reason
                            ),
                            reason_code=(
                                "no_detection_in_projected_slot"
                                if confirmed_projection
                                else "scene_pose_unconfirmed"
                            ),
                        ),
                        projection_debug=projection_debug,
                    ),
            )
        )

    occupied_detection_indices = (
        used_detections | used_unmatched_detections | used_same_class_detections
    )
    extra_items = _collapse_extra_detections(
        detection_items,
        projected,
        detection_by_index=detection_by_index,
        occupied_detection_indices=occupied_detection_indices,
        expected=expected,
    )
    for det, debug, display_item in extra_items:
        matches.append(
            _detection_match(
                det.detection,
                name=display_item.name if display_item is not None else "Лишняя деталь",
                hue=display_item.hue if display_item is not None else None,
                status="extra",
                detected_polygon=det.polygon,
                debug=debug,
            )
        )

    return matches


def _find_ambiguous_same_class_slots(
    projected: list[_ProjectedSlot],
    detection_items: list[_DetectionSlot],
) -> dict[int, list[int]]:
    ambiguous: dict[int, list[int]] = {}
    for slot in projected:
        same_class_indices: list[int] = []
        for det in detection_items:
            if det.bbox is None:
                continue
            if det.detection.class_key != slot.item.class_key:
                continue
            if float(det.detection.confidence or 0.0) < _MIN_EXPECTED_SLOT_CONFIDENCE:
                continue
            if _same_class_detection_in_slot_zone(slot, det):
                same_class_indices.append(det.index)

        if len(same_class_indices) >= _AMBIGUOUS_SLOT_MIN_SAME_CLASS_COUNT:
            ambiguous[slot.index] = same_class_indices
    return ambiguous


def _same_class_detection_in_slot_zone(slot: _ProjectedSlot, det: _DetectionSlot) -> bool:
    if det.bbox is None:
        return False
    iou = _bbox_iou(slot.bbox, det.bbox)
    center_factor = _center_distance_factor(slot.bbox, det.bbox)
    center_inside = _point_in_bbox(_bbox_center(det.bbox), slot.bbox)
    return (
        center_inside
        or iou >= _RELAXED_SAME_CLASS_MIN_IOU
        or center_factor <= _RELAXED_SAME_CLASS_CENTER_FACTOR
    )

def _choose_unmatched_assignments(
    projected: list[_ProjectedSlot],
    detection_items: list[_DetectionSlot],
    *,
    used_expected: set[int],
    used_detections: set[int],
) -> dict[int, tuple[int, dict[str, Any]]]:
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for slot in projected:
        if slot.index in used_expected:
            continue
        for det in detection_items:
            if det.index in used_detections:
                continue
            candidate = _unmatched_candidate(slot, det)
            if candidate is None:
                continue
            score, debug = candidate
            candidates.append((score, slot.index, det.index, debug))

    candidates.sort(key=lambda item: item[0], reverse=True)
    assignments: dict[int, tuple[int, dict[str, Any]]] = {}
    occupied_detections: set[int] = set()
    for _score, expected_index, detection_index, debug in candidates:
        if expected_index in assignments or detection_index in occupied_detections:
            continue
        assignments[expected_index] = (detection_index, debug)
        occupied_detections.add(detection_index)
    return assignments


def _unmatched_candidate(
    slot: _ProjectedSlot,
    det: _DetectionSlot,
) -> tuple[float, dict[str, Any]] | None:
    if det.bbox is None:
        return None

    iou = _bbox_iou(slot.bbox, det.bbox)
    center_factor = _center_distance_factor(slot.bbox, det.bbox)
    center_inside = _point_in_bbox(_bbox_center(det.bbox), slot.bbox)
    class_matches = slot.item.class_key == det.detection.class_key

    if class_matches:
        accepted = (
            center_inside
            or iou >= _UNMATCHED_RELAXED_IOU
            or center_factor <= _UNMATCHED_NEARBY_CENTER_DISTANCE_FACTOR
        )
    else:
        accepted = center_inside or iou >= _MIN_MATCH_IOU

    if not accepted:
        return None

    score = (
        (2.0 if class_matches else 0.0)
        + iou * 3.0
        - center_factor
        + 0.001 * float(det.detection.confidence or 0.0)
    )
    reason_code = (
        "same_class_detection_near_projected_slot_but_not_matched"
        if class_matches
        else "different_class_detection_inside_projected_slot"
    )
    debug = {
        "reason": _reason_message(reason_code),
        "reason_code": reason_code,
        "expected_class_key": slot.item.class_key,
        "detected_class_key": det.detection.class_key,
        "iou": _round(iou),
        "center_distance_factor": _round(center_factor),
        "center_inside_projected_slot": center_inside,
    }
    return score, debug


def _choose_same_class_unresolved_assignments(
    unresolved: list[
        tuple[int, ExpectedSegment, list[list[float]] | None, BBox | None, str, dict[str, Any]]
    ],
    detection_items: list[_DetectionSlot],
    *,
    occupied_detection_indices: set[int],
) -> dict[int, tuple[int, dict[str, Any]]]:
    candidates: list[tuple[float, int, int, dict[str, Any]]] = []
    for expected_index, item, _polygon, bbox, reason, _projection_debug in unresolved:
        for det in detection_items:
            if det.index in occupied_detection_indices:
                continue
            if det.detection.class_key != item.class_key:
                continue
            score = float(det.detection.confidence or 0.0)
            center_factor: float | None = None
            iou: float | None = None
            if bbox is not None and det.bbox is not None:
                center_factor = _center_distance_factor(bbox, det.bbox)
                iou = _bbox_iou(bbox, det.bbox)
                score += (iou * 1.5) - (0.05 * min(center_factor, 20.0))
            reason_code = "same_class_detection_exists_but_expected_slot_not_confirmed"
            debug = {
                "reason": _reason_message(reason_code),
                "reason_code": reason_code,
                "raw_missing_reason": reason,
                "expected_class_key": item.class_key,
                "detected_class_key": det.detection.class_key,
                "assignment_scope": "expected_slot_before_extra",
            }
            if center_factor is not None:
                debug["center_distance_factor"] = _round(center_factor)
            if iou is not None:
                debug["iou"] = _round(iou)
            candidates.append((score, expected_index, det.index, debug))

    candidates.sort(key=lambda item: item[0], reverse=True)
    assignments: dict[int, tuple[int, dict[str, Any]]] = {}
    used_detections: set[int] = set()
    for _score, expected_index, detection_index, debug in candidates:
        if expected_index in assignments or detection_index in used_detections:
            continue
        assignments[expected_index] = (detection_index, debug)
        used_detections.add(detection_index)
    return assignments


def _collapse_extra_detections(
    detection_items: list[_DetectionSlot],
    projected: list[_ProjectedSlot],
    *,
    detection_by_index: dict[int, _DetectionSlot],
    occupied_detection_indices: set[int],
    expected: list[ExpectedSegment],
) -> list[tuple[_DetectionSlot, dict[str, Any], ExpectedSegment | None]]:
    expected_by_class: dict[str, ExpectedSegment] = {}
    for item in expected:
        expected_by_class.setdefault(item.class_key, item)

    clusters: list[tuple[_DetectionSlot, list[int]]] = []

    for det in detection_items:
        if det.index in occupied_detection_indices:
            continue
        if _containing_slot(det, projected) is not None:
            continue
        if _is_duplicate_of_occupied_detection(
            det,
            occupied_detection_indices,
            detection_by_index=detection_by_index,
        ):
            continue
        if float(det.detection.confidence or 0.0) < _MIN_EXTRA_CONFIDENCE:
            continue

        replacement_index: int | None = None
        for index, (current_det, _suppressed) in enumerate(clusters):
            if _detections_are_same_physical_object(det, current_det):
                replacement_index = index
                break

        if replacement_index is None:
            clusters.append((det, []))
            continue

        current_det, suppressed = clusters[replacement_index]
        if _extra_detection_sort_key(det) > _extra_detection_sort_key(current_det):
            suppressed.append(current_det.index)
            clusters[replacement_index] = (det, suppressed)
        else:
            suppressed.append(det.index)

    collapsed: list[tuple[_DetectionSlot, dict[str, Any], ExpectedSegment | None]] = []
    for det, suppressed in clusters:
        reason_code = "extra_detection_outside_all_expected_slots"
        debug = {
            "reason": _reason_message(reason_code),
            "reason_code": reason_code,
            "raw_reason": "outside_projected_slots",
            "dedupe_scope": "physical_detection_cluster",
            "duplicate_policy": "overlap_or_containment_only",
            "duplicate_iou_threshold": _DUPLICATE_DETECTION_IOU,
            "duplicate_containment_threshold": _DUPLICATE_DETECTION_CONTAINMENT,
            "extra_min_confidence": _MIN_EXTRA_CONFIDENCE,
            "duplicate_count": len(suppressed) + 1,
            "suppressed_detection_indices": suppressed,
        }
        collapsed.append((det, debug, expected_by_class.get(det.detection.class_key)))

    collapsed.sort(key=lambda item: _extra_detection_sort_key(item[0]), reverse=True)
    return collapsed


def _dedupe_detection_items(
    detection_items: list[_DetectionSlot],
) -> list[_DetectionSlot]:
    if len(detection_items) <= 1:
        return detection_items

    kept: list[_DetectionSlot] = []
    suppressed: set[int] = set()
    ordered = sorted(detection_items, key=_extra_detection_sort_key, reverse=True)

    for det in ordered:
        if det.index in suppressed:
            continue
        if any(
            _detections_are_same_physical_object(det, existing) for existing in kept
        ):
            suppressed.add(det.index)
            continue
        kept.append(det)

    kept.sort(key=lambda item: item.index)
    return kept


def _detections_are_same_physical_object(a: _DetectionSlot, b: _DetectionSlot) -> bool:
    if a.bbox is None or b.bbox is None:
        return False
    if a.detection.class_key != b.detection.class_key:
        return False

    iou = _bbox_iou(a.bbox, b.bbox)
    if iou >= _DUPLICATE_DETECTION_IOU:
        return True

    containment = _bbox_max_containment(a.bbox, b.bbox)
    return containment >= _DUPLICATE_DETECTION_CONTAINMENT


def _bbox_inner_containment(inner: BBox, outer: BBox) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    inner_area = _bbox_area(inner)
    if inner_area <= 0.0:
        return 0.0
    return float(inter_area / inner_area)


def _bbox_max_containment(a: BBox, b: BBox) -> float:
    return max(_bbox_inner_containment(a, b), _bbox_inner_containment(b, a))


def _extra_detection_sort_key(det: _DetectionSlot) -> tuple[float, float]:
    return (float(det.detection.confidence or 0.0), _bbox_area(det.bbox))


def _is_duplicate_of_occupied_detection(
    det: _DetectionSlot,
    occupied_detection_indices: set[int],
    *,
    detection_by_index: dict[int, _DetectionSlot],
) -> bool:
    if det.bbox is None:
        return False
    for occupied_index in occupied_detection_indices:
        other = detection_by_index.get(occupied_index)
        if other is None or other.bbox is None:
            continue
        if other.detection.class_key != det.detection.class_key:
            continue
        if _detections_are_same_physical_object(det, other):
            return True
    return False


def _bbox_area(bbox: BBox | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


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

    matches: list[SegmentMatch] = []
    for class_key in sorted(set(expected_by_class) | set(detections_by_class)):
        expected_items = expected_by_class.get(class_key, [])
        detected_items = detections_by_class.get(class_key, [])
        matched_count = min(len(expected_items), len(detected_items))
        fallback_name = expected_items[0].name if expected_items else "Лишнее"
        fallback_hue = expected_items[0].hue if expected_items else None

        for index in range(matched_count):
            expected_item = expected_items[index]
            detection = detected_items[index]
            matches.append(
                _expected_match(
                    expected_item,
                    status="ok",
                    confidence=detection.confidence,
                    detected_polygon=_detection_polygon(detection),
                    detected_bbox=detection.bbox,
                )
            )

        for expected_item in expected_items[matched_count:]:
            matches.append(_expected_match(expected_item, status="missing"))

        for detection in detected_items[matched_count:]:
            matches.append(
                _detection_match(
                    detection,
                    name=fallback_name,
                    hue=fallback_hue,
                    status="extra",
                    detected_polygon=_detection_polygon(detection),
                )
            )

    return matches


def summarize(matches: list[SegmentMatch]) -> tuple[int, int, list[str]]:
    expected_matches = [match for match in matches if match.annotation_id is not None]
    total = len(expected_matches)
    matched = sum(1 for match in expected_matches if match.status == "ok")
    missing_names = [
        match.name
        for match in expected_matches
        if match.status in {"missing", "unmatched"}
    ]
    return total, matched, missing_names


def all_ok(matches: list[SegmentMatch], *, expected_total: int | None = None) -> bool:
    expected_matches = [match for match in matches if match.annotation_id is not None]
    total = expected_total if expected_total is not None else len(expected_matches)
    if len(expected_matches) != total:
        return False
    if any(match.status != "ok" for match in expected_matches):
        return False
    return not any(match.status in {"extra", "unmatched"} for match in matches)


class _ProjectedSlot:
    __slots__ = ("index", "item", "polygon", "bbox", "debug")

    def __init__(
        self,
        *,
        index: int,
        item: ExpectedSegment,
        polygon: list[list[float]],
        bbox: BBox,
        debug: dict[str, Any] | None = None,
    ) -> None:
        self.index = index
        self.item = item
        self.polygon = polygon
        self.bbox = bbox
        self.debug = debug


class _DetectionSlot:
    __slots__ = ("index", "detection", "polygon", "bbox")

    def __init__(
        self, *, index: int, detection: YoloDetection, polygon: list[list[float]] | None
    ) -> None:
        self.index = index
        self.detection = detection
        self.polygon = polygon
        self.bbox = (
            _bbox_from_polygon(polygon) if polygon else _bbox_from_detection(detection)
        )


def _expected_match(
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


def _detection_match(
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


def _reason_message(reason_code: str | None) -> str:
    messages = {
        "projected_polygon_outside_frame": "Эталонная зона частично или полностью вне кадра",
        "scene_pose_unconfirmed": "Сцена не подтверждена, зона не рисуется как уверенная",
        "no_detection_in_projected_slot": "YOLO не нашёл объект в эталонной зоне",
        "same_class_detection_near_projected_slot_but_not_matched": "Найден объект этого класса рядом с эталонной зоной, но геометрия слабая",
        "same_class_detection_exists_but_expected_slot_not_confirmed": "Найден объект этого класса, но слот не подтверждён геометрией",
        "different_class_detection_inside_projected_slot": "В эталонной зоне найден объект другого класса",
        "multiple_same_class_detections_inside_projected_slot": "В эталонной зоне найдено несколько объектов этого класса",
        "extra_detection_outside_all_expected_slots": "YOLO нашёл объект вне всех эталонных зон",
        "invalid_projected_polygon": "Не удалось построить корректный полигон зоны",
    }
    return messages.get(str(reason_code or ""), str(reason_code or "unknown"))


def _missing_debug(
    *,
    projection: str,
    safety: str,
    reason: str,
    reason_code: str,
) -> dict[str, Any]:
    return {
        "missing_polygon_projection": projection,
        "projection": projection,
        "missing_polygon_projection_safety": safety,
        "reason": _reason_message(reason_code or reason),
        "reason_code": reason_code,
        "raw_reason": reason,
    }
def _with_projection_debug(
    debug: dict[str, Any],
    *,
    projection_debug: dict[str, Any] | None,
) -> dict[str, Any]:
    updated = dict(debug)
    if projection_debug:
        updated["projection"] = dict(projection_debug)
    return updated


def _resolve_homography(
    homography: np.ndarray | None,
    projection_data: LocalProjectionData | None,
) -> np.ndarray | None:
    if homography is not None:
        return homography
    if projection_data is not None:
        return projection_data.global_homography
    return None


def _resolve_frame_size(
    frame_size: tuple[int, int] | None,
    projection_data: LocalProjectionData | None,
) -> tuple[int, int] | None:
    if frame_size is not None:
        return frame_size
    if projection_data is not None:
        return projection_data.frame_size
    return None


def _project_expected_polygon(
    item: ExpectedSegment,
    homography: np.ndarray | None,
    *,
    frame_size: tuple[int, int] | None,
) -> tuple[list[list[float]] | None, str | None, dict[str, Any]]:
    projection_debug: dict[str, Any] = {}

    if homography is None:
        projection_debug["projection_source"] = "none"
        projection_debug["reason_code"] = "scene_pose_unconfirmed"
        return None, "scene_pose_unconfirmed", projection_debug

    try:
        projected = project_polygon(item.reference_polygon, homography)
    except cv2.error:
        projection_debug["projection_source"] = "failed"
        projection_debug["reason_code"] = "projection_failed"
        return None, "projection_failed", projection_debug

    if len(projected) < _MIN_POLYGON_POINTS or not _polygon_is_finite(projected):
        projection_debug["projection_source"] = "failed"
        projection_debug["reason_code"] = "invalid_projected_polygon"
        return None, "invalid_projected_polygon", projection_debug

    bbox = _bbox_from_polygon(projected)
    if bbox is None:
        projection_debug["projection_source"] = "failed"
        projection_debug["reason_code"] = "invalid_projected_bbox"
        return None, "invalid_projected_bbox", projection_debug

    reason: str | None = None
    if frame_size is not None and not _bbox_visible(bbox, frame_size):
        reason = "projected_polygon_outside_frame"

    projection_debug["projection_source"] = "global_direct"
    return _clean_polygon(projected), reason, projection_debug

def _detection_polygon(detection: YoloDetection) -> list[list[float]] | None:
    if detection.polygon and len(detection.polygon) >= _MIN_POLYGON_POINTS:
        return _clean_polygon(detection.polygon)
    bbox = _bbox_from_detection(detection)
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _score_slot_detection(
    slot: _ProjectedSlot, det: _DetectionSlot
) -> tuple[float, float, float, bool]:
    if det.bbox is None:
        return -1.0, 0.0, 999.0, False
    iou = _bbox_iou(slot.bbox, det.bbox)
    center_factor = _center_distance_factor(slot.bbox, det.bbox)
    center_inside = _point_in_bbox(_bbox_center(det.bbox), slot.bbox)
    score = (
        iou
        - 0.18 * center_factor
        + (0.20 if center_inside else 0.0)
        + 0.001 * float(det.detection.confidence or 0.0)
    )
    return score, iou, center_factor, center_inside


def _candidate_is_acceptable(
    *,
    iou: float,
    center_factor: float,
    center_inside: bool,
    class_matches: bool,
    confidence: float | None,
) -> tuple[bool, str]:
    if iou >= _MIN_MATCH_IOU:
        return True, "projected_slot_iou"

    if center_inside and center_factor <= _MAX_CENTER_DISTANCE_FACTOR:
        return True, "projected_slot_center_inside"

    if (
        class_matches
        and center_inside
        and float(confidence or 0.0) >= _MIN_EXPECTED_SLOT_CONFIDENCE
        and (
            iou >= _RELAXED_SAME_CLASS_MIN_IOU
            or center_factor <= _RELAXED_SAME_CLASS_CENTER_FACTOR
        )
    ):
        return True, "projected_slot_relaxed_same_class_inside"

    return False, "rejected_geometry"


def _containing_slot(
    det: _DetectionSlot, slots: list[_ProjectedSlot]
) -> _ProjectedSlot | None:
    if det.bbox is None:
        return None
    center = _bbox_center(det.bbox)
    best: tuple[float, _ProjectedSlot] | None = None
    for slot in slots:
        if not _point_in_bbox(center, slot.bbox):
            continue
        iou = _bbox_iou(slot.bbox, det.bbox)
        if best is None or iou > best[0]:
            best = (iou, slot)
    return best[1] if best is not None else None


def _bbox_from_detection(detection: YoloDetection) -> BBox | None:
    bbox = detection.bbox or {}
    candidates = [
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ]
    for keys in candidates:
        if all(key in bbox for key in keys):
            return _normalize_bbox(tuple(float(bbox[key]) for key in keys))
    if all(key in bbox for key in ("x", "y", "width", "height")):
        x = float(bbox["x"])
        y = float(bbox["y"])
        return _normalize_bbox(
            (x, y, x + float(bbox["width"]), y + float(bbox["height"]))
        )
    return None


def _bbox_from_polygon(polygon: list[list[float]] | None) -> BBox | None:
    if polygon is None or len(polygon) < _MIN_POLYGON_POINTS:
        return None
    arr = np.asarray(polygon, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] < 2 or not np.isfinite(arr[:, :2]).all():
        return None
    x1 = float(np.min(arr[:, 0]))
    y1 = float(np.min(arr[:, 1]))
    x2 = float(np.max(arr[:, 0]))
    y2 = float(np.max(arr[:, 1]))
    return _normalize_bbox((x1, y1, x2, y2))


def _normalize_bbox(values: tuple[float, float, float, float]) -> BBox | None:
    x1, y1, x2, y2 = values
    if not all(np.isfinite(value) for value in values):
        return None
    left, right = sorted((x1, x2))
    top, bottom = sorted((y1, y2))
    if right - left <= 1e-6 or bottom - top <= 1e-6:
        return None
    return left, top, right, bottom


def _bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def _center_distance_factor(a: BBox, b: BBox) -> float:
    ac = _bbox_center(a)
    bc = _bbox_center(b)
    diag = max(_bbox_diag(a), 1.0)
    return float(np.hypot(ac[0] - bc[0], ac[1] - bc[1]) / diag)


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return (x1 + x2) * 0.5, (y1 + y2) * 0.5


def _bbox_diag(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1))


def _point_in_bbox(point: tuple[float, float], bbox: BBox) -> bool:
    x, y = point
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def _bbox_visible(bbox: BBox, frame_size: tuple[int, int]) -> bool:
    height, width = frame_size
    frame = (0.0, 0.0, float(width), float(height))
    visible_area = _intersection_area(bbox, frame)
    bbox_area = max(1.0, (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]))
    return visible_area / bbox_area >= _MIN_VISIBLE_FRACTION


def _intersection_area(a: BBox, b: BBox) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)


def _polygon_is_finite(polygon: list[list[float]]) -> bool:
    arr = np.asarray(polygon, dtype=np.float64)
    return bool(
        arr.ndim == 2
        and arr.shape[0] >= _MIN_POLYGON_POINTS
        and np.isfinite(arr[:, :2]).all()
    )


def _clean_polygon(polygon: list[list[float]]) -> list[list[float]]:
    return [[float(point[0]), float(point[1])] for point in polygon if len(point) >= 2]


def _round(value: float | None, digits: int = 4) -> float | None:
    if value is None or not np.isfinite(value):
        return None
    return round(float(value), digits)
