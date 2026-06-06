from __future__ import annotations

from typing import Any

import numpy as np
from app.config import settings
from modules.core.standards.reference_constants import IOU_MATCH_THRESHOLD
from modules.yolo.inspection.domain.matcher_debug import (
    _round_debug,
    _slot_debug_payload,
)
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area,
    bbox_center,
    bbox_containment,
    bbox_diag,
    bbox_iou,
    polygon_iou,
)
from modules.yolo.inspection.domain.matcher_runtime_fusion import (
    try_runtime_yolo_anchor_candidate,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    DetectionCandidate,
    ExpectedSlot,
    ProjectedExpected,
)
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds

_MIN_MATCH_SCORE = 0.42
_MAX_CENTER_DISTANCE_FACTOR = 0.85
_MIN_AREA_RATIO = 0.20
_MAX_AREA_RATIO = 5.00
_MAX_OPTIMAL_ASSIGNMENT_CANDIDATES = 72
_MAX_OPTIMAL_ASSIGNMENT_ITEMS = 18
_DUPLICATE_MATCHED_BBOX_IOU = 0.55
_DUPLICATE_MATCHED_CONTAINMENT = 0.80

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

        duplicate_iou = bbox_iou(detection.bbox, occupied_detection.bbox)
        containment = max(
            bbox_containment(detection.bbox, occupied_detection.bbox),
            bbox_containment(occupied_detection.bbox, detection.bbox),
        )

        if (
            duplicate_iou < _DUPLICATE_MATCHED_BBOX_IOU
            and containment < _DUPLICATE_MATCHED_CONTAINMENT
        ):
            continue

        return {
            "reason": "duplicate_of_accepted_detection",
            "accepted_detection_index": occupied_index,
            "accepted_class_key": occupied_detection.detection.class_key,
            "bbox_iou": _round_debug(duplicate_iou),
            "containment": _round_debug(containment),
        }

    return None
