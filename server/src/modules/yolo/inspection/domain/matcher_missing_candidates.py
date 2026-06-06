from __future__ import annotations

from typing import Any

import numpy as np
from modules.yolo.inspection.domain.alignment import LocalProjectionData
from modules.yolo.inspection.domain.matcher_geometry import (
    bbox_area_similarity,
    bbox_center_distance_factor,
    bbox_containment,
    bbox_from_polygon,
    bbox_iou,
    is_visible_in_frame,
)
from modules.yolo.inspection.domain.matcher_structs import (
    BBox,
    ExpectedSlot,
    ProjectedExpected,
)


_CANDIDATE_REGISTRY_LIMIT = 16
_CANDIDATE_AGREEMENT_SOURCE_LIMIT = 8
_CANDIDATE_AGREEMENT_MIN_IOU = 0.55
_CANDIDATE_AGREEMENT_MIN_AREA_SCORE = 0.70
_CANDIDATE_AGREEMENT_MAX_CENTER_FACTOR = 0.55



def max_overlap_with_other_expected(
    expected_item: ProjectedExpected,
    *,
    bbox: BBox,
    all_expected: list[ProjectedExpected] | None,
) -> float:
    if not all_expected or len(all_expected) <= 1:
        return 0.0

    max_overlap = 0.0
    for other in all_expected:
        if other.index == expected_item.index:
            continue
        overlap = max(
            bbox_iou(bbox, other.bbox),
            bbox_containment(bbox, other.bbox),
            bbox_containment(other.bbox, bbox),
        )
        max_overlap = max(max_overlap, float(overlap))
    return float(max_overlap)


def reference_area_score_for_expected(
    expected_item: ProjectedExpected,
    *,
    slot: ExpectedSlot | None,
    bbox: BBox,
) -> float | None:
    reference_bbox = slot.reference_bbox if slot is not None else None
    if reference_bbox is None:
        reference_bbox = bbox_from_polygon(expected_item.item.reference_polygon)
    if reference_bbox is None:
        return None
    return bbox_area_similarity(bbox, reference_bbox)


def _round_optional_debug(value: float | None) -> float | None:
    if value is None:
        return None
    return _round_debug(value)


def append_missing_projection_candidate(
    debug: dict[str, Any],
    *,
    name: str,
    expected_item: ProjectedExpected,
    polygon: list[list[float]] | None,
    bbox: BBox | None,
    slot: ExpectedSlot | None,
    projection_data: LocalProjectionData | None,
    all_expected: list[ProjectedExpected] | None,
    selected: bool = False,
    hidden_release: bool = False,
    reason: str | None = None,
) -> None:
    registry = debug.setdefault("missing_polygon_candidate_registry", [])
    if not isinstance(registry, list) or len(registry) >= _CANDIDATE_REGISTRY_LIMIT:
        return

    available = polygon is not None and bbox is not None and len(polygon) >= 3
    item: dict[str, Any] = {
        "name": name,
        "available": bool(available),
        "selected": bool(selected),
        "hidden_release": bool(hidden_release),
    }
    if reason:
        item["reason"] = reason

    if not available or bbox is None:
        registry.append(item)
        return

    reference_area_score = reference_area_score_for_expected(
        expected_item,
        slot=slot,
        bbox=bbox,
    )
    visible = True
    if projection_data is not None and projection_data.frame_size is not None:
        visible = is_visible_in_frame(
            bbox,
            projection_data.frame_size,
            min_visible_fraction=0.01,
        )

    item.update(
        {
            "bbox": _bbox_debug(bbox),
            "area_score_vs_expected_slot": _round_debug(
                bbox_area_similarity(bbox, expected_item.bbox)
            ),
            "center_factor_vs_expected_slot": _round_debug(
                bbox_center_distance_factor(bbox, expected_item.bbox)
            ),
            "reference_area_score": _round_optional_debug(reference_area_score),
            "max_other_expected_overlap": _round_debug(
                max_overlap_with_other_expected(
                    expected_item,
                    bbox=bbox,
                    all_expected=all_expected,
                )
            ),
            "visible_in_frame": bool(visible),
        }
    )
    registry.append(item)


def finalize_missing_candidate_agreement_debug(
    debug: dict[str, Any],
    *,
    final_projection: str | None = None,
) -> dict[str, Any]:
    raw_registry = debug.get("missing_polygon_candidate_registry")
    if not isinstance(raw_registry, list):
        debug.update(
            {
                "missing_polygon_candidate_agreement_enabled": True,
                "missing_polygon_candidate_agreement_available_count": 0,
                "missing_polygon_candidate_agreement_level": "no_registry",
                "missing_polygon_candidate_confidence": "unknown",
            }
        )
        return debug

    candidates: list[tuple[dict[str, Any], BBox]] = []
    selected_item: dict[str, Any] | None = None
    selected_bbox: BBox | None = None
    for raw_item in raw_registry:
        if not isinstance(raw_item, dict):
            continue
        bbox = _candidate_debug_bbox(raw_item)
        if bbox is None:
            continue
        candidates.append((raw_item, bbox))
        if bool(raw_item.get("selected")):
            selected_item = raw_item
            selected_bbox = bbox

    projection = final_projection or _missing_debug_projection_name(debug)
    if selected_item is None and candidates:
        selected_item, selected_bbox = candidates[-1]

    if selected_item is None or selected_bbox is None:
        debug.update(
            {
                "missing_polygon_candidate_agreement_enabled": True,
                "missing_polygon_candidate_agreement_available_count": len(candidates),
                "missing_polygon_candidate_agreement_level": "no_selected_candidate",
                "missing_polygon_candidate_confidence": "unknown",
            }
        )
        return debug

    selected_name = str(selected_item.get("name") or "")
    selected_base = _candidate_base_name(selected_name)
    comparisons = 0
    agreement_count = 0
    best_iou = 0.0
    best_area_score = 0.0
    min_center_factor: float | None = None
    closest_source: str | None = None
    agreeing_sources: list[str] = []

    for item, bbox in candidates:
        other_name = str(item.get("name") or "")
        other_base = _candidate_base_name(other_name)
        if item is selected_item or other_base == selected_base:
            continue
        comparisons += 1
        agrees, iou, area_score, center_factor = _candidate_agrees_with_selected(
            selected_bbox,
            bbox,
        )
        if iou > best_iou:
            best_iou = iou
        if area_score > best_area_score:
            best_area_score = area_score
        if min_center_factor is None or center_factor < min_center_factor:
            min_center_factor = center_factor
            closest_source = other_name
        if agrees:
            agreement_count += 1
            agreeing_sources.append(other_name)

    if comparisons <= 0:
        agreement_level = "isolated"
    elif agreement_count >= 2:
        agreement_level = "strong"
    elif agreement_count == 1:
        agreement_level = "weak"
    else:
        agreement_level = "conflict"

    confidence, recommended_action = _candidate_confidence_and_action(
        projection=projection,
        agreement_level=agreement_level,
    )

    debug.update(
        {
            "missing_polygon_candidate_agreement_enabled": True,
            "missing_polygon_candidate_agreement_available_count": len(candidates),
            "missing_polygon_candidate_agreement_comparison_count": comparisons,
            "missing_polygon_candidate_agreement_selected": selected_name,
            "missing_polygon_candidate_agreement_selected_base": selected_base,
            "missing_polygon_candidate_agreement_count": agreement_count,
            "missing_polygon_candidate_agreement_level": agreement_level,
            "missing_polygon_candidate_agreement_best_iou": _round_debug(best_iou),
            "missing_polygon_candidate_agreement_best_area_score": _round_debug(
                best_area_score
            ),
            "missing_polygon_candidate_agreement_min_center_factor": _round_optional_debug(
                min_center_factor
            ),
            "missing_polygon_candidate_agreement_closest_source": closest_source,
            "missing_polygon_candidate_agreement_sources": agreeing_sources[
                :_CANDIDATE_AGREEMENT_SOURCE_LIMIT
            ],
            "missing_polygon_candidate_confidence": confidence,
            "missing_polygon_candidate_recommended_action": recommended_action,
        }
    )
    return debug


def _candidate_confidence_and_action(
    *,
    projection: str,
    agreement_level: str,
) -> tuple[str, str]:
    trusted_projection = projection in {
        "context_feature_affine_translation_rescue",
        "expected_slot_context_translation_rescue",
        "expected_slot_scene_translation_rescue",
        "expected_slot_anchor_release",
        "expected_slot_local_displacement",
    }
    guarded_projection = projection in {
        "expected_slot",
        "expected_slot_global_fallback",
        "expected_slot_global_fallback_hidden_release",
        "expected_slot_agreement_hidden_release",
    }
    if trusted_projection:
        return "trusted_method", "render_confident_expected_zone"
    if projection == "unsafe_hidden":
        return "hidden", "keep_hidden_or_require_more_evidence"
    if projection == "none":
        return "failed", "no_projection_available"
    if guarded_projection and agreement_level in {"weak", "strong"}:
        return "guarded_unconfirmed", "render_as_unconfirmed_expected_zone"
    if guarded_projection:
        return "weak_unconfirmed", "prefer_uncertain_or_hidden_in_ui"
    return "unknown", "inspect_candidate_manually"


def _candidate_base_name(name: str | None) -> str:
    if not name:
        return ""
    result = str(name)
    if result.startswith("selected:"):
        result = result.removeprefix("selected:")
    if result.startswith("selective_hidden_release:"):
        result = result.removeprefix("selective_hidden_release:")
    return result


def _candidate_debug_bbox(item: dict[str, Any]) -> BBox | None:
    raw = item.get("bbox")
    if not isinstance(raw, dict):
        return None

    raw_x = raw.get("x")
    raw_y = raw.get("y")
    raw_w = raw.get("w")
    raw_h = raw.get("h")
    if raw_x is None or raw_y is None or raw_w is None or raw_h is None:
        return None

    try:
        x = float(raw_x)
        y = float(raw_y)
        w = float(raw_w)
        h = float(raw_h)
    except (TypeError, ValueError):
        return None
    if not np.isfinite([x, y, w, h]).all() or w <= 0.0 or h <= 0.0:
        return None
    return (x, y, x + w, y + h)


def _candidate_agrees_with_selected(
    selected_bbox: BBox,
    other_bbox: BBox,
) -> tuple[bool, float, float, float]:
    iou = bbox_iou(selected_bbox, other_bbox)
    area_score = bbox_area_similarity(selected_bbox, other_bbox)
    center_factor = bbox_center_distance_factor(other_bbox, selected_bbox)
    agrees = bool(
        iou >= _CANDIDATE_AGREEMENT_MIN_IOU
        or (
            area_score >= _CANDIDATE_AGREEMENT_MIN_AREA_SCORE
            and center_factor
            <= _CANDIDATE_AGREEMENT_MAX_CENTER_FACTOR
        )
    )
    return agrees, iou, area_score, center_factor


def _missing_debug_projection_name(debug: dict[str, Any] | None) -> str:
    if not debug:
        return ""
    value = debug.get("missing_polygon_projection") or debug.get("projection")
    return str(value or "")


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


def _round_debug(value: float) -> float:
    return round(float(value), 4)
