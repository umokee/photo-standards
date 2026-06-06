from __future__ import annotations

from typing import Any

import numpy as np
from modules.yolo.inspection.domain.matcher_structs import BBox, ExpectedSlot
from modules.yolo.inspection.domain.matcher_thresholds import thresholds as _thresholds


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


def _round_debug(value: float) -> float:
    return round(float(value), 4)
