from __future__ import annotations

RUNTIME_FUSION_MODE_YOLO_ANCHOR_RESCUE = "yolo_anchor_rescue"
RUNTIME_FUSION_SUMMARY_MODE = "runtime_yolo_anchor_fusion"

YOLO_ANCHOR_STATUS_TRUSTED = "trusted_anchor"
YOLO_ANCHOR_STATUS_REJECTED = "rejected_anchor"
YOLO_ANCHOR_STATUS_NO_ANCHOR = "no_anchor"

APPLIED_GEOMETRY_SOURCE_BASELINE = "baseline_geometry"
APPLIED_GEOMETRY_SOURCE_TRUSTED_YOLO_MASK = "trusted_yolo_mask"
APPLIED_GEOMETRY_SOURCE_TRUSTED_ANCHOR_TRANSFORM = "trusted_anchor_transform"
APPLIED_GEOMETRY_SOURCE_UNRESOLVED_BASELINE_FAILURE = "unresolved_baseline_failure"

RUNTIME_REASON_TRUSTED_YOLO_ANCHOR = "runtime_trusted_yolo_anchor"
RUNTIME_REASON_YOLO_ANCHOR_CANDIDATE_REJECTED = (
    "runtime_yolo_anchor_candidate_rejected"
)
RUNTIME_REASON_NO_TRUSTED_YOLO_ANCHOR = "runtime_no_trusted_yolo_anchor"

RUNTIME_ANCHOR_REJECT_LOW_CONFIDENCE = "runtime_anchor_low_confidence"
RUNTIME_ANCHOR_REJECT_CENTER_OUTSIDE_SLOT = (
    "runtime_anchor_center_outside_expected_slot"
)
RUNTIME_ANCHOR_REJECT_LOW_SLOT_IOU = "runtime_anchor_low_slot_iou"
RUNTIME_ANCHOR_REJECT_LOW_DETECTION_CONTAINMENT = (
    "runtime_anchor_detection_not_contained_in_slot"
)
RUNTIME_ANCHOR_REJECT_LOW_SLOT_COVERAGE = (
    "runtime_anchor_expected_slot_not_covered"
)
RUNTIME_ANCHOR_REJECT_BAD_AREA_RATIO = "runtime_anchor_bad_area_ratio"
RUNTIME_ANCHOR_REJECT_UNKNOWN = "runtime_anchor_rejected_unknown"

NO_ANCHOR_SEMANTICS = (
    "no_anchor means YOLO did not provide trusted geometry help; "
    "it is not a standalone missing decision."
)
NO_ANCHOR_NOTE = (
    "LightGlue expected geometry is kept; YOLO absence is reported as "
    "unconfirmed geometry, not as independent proof of absence."
)

__all__ = [
    "APPLIED_GEOMETRY_SOURCE_BASELINE",
    "APPLIED_GEOMETRY_SOURCE_TRUSTED_ANCHOR_TRANSFORM",
    "APPLIED_GEOMETRY_SOURCE_TRUSTED_YOLO_MASK",
    "APPLIED_GEOMETRY_SOURCE_UNRESOLVED_BASELINE_FAILURE",
    "NO_ANCHOR_NOTE",
    "NO_ANCHOR_SEMANTICS",
    "RUNTIME_ANCHOR_REJECT_BAD_AREA_RATIO",
    "RUNTIME_ANCHOR_REJECT_CENTER_OUTSIDE_SLOT",
    "RUNTIME_ANCHOR_REJECT_LOW_CONFIDENCE",
    "RUNTIME_ANCHOR_REJECT_LOW_DETECTION_CONTAINMENT",
    "RUNTIME_ANCHOR_REJECT_LOW_SLOT_COVERAGE",
    "RUNTIME_ANCHOR_REJECT_LOW_SLOT_IOU",
    "RUNTIME_ANCHOR_REJECT_UNKNOWN",
    "RUNTIME_FUSION_MODE_YOLO_ANCHOR_RESCUE",
    "RUNTIME_FUSION_SUMMARY_MODE",
    "RUNTIME_REASON_NO_TRUSTED_YOLO_ANCHOR",
    "RUNTIME_REASON_TRUSTED_YOLO_ANCHOR",
    "RUNTIME_REASON_YOLO_ANCHOR_CANDIDATE_REJECTED",
    "YOLO_ANCHOR_STATUS_NO_ANCHOR",
    "YOLO_ANCHOR_STATUS_REJECTED",
    "YOLO_ANCHOR_STATUS_TRUSTED",
]
