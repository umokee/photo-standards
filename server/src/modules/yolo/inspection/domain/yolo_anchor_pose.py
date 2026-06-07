from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from modules.yolo.inspection.domain.alignment import AlignmentStatus, FrameAlignment
from modules.yolo.inspection.domain.types import ExpectedSegment, YoloDetection


@dataclass(slots=True)
class YoloAnchorPoseDebug:
    attempted: bool = False
    accepted: bool = False
    reject_reason: str | None = None
    candidate_pair_count: int = 0
    anchor_count: int = 0
    inlier_anchor_count: int = 0
    raw_point_count: int = 0
    inlier_point_count: int = 0
    point_inlier_ratio: float | None = None
    median_center_error: float | None = None
    p90_center_error: float | None = None
    anchor_spread_score: float | None = None
    homography_condition: float | None = None

    def to_payload(self) -> dict[str, Any]:
        return {
            "attempted": self.attempted,
            "accepted": self.accepted,
            "reject_reason": self.reject_reason,
            "candidate_pair_count": self.candidate_pair_count,
            "anchor_count": self.anchor_count,
            "inlier_anchor_count": self.inlier_anchor_count,
            "raw_point_count": self.raw_point_count,
            "inlier_point_count": self.inlier_point_count,
            "point_inlier_ratio": self.point_inlier_ratio,
            "median_center_error_px": self.median_center_error,
            "p90_center_error_px": self.p90_center_error,
            "anchor_spread_score": self.anchor_spread_score,
            "homography_condition": self.homography_condition,
        }


_MIN_ANCHORS = 2
_MIN_INLIER_ANCHORS = 2
_MIN_POINT_INLIER_RATIO = 0.35
_MAX_MEDIAN_CENTER_ERROR_PX = 18.0
_MAX_P90_CENTER_ERROR_PX = 36.0
_MIN_ANCHOR_SPREAD_SCORE = 0.10
_MAX_HOMOGRAPHY_CONDITION = 1.0e8
_RANSAC_REPROJ_THRESHOLD_PX = 7.0


def estimate_yolo_anchor_alignment(
    expected: list[ExpectedSegment],
    detections: list[YoloDetection],
    *,
    frame_size: tuple[int, int] | None = None,
    debug_out: dict[str, Any] | None = None,
) -> FrameAlignment | None:
    """Estimate scene pose from visible YOLO objects.

    This is intentionally a pose-estimation step, not a missing-object matcher.
    If YOLO found at least a few visible objects, their detected boxes are used
    as anchors for projecting the reference layout. If YOLO found nothing, this
    function returns None and the caller keeps the normal feature/slot fallback.
    """

    debug = YoloAnchorPoseDebug(attempted=True)
    if not detections:
        return _reject(debug, "no_detections", debug_out)

    candidate_pairs = _build_candidate_pairs(expected, detections)
    debug.candidate_pair_count = len(candidate_pairs)
    if len(candidate_pairs) < _MIN_ANCHORS:
        return _reject(debug, "too_few_class_matched_anchors", debug_out)

    ref_points, frame_points, pair_indices = _candidate_corner_points(candidate_pairs)
    debug.raw_point_count = int(len(ref_points))
    if len(ref_points) < 8:
        return _reject(debug, "too_few_anchor_points", debug_out)

    homography, mask = cv2.findHomography(
        ref_points.astype(np.float32),
        frame_points.astype(np.float32),
        cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_THRESHOLD_PX,
    )
    if homography is None or mask is None:
        return _reject(debug, "homography_failed", debug_out)

    homography = np.asarray(homography, dtype=np.float64)
    condition = _homography_condition(homography)
    debug.homography_condition = condition
    if condition is None or condition > _MAX_HOMOGRAPHY_CONDITION:
        return _reject(debug, "homography_condition_bad", debug_out)

    inlier_mask = mask.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    debug.inlier_point_count = inlier_count
    inlier_ratio = inlier_count / max(1, len(inlier_mask))
    debug.point_inlier_ratio = float(inlier_ratio)
    if inlier_count < 8 or inlier_ratio < _MIN_POINT_INLIER_RATIO:
        return _reject(debug, "too_few_ransac_inliers", debug_out)

    center_errors, inlier_anchor_count = _anchor_center_errors(
        candidate_pairs,
        homography,
        point_inlier_mask=inlier_mask,
        pair_indices=pair_indices,
    )
    debug.anchor_count = len(center_errors)
    debug.inlier_anchor_count = inlier_anchor_count
    if inlier_anchor_count < _MIN_INLIER_ANCHORS:
        return _reject(debug, "too_few_inlier_anchors", debug_out)

    spread_score = _anchor_spread_score(
        candidate_pairs,
        point_inlier_mask=inlier_mask,
        pair_indices=pair_indices,
        frame_size=frame_size,
    )
    debug.anchor_spread_score = spread_score
    if spread_score is not None and spread_score < _MIN_ANCHOR_SPREAD_SCORE:
        return _reject(debug, "anchor_spread_too_low", debug_out)

    if not center_errors:
        return _reject(debug, "no_center_errors", debug_out)

    median_error = float(np.median(center_errors))
    p90_error = float(np.percentile(center_errors, 90))
    debug.median_center_error = median_error
    debug.p90_center_error = p90_error

    if median_error > _MAX_MEDIAN_CENTER_ERROR_PX:
        return _reject(debug, "median_center_error_too_high", debug_out)
    if p90_error > _MAX_P90_CENTER_ERROR_PX:
        return _reject(debug, "p90_center_error_too_high", debug_out)

    debug.accepted = True
    _write_debug(debug, debug_out)
    return FrameAlignment(
        status=AlignmentStatus.SUCCESS,
        homography=homography,
        raw_match_count=int(len(ref_points)),
        inlier_count=int(inlier_count),
        median_error=median_error,
        reference_inliers=ref_points[inlier_mask].astype(np.float32),
        frame_inliers=frame_points[inlier_mask].astype(np.float32),
        reference_matches=ref_points.astype(np.float32),
        frame_matches=frame_points.astype(np.float32),
        method="yolo_anchor_pose",
        stage="yolo_anchor_pose",
        reason="pose_estimated_from_visible_yolo_objects",
        reference_feature_count=len(expected),
        frame_feature_count=len(detections),
        frame_size=frame_size,
        extra_debug={"yolo_anchor_pose": debug.to_payload()},
    )


def _reject(
    debug: YoloAnchorPoseDebug,
    reason: str,
    debug_out: dict[str, Any] | None,
) -> None:
    debug.reject_reason = reason
    _write_debug(debug, debug_out)
    return None


def _write_debug(
    debug: YoloAnchorPoseDebug,
    debug_out: dict[str, Any] | None,
) -> None:
    if debug_out is not None:
        debug_out["yolo_anchor_pose"] = debug.to_payload()


def _build_candidate_pairs(
    expected: list[ExpectedSegment],
    detections: list[YoloDetection],
) -> list[tuple[ExpectedSegment, YoloDetection]]:
    pairs: list[tuple[ExpectedSegment, YoloDetection]] = []
    expected_by_class: dict[str, list[ExpectedSegment]] = {}
    for item in expected:
        expected_by_class.setdefault(item.class_key, []).append(item)

    detections_by_class: dict[str, list[YoloDetection]] = {}
    for detection in detections:
        detections_by_class.setdefault(detection.class_key, []).append(detection)

    for class_key, class_detections in detections_by_class.items():
        class_expected = expected_by_class.get(class_key) or []
        if not class_expected:
            continue

        # Include all same-class alternatives. RANSAC is expected to reject the
        # wrong alternatives when repeated classes exist.
        for detection in class_detections:
            for item in class_expected:
                if _reference_bbox(item) is None or _detection_bbox(detection) is None:
                    continue
                pairs.append((item, detection))

    # Keep the point set bounded for repeated classes.
    if len(pairs) > 160:
        pairs.sort(key=lambda pair: pair[1].confidence, reverse=True)
        pairs = pairs[:160]
    return pairs


def _candidate_corner_points(
    candidate_pairs: list[tuple[ExpectedSegment, YoloDetection]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ref_points: list[tuple[float, float]] = []
    frame_points: list[tuple[float, float]] = []
    pair_indices: list[int] = []

    for pair_index, (expected, detection) in enumerate(candidate_pairs):
        ref_bbox = _reference_bbox(expected)
        frame_bbox = _detection_bbox(detection)
        if ref_bbox is None or frame_bbox is None:
            continue

        for ref_point, frame_point in zip(
            _bbox_corners(ref_bbox),
            _bbox_corners(frame_bbox),
            strict=True,
        ):
            ref_points.append(ref_point)
            frame_points.append(frame_point)
            pair_indices.append(pair_index)

    return (
        np.asarray(ref_points, dtype=np.float32),
        np.asarray(frame_points, dtype=np.float32),
        np.asarray(pair_indices, dtype=np.int32),
    )


def _anchor_center_errors(
    candidate_pairs: list[tuple[ExpectedSegment, YoloDetection]],
    homography: np.ndarray,
    *,
    point_inlier_mask: np.ndarray,
    pair_indices: np.ndarray,
) -> tuple[list[float], int]:
    errors: list[float] = []
    inlier_anchor_count = 0

    for pair_index, (expected, detection) in enumerate(candidate_pairs):
        ref_bbox = _reference_bbox(expected)
        frame_bbox = _detection_bbox(detection)
        if ref_bbox is None or frame_bbox is None:
            continue

        pair_point_mask = pair_indices == pair_index
        if int(np.count_nonzero(point_inlier_mask & pair_point_mask)) >= 3:
            inlier_anchor_count += 1

        projected = _project_point(_bbox_center(ref_bbox), homography)
        if projected is None:
            continue
        actual = _bbox_center(frame_bbox)
        errors.append(float(np.hypot(projected[0] - actual[0], projected[1] - actual[1])))

    return errors, inlier_anchor_count


def _anchor_spread_score(
    candidate_pairs: list[tuple[ExpectedSegment, YoloDetection]],
    *,
    point_inlier_mask: np.ndarray,
    pair_indices: np.ndarray,
    frame_size: tuple[int, int] | None,
) -> float | None:
    if frame_size is None:
        return None

    frame_width, frame_height = frame_size
    frame_diag = float(np.hypot(max(frame_width, 1), max(frame_height, 1)))
    if frame_diag <= 0:
        return None

    centers: list[tuple[float, float]] = []
    for pair_index, (_, detection) in enumerate(candidate_pairs):
        frame_bbox = _detection_bbox(detection)
        if frame_bbox is None:
            continue

        pair_point_mask = pair_indices == pair_index
        if int(np.count_nonzero(point_inlier_mask & pair_point_mask)) >= 3:
            centers.append(_bbox_center(frame_bbox))

    if len(centers) < 2:
        return 0.0

    pts = np.asarray(centers, dtype=np.float32)
    min_xy = np.min(pts, axis=0)
    max_xy = np.max(pts, axis=0)
    spread_diag = float(np.hypot(*(max_xy - min_xy)))
    return spread_diag / frame_diag


def _homography_condition(homography: np.ndarray) -> float | None:
    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        return None
    scale = float(homography[2, 2])
    if abs(scale) < 1e-9:
        return None
    normalized = homography / scale
    try:
        condition = float(np.linalg.cond(normalized))
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(condition):
        return None
    return condition


def _reference_bbox(item: ExpectedSegment) -> tuple[float, float, float, float] | None:
    points = np.asarray(item.reference_polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    min_xy = np.min(points[:, :2], axis=0)
    max_xy = np.max(points[:, :2], axis=0)
    return (float(min_xy[0]), float(min_xy[1]), float(max_xy[0]), float(max_xy[1]))


def _detection_bbox(
    detection: YoloDetection,
) -> tuple[float, float, float, float] | None:
    bbox = detection.bbox
    try:
        x1 = float(bbox["x1"])
        y1 = float(bbox["y1"])
        x2 = float(bbox["x2"])
        y2 = float(bbox["y2"])
    except (KeyError, TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return (x1, y1, x2, y2)


def _bbox_corners(
    bbox: tuple[float, float, float, float],
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]:
    x1, y1, x2, y2 = bbox
    return ((x1, y1), (x2, y1), (x2, y2), (x1, y2))


def _bbox_center(bbox: tuple[float, float, float, float]) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _project_point(
    point: tuple[float, float],
    homography: np.ndarray,
) -> tuple[float, float] | None:
    source = np.asarray([[[point[0], point[1]]]], dtype=np.float32)
    projected = cv2.perspectiveTransform(source, homography)
    if projected is None or projected.shape != (1, 1, 2):
        return None
    x = float(projected[0, 0, 0])
    y = float(projected[0, 0, 1])
    if not np.isfinite([x, y]).all():
        return None
    return x, y
