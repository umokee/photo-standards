from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import cv2
import numpy as np
from modules.core.standards.reference_features import compute_features
from modules.core.standards.reference_runtime import match_feature_arrays

PolygonPoints = list[list[float]]
BBox = tuple[float, float, float, float]

_CROP_SCALE = 2.8
_CROP_MIN_PADDING = 48.0
_CROP_MAX_PADDING = 220.0

_MIN_CROP_SIDE = 48
_MIN_FEATURE_COUNT = 8
_MIN_RAW_MATCHES = 8
_MIN_AFFINE_INLIERS = 5
_MIN_HOMOGRAPHY_INLIERS = 6

_RANSAC_REPROJ_THRESHOLD = 5.0
_MAX_MEDIAN_REPROJ_ERROR = 10.0

_MIN_PROJECTED_AREA_RATIO = 0.05
_MAX_PROJECTED_AREA_RATIO = 20.0

_MAX_CENTER_DISTANCE_FACTOR = 1.15
_MAX_CENTER_DISTANCE_MIN_PX = 40.0


@dataclass(slots=True)
class ObjectLocalRefinement:
    success: bool
    projected_polygon: PolygonPoints | None = None
    method: str | None = None
    raw_match_count: int = 0
    inlier_count: int = 0
    median_error: float | None = None
    reason: str | None = None
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class ObjectLocalRefiner:
    """Crop-to-crop object refiner for etalon-to-frame projection.

    The global homography is used only as a rough search hint.  This refiner
    re-runs SuperPoint/LightGlue on two local crops: the expected reference
    object area and the YOLO candidate area in the current frame.
    """

    reference_frame: np.ndarray
    frame: np.ndarray
    max_side: int = 768
    max_keypoints: int = 1536

    def refine(
        self,
        *,
        reference_polygon: PolygonPoints,
        detection_polygon: PolygonPoints | None,
        detection_bbox: BBox,
    ) -> ObjectLocalRefinement:
        reference_points = _as_polygon_array(reference_polygon)
        if reference_points is None:
            return _failed("invalid_reference_polygon")

        reference_bbox = _bbox_from_points(reference_points)
        if reference_bbox is None:
            return _failed("invalid_reference_bbox")

        frame_bbox = _bbox_from_detection(detection_polygon, detection_bbox)
        if frame_bbox is None:
            return _failed("invalid_detection_bbox")

        reference_crop_box = _expanded_crop_box(
            reference_bbox,
            image_shape=self.reference_frame.shape[:2],
        )
        frame_crop_box = _expanded_crop_box(
            frame_bbox,
            image_shape=self.frame.shape[:2],
        )

        reference_crop = _crop(self.reference_frame, reference_crop_box)
        frame_crop = _crop(self.frame, frame_crop_box)

        if reference_crop is None or frame_crop is None:
            return _failed(
                "empty_crop",
                reference_crop=_crop_debug(reference_crop_box),
                frame_crop=_crop_debug(frame_crop_box),
            )

        try:
            reference_features = compute_features(
                reference_crop,
                max_side=self.max_side,
                max_keypoints=self.max_keypoints,
            )
            frame_features = compute_features(
                frame_crop,
                max_side=self.max_side,
                max_keypoints=self.max_keypoints,
            )
        except Exception as exc:  # noqa: BLE001
            return _failed(
                "feature_extraction_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                reference_crop=_crop_debug(reference_crop_box),
                frame_crop=_crop_debug(frame_crop_box),
            )

        if (
            reference_features.count < _MIN_FEATURE_COUNT
            or frame_features.count < _MIN_FEATURE_COUNT
        ):
            return _failed(
                "not_enough_local_features",
                reference_feature_count=reference_features.count,
                frame_feature_count=frame_features.count,
                reference_crop=_crop_debug(reference_crop_box),
                frame_crop=_crop_debug(frame_crop_box),
            )

        try:
            matched = match_feature_arrays(
                reference_keypoints=reference_features.keypoints.astype(
                    np.float32,
                    copy=False,
                ),
                reference_descriptors=reference_features.descriptors.astype(
                    np.float32,
                    copy=False,
                ),
                reference_size=(
                    reference_features.image_width,
                    reference_features.image_height,
                ),
                frame_keypoints=frame_features.keypoints.astype(np.float32, copy=False),
                frame_descriptors=frame_features.descriptors.astype(
                    np.float32,
                    copy=False,
                ),
                frame_size=(frame_features.image_width, frame_features.image_height),
                max_keypoints=self.max_keypoints,
            )
        except Exception as exc:  # noqa: BLE001
            return _failed(
                "feature_matching_failed",
                error_type=type(exc).__name__,
                error=str(exc),
                reference_feature_count=reference_features.count,
                frame_feature_count=frame_features.count,
                reference_crop=_crop_debug(reference_crop_box),
                frame_crop=_crop_debug(frame_crop_box),
            )

        if matched is None:
            return _failed(
                "no_local_matches",
                reference_feature_count=reference_features.count,
                frame_feature_count=frame_features.count,
                reference_crop=_crop_debug(reference_crop_box),
                frame_crop=_crop_debug(frame_crop_box),
            )

        reference_matches, frame_matches = matched
        reference_matches = np.asarray(reference_matches, dtype=np.float32).reshape(-1, 2)
        frame_matches = np.asarray(frame_matches, dtype=np.float32).reshape(-1, 2)

        raw_match_count = int(len(reference_matches))
        if raw_match_count < _MIN_RAW_MATCHES:
            return _failed(
                "not_enough_local_matches",
                raw_match_count=raw_match_count,
                reference_feature_count=reference_features.count,
                frame_feature_count=frame_features.count,
                reference_crop=_crop_debug(reference_crop_box),
                frame_crop=_crop_debug(frame_crop_box),
            )

        polygon_in_crop = reference_points.copy()
        polygon_in_crop[:, 0] -= reference_crop_box[0]
        polygon_in_crop[:, 1] -= reference_crop_box[1]

        attempts = (
            _try_affine(
                polygon_in_crop=polygon_in_crop,
                reference_matches=reference_matches,
                frame_matches=frame_matches,
            ),
            _try_homography(
                polygon_in_crop=polygon_in_crop,
                reference_matches=reference_matches,
                frame_matches=frame_matches,
            ),
        )

        best_failure: ObjectLocalRefinement | None = None
        for attempt in attempts:
            if not attempt.success or attempt.projected_polygon is None:
                if best_failure is None:
                    best_failure = attempt
                continue

            projected = np.asarray(attempt.projected_polygon, dtype=np.float32)
            projected[:, 0] += frame_crop_box[0]
            projected[:, 1] += frame_crop_box[1]

            if not _projected_polygon_ok(
                source_polygon=reference_points,
                projected_polygon=projected,
                detection_bbox=frame_bbox,
                frame_shape=self.frame.shape[:2],
            ):
                best_failure = _failed(
                    "local_projection_not_near_detection",
                    method=attempt.method,
                    raw_match_count=attempt.raw_match_count,
                    inlier_count=attempt.inlier_count,
                    median_error=attempt.median_error,
                    reference_crop=_crop_debug(reference_crop_box),
                    frame_crop=_crop_debug(frame_crop_box),
                )
                continue

            return ObjectLocalRefinement(
                success=True,
                projected_polygon=_to_points(projected),
                method=attempt.method,
                raw_match_count=attempt.raw_match_count,
                inlier_count=attempt.inlier_count,
                median_error=attempt.median_error,
                debug={
                    "object_local_method": attempt.method,
                    "object_local_raw_matches": attempt.raw_match_count,
                    "object_local_inliers": attempt.inlier_count,
                    "object_local_median_error": _round_debug(attempt.median_error),
                    "object_local_reference_features": reference_features.count,
                    "object_local_frame_features": frame_features.count,
                    "object_local_reference_crop": _crop_debug(reference_crop_box),
                    "object_local_frame_crop": _crop_debug(frame_crop_box),
                },
            )

        if best_failure is not None:
            best_failure.debug.update(
                {
                    "object_local_reference_features": reference_features.count,
                    "object_local_frame_features": frame_features.count,
                    "object_local_reference_crop": _crop_debug(reference_crop_box),
                    "object_local_frame_crop": _crop_debug(frame_crop_box),
                }
            )
            return best_failure

        return _failed(
            "local_transform_failed",
            raw_match_count=raw_match_count,
            reference_feature_count=reference_features.count,
            frame_feature_count=frame_features.count,
            reference_crop=_crop_debug(reference_crop_box),
            frame_crop=_crop_debug(frame_crop_box),
        )


def _try_affine(
    *,
    polygon_in_crop: np.ndarray,
    reference_matches: np.ndarray,
    frame_matches: np.ndarray,
) -> ObjectLocalRefinement:
    affine, inliers = cv2.estimateAffinePartial2D(
        reference_matches,
        frame_matches,
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_THRESHOLD,
        maxIters=1200,
        confidence=0.995,
        refineIters=12,
    )

    if affine is None or inliers is None:
        return _failed("affine_failed", method="affine", raw_match_count=len(reference_matches))

    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < _MIN_AFFINE_INLIERS:
        return _failed(
            "not_enough_affine_inliers",
            method="affine",
            raw_match_count=len(reference_matches),
            inlier_count=inlier_count,
        )

    projected = cv2.transform(
        polygon_in_crop.reshape(-1, 1, 2),
        affine.astype(np.float32),
    ).reshape(-1, 2)

    median_error = _median_transform_error(
        reference_matches[inlier_mask],
        frame_matches[inlier_mask],
        affine,
        perspective=False,
    )
    if median_error is None or median_error > _MAX_MEDIAN_REPROJ_ERROR:
        return _failed(
            "affine_reprojection_error",
            method="affine",
            raw_match_count=len(reference_matches),
            inlier_count=inlier_count,
            median_error=median_error,
        )

    return ObjectLocalRefinement(
        success=True,
        projected_polygon=_to_points(projected),
        method="affine",
        raw_match_count=len(reference_matches),
        inlier_count=inlier_count,
        median_error=median_error,
    )


def _try_homography(
    *,
    polygon_in_crop: np.ndarray,
    reference_matches: np.ndarray,
    frame_matches: np.ndarray,
) -> ObjectLocalRefinement:
    homography, inliers = cv2.findHomography(
        reference_matches.reshape(-1, 1, 2),
        frame_matches.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=_RANSAC_REPROJ_THRESHOLD,
    )

    if homography is None or inliers is None:
        return _failed(
            "homography_failed",
            method="homography",
            raw_match_count=len(reference_matches),
        )

    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < _MIN_HOMOGRAPHY_INLIERS:
        return _failed(
            "not_enough_homography_inliers",
            method="homography",
            raw_match_count=len(reference_matches),
            inlier_count=inlier_count,
        )

    projected = cv2.perspectiveTransform(
        polygon_in_crop.reshape(-1, 1, 2),
        homography.astype(np.float32),
    ).reshape(-1, 2)

    median_error = _median_transform_error(
        reference_matches[inlier_mask],
        frame_matches[inlier_mask],
        homography,
        perspective=True,
    )
    if median_error is None or median_error > _MAX_MEDIAN_REPROJ_ERROR:
        return _failed(
            "homography_reprojection_error",
            method="homography",
            raw_match_count=len(reference_matches),
            inlier_count=inlier_count,
            median_error=median_error,
        )

    return ObjectLocalRefinement(
        success=True,
        projected_polygon=_to_points(projected),
        method="homography",
        raw_match_count=len(reference_matches),
        inlier_count=inlier_count,
        median_error=median_error,
    )


def _projected_polygon_ok(
    *,
    source_polygon: np.ndarray,
    projected_polygon: np.ndarray,
    detection_bbox: BBox,
    frame_shape: tuple[int, int],
) -> bool:
    if not np.isfinite(projected_polygon).all():
        return False

    if _bbox_from_points(projected_polygon) is None:
        return False

    if not _has_visible_bbox(projected_polygon, frame_shape=frame_shape):
        return False

    source_area = abs(float(cv2.contourArea(source_polygon.reshape(-1, 1, 2))))
    projected_area = abs(float(cv2.contourArea(projected_polygon.reshape(-1, 1, 2))))
    if source_area <= 1.0 or projected_area <= 1.0:
        return False

    area_ratio = projected_area / source_area
    if area_ratio < _MIN_PROJECTED_AREA_RATIO or area_ratio > _MAX_PROJECTED_AREA_RATIO:
        return False

    projected_bbox = _bbox_from_points(projected_polygon)
    if projected_bbox is None:
        return False

    projected_center = _bbox_center(projected_bbox)
    detection_center = _bbox_center(detection_bbox)
    distance = float(
        np.hypot(
            projected_center[0] - detection_center[0],
            projected_center[1] - detection_center[1],
        )
    )
    limit = max(
        _MAX_CENTER_DISTANCE_MIN_PX,
        _bbox_diag(detection_bbox) * _MAX_CENTER_DISTANCE_FACTOR,
    )

    return distance <= limit


def _expanded_crop_box(
    bbox: BBox,
    *,
    image_shape: tuple[int, int],
) -> tuple[int, int, int, int]:
    image_h, image_w = image_shape
    x1, y1, x2, y2 = bbox

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    side = max(width, height)

    padding = max(_CROP_MIN_PADDING, side * (_CROP_SCALE - 1.0) * 0.5)
    padding = min(padding, _CROP_MAX_PADDING)

    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    half_w = max(width * 0.5 + padding, _MIN_CROP_SIDE * 0.5)
    half_h = max(height * 0.5 + padding, _MIN_CROP_SIDE * 0.5)

    crop_x1 = int(max(0, np.floor(cx - half_w)))
    crop_y1 = int(max(0, np.floor(cy - half_h)))
    crop_x2 = int(min(image_w, np.ceil(cx + half_w)))
    crop_y2 = int(min(image_h, np.ceil(cy + half_h)))

    return (crop_x1, crop_y1, crop_x2, crop_y2)


def _crop(
    image: np.ndarray,
    crop_box: tuple[int, int, int, int],
) -> np.ndarray | None:
    x1, y1, x2, y2 = crop_box
    if x2 - x1 < _MIN_CROP_SIDE or y2 - y1 < _MIN_CROP_SIDE:
        return None

    crop = image[y1:y2, x1:x2]
    if crop.size == 0:
        return None

    return np.ascontiguousarray(crop)


def _as_polygon_array(polygon: PolygonPoints | None) -> np.ndarray | None:
    if polygon is None or len(polygon) < 3:
        return None

    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None

    points = points[:, :2]
    if not np.isfinite(points).all():
        return None

    return points


def _bbox_from_detection(
    detection_polygon: PolygonPoints | None,
    detection_bbox: BBox,
) -> BBox | None:
    polygon = _as_polygon_array(detection_polygon)
    if polygon is not None:
        polygon_bbox = _bbox_from_points(polygon)
        if polygon_bbox is not None:
            return polygon_bbox

    x1, y1, x2, y2 = detection_bbox
    if x2 <= x1 or y2 <= y1:
        return None

    return (float(x1), float(y1), float(x2), float(y2))


def _bbox_from_points(points: np.ndarray) -> BBox | None:
    if points.ndim != 2 or points.shape[0] == 0 or points.shape[1] < 2:
        return None

    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def _bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def _bbox_diag(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1))


def _has_visible_bbox(
    points: np.ndarray,
    *,
    frame_shape: tuple[int, int],
) -> bool:
    frame_h, frame_w = frame_shape
    bbox = _bbox_from_points(points)
    if bbox is None:
        return False

    x1, y1, x2, y2 = bbox
    visible_x1 = max(0.0, x1)
    visible_y1 = max(0.0, y1)
    visible_x2 = min(float(frame_w), x2)
    visible_y2 = min(float(frame_h), y2)

    visible_area = max(0.0, visible_x2 - visible_x1) * max(0.0, visible_y2 - visible_y1)
    full_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)

    return full_area > 0 and visible_area / full_area >= 0.25


def _median_transform_error(
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    transform: np.ndarray,
    *,
    perspective: bool,
) -> float | None:
    if len(reference_points) == 0:
        return None

    if perspective:
        projected = cv2.perspectiveTransform(
            reference_points.reshape(-1, 1, 2),
            transform.astype(np.float32),
        ).reshape(-1, 2)
    else:
        projected = cv2.transform(
            reference_points.reshape(-1, 1, 2),
            transform.astype(np.float32),
        ).reshape(-1, 2)

    if not np.isfinite(projected).all():
        return None

    errors = np.linalg.norm(projected - frame_points, axis=1)
    if len(errors) == 0:
        return None

    return float(np.median(errors))


def _to_points(points: np.ndarray) -> PolygonPoints:
    return [[float(x), float(y)] for x, y in points[:, :2]]


def _crop_debug(crop_box: tuple[int, int, int, int]) -> dict[str, int]:
    x1, y1, x2, y2 = crop_box
    return {
        "x": int(x1),
        "y": int(y1),
        "w": int(x2 - x1),
        "h": int(y2 - y1),
    }


def _round_debug(value: float | None) -> float | None:
    if value is None:
        return None
    return round(float(value), 4)


def _failed(
    reason: str,
    *,
    method: str | None = None,
    raw_match_count: int = 0,
    inlier_count: int = 0,
    median_error: float | None = None,
    **debug: Any,
) -> ObjectLocalRefinement:
    return ObjectLocalRefinement(
        success=False,
        method=method,
        raw_match_count=raw_match_count,
        inlier_count=inlier_count,
        median_error=median_error,
        reason=reason,
        debug={
            "object_local_reason": reason,
            "object_local_method": method,
            "object_local_raw_matches": raw_match_count,
            "object_local_inliers": inlier_count,
            "object_local_median_error": _round_debug(median_error),
            **debug,
        },
    )
