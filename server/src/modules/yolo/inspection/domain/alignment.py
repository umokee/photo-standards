from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import cv2
import numpy as np

PolygonPoints = list[list[float]]
BBox = tuple[float, float, float, float]

_MIN_LOCAL_INLIERS = 5
_MAX_LOCAL_POINTS = 80

_LOCAL_RANSAC_REPROJ_THRESHOLD = 5.0
_LOCAL_MAX_MEDIAN_ERROR_PX = 9.0

_PADDING_MIN_PX = 35.0
_PADDING_MAX_PX = 140.0
_PADDING_SCALE = 0.85

_MIN_AFFINE_SCALE = 0.60
_MAX_AFFINE_SCALE = 1.85

_MAX_CENTER_DRIFT_MIN_PX = 60.0
_MAX_CENTER_DRIFT_FACTOR = 1.00


class AlignmentStatus(str, Enum):
    SUCCESS = "success"
    INSUFFICIENT_MATCHES = "insufficient_matches"
    INSUFFICIENT_INLIERS = "insufficient_inliers"
    HOMOGRAPHY_FAILED = "homography_failed"


@dataclass(frozen=True, slots=True)
class LocalProjectionData:
    global_homography: np.ndarray | None
    reference_points: np.ndarray | None
    frame_points: np.ndarray | None
    frame_size: tuple[int, int] | None = None
    frame: np.ndarray | None = None
    reference_frame: np.ndarray | None = None
    reference_feature_count: int | None = None
    frame_feature_count: int | None = None
    frame_max_keypoints: int | None = None
    frame_keypoint_grid: tuple[int, int] | None = None
    masked_alignment_used: bool = False
    original_reference_feature_count: int | None = None
    masked_reference_feature_count: int | None = None
    original_reference_keypoints: np.ndarray | None = None
    masked_reference_keypoints: np.ndarray | None = None
    frame_keypoints: np.ndarray | None = None

    @property
    def has_local_points(self) -> bool:
        reference = _as_points(self.reference_points)
        frame = _as_points(self.frame_points)
        return (
            reference is not None
            and frame is not None
            and len(reference) == len(frame)
            and len(reference) >= _MIN_LOCAL_INLIERS
        )


@dataclass(slots=True)
class FrameAlignment:
    status: AlignmentStatus
    homography: np.ndarray | None
    raw_match_count: int
    inlier_count: int
    median_error: float | None = None
    reference_inliers: np.ndarray | None = None
    frame_inliers: np.ndarray | None = None
    reference_matches: np.ndarray | None = None
    frame_matches: np.ndarray | None = None
    method: str = "unknown"
    stage: str | None = None
    reason: str | None = None
    reference_feature_count: int | None = None
    frame_feature_count: int | None = None
    frame_max_keypoints: int | None = None
    frame_keypoint_grid: tuple[int, int] | None = None
    masked_alignment_used: bool = False
    original_reference_feature_count: int | None = None
    masked_reference_feature_count: int | None = None
    reference_size: tuple[int, int] | None = None
    frame_size: tuple[int, int] | None = None

    @property
    def is_success(self) -> bool:
        return self.status == AlignmentStatus.SUCCESS and self.homography is not None

    def to_debug_payload(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "status": self.status.value,
            "stage": self.stage,
            "reason": self.reason,
            "raw_match_count": self.raw_match_count,
            "inlier_count": self.inlier_count,
            "local_reference_match_count": (
                int(len(self.reference_matches))
                if self.reference_matches is not None
                else None
            ),
            "local_frame_match_count": (
                int(len(self.frame_matches)) if self.frame_matches is not None else None
            ),
            "median_error": self.median_error,
            "reference_feature_count": self.reference_feature_count,
            "frame_feature_count": self.frame_feature_count,
            "frame_max_keypoints": self.frame_max_keypoints,
            "frame_keypoint_grid": (
                {
                    "rows": self.frame_keypoint_grid[0],
                    "cols": self.frame_keypoint_grid[1],
                }
                if self.frame_keypoint_grid is not None
                else None
            ),
            "masked_alignment_used": self.masked_alignment_used,
            "original_reference_feature_count": self.original_reference_feature_count,
            "masked_reference_feature_count": self.masked_reference_feature_count,
            "reference_size": (
                {
                    "width": self.reference_size[0],
                    "height": self.reference_size[1],
                }
                if self.reference_size is not None
                else None
            ),
            "frame_size": (
                {
                    "width": self.frame_size[0],
                    "height": self.frame_size[1],
                }
                if self.frame_size is not None
                else None
            ),
        }


def project_polygon(
    polygon: PolygonPoints,
    homography: np.ndarray,
) -> PolygonPoints:
    if not polygon:
        return []

    source = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    projected = cv2.perspectiveTransform(source, homography)
    return projected.reshape(-1, 2).tolist()


def project_polygon_adaptive(
    polygon: PolygonPoints,
    *,
    data: LocalProjectionData | None,
) -> PolygonPoints:
    if data is None:
        return []

    local_projected = _project_polygon_local(polygon, data=data)
    if local_projected:
        return local_projected

    if data.global_homography is not None:
        return _project_polygon_global(polygon, data.global_homography)

    return _project_polygon_scene_affine(polygon, data=data)


def _project_polygon_scene_affine(
    polygon: PolygonPoints,
    *,
    data: LocalProjectionData,
) -> PolygonPoints:
    reference_points = _as_points(data.reference_points)
    frame_points = _as_points(data.frame_points)
    polygon_array = _as_polygon_array(polygon)

    if reference_points is None or frame_points is None or polygon_array is None:
        return []
    if len(reference_points) != len(frame_points) or len(reference_points) < 12:
        return []

    try:
        affine, inliers = cv2.estimateAffinePartial2D(
            reference_points,
            frame_points,
            method=cv2.RANSAC,
            ransacReprojThreshold=_LOCAL_RANSAC_REPROJ_THRESHOLD * 1.2,
            maxIters=1200,
            confidence=0.995,
            refineIters=10,
        )
    except cv2.error:
        return []

    if affine is None or inliers is None:
        return []
    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        return []

    inlier_mask = inliers.reshape(-1).astype(bool)
    inlier_count = int(inlier_mask.sum())
    if inlier_count < 10:
        return []

    inlier_ratio = inlier_count / max(len(reference_points), 1)
    if inlier_ratio < 0.42:
        return []

    if not _affine_scale_ok(affine):
        return []

    if not _reprojection_quality_ok(
        reference_points=reference_points[inlier_mask],
        frame_points=frame_points[inlier_mask],
        transform=affine,
        perspective=False,
    ):
        return []

    projected = cv2.transform(
        polygon_array.reshape(-1, 1, 2),
        affine.astype(np.float32),
    ).reshape(-1, 2)

    if not _projected_polygon_ok(
        source_polygon=polygon_array,
        projected=projected,
        data=data,
        require_global_consistency=False,
    ):
        return []

    return _to_points(projected)


def alignment_message(alignment: FrameAlignment) -> str:
    if alignment.is_success:
        median_error = alignment.median_error or 0.0
        return (
            f"{alignment.method}_success:"
            f"inliers={alignment.inlier_count}/{alignment.raw_match_count},"
            f"err={median_error:.2f}"
        )

    parts = [
        alignment.method or "unknown",
        alignment.status.value,
    ]

    if alignment.stage:
        parts.append(f"stage={alignment.stage}")

    if alignment.reason:
        parts.append(f"reason={alignment.reason}")

    if alignment.frame_feature_count is not None:
        parts.append(f"frame_features={alignment.frame_feature_count}")

    if alignment.reference_feature_count is not None:
        parts.append(f"ref_features={alignment.reference_feature_count}")

    if alignment.raw_match_count:
        parts.append(f"raw={alignment.raw_match_count}")

    if alignment.inlier_count:
        parts.append(f"inliers={alignment.inlier_count}")

    return ":".join(parts[:3]) + ("," + ",".join(parts[3:]) if len(parts) > 3 else "")


def failed_alignment(
    status: AlignmentStatus = AlignmentStatus.INSUFFICIENT_MATCHES,
    *,
    raw_match_count: int = 0,
    inlier_count: int = 0,
) -> FrameAlignment:
    return FrameAlignment(
        status=status,
        homography=None,
        raw_match_count=raw_match_count,
        inlier_count=inlier_count,
        reference_matches=None,
        frame_matches=None,
        method="unknown",
        stage="external_failed_alignment",
        reason=status.value,
    )


def _project_polygon_local(
    polygon: PolygonPoints,
    *,
    data: LocalProjectionData,
) -> PolygonPoints:
    reference_points = _as_points(data.reference_points)
    frame_points = _as_points(data.frame_points)
    polygon_array = _as_polygon_array(polygon)

    if reference_points is None or frame_points is None or polygon_array is None:
        return []

    if len(reference_points) != len(frame_points):
        return []

    if len(reference_points) < _MIN_LOCAL_INLIERS:
        return []

    bbox = _bbox_from_array(polygon_array)
    if bbox is None:
        return []

    local_reference, local_frame = _select_local_matches(
        reference_points=reference_points,
        frame_points=frame_points,
        bbox=bbox,
    )

    if len(local_reference) >= 4:
        projected = _try_local_homography(
            polygon_array=polygon_array,
            local_reference=local_reference,
            local_frame=local_frame,
            data=data,
        )
        if projected:
            return projected

    if len(local_reference) >= _MIN_LOCAL_INLIERS:
        projected = _try_local_affine(
            polygon_array=polygon_array,
            local_reference=local_reference,
            local_frame=local_frame,
            data=data,
        )
        if projected:
            return projected

    return []


def _try_local_homography(
    *,
    polygon_array: np.ndarray,
    local_reference: np.ndarray,
    local_frame: np.ndarray,
    data: LocalProjectionData,
    require_global_consistency: bool = True,
) -> PolygonPoints:
    homography, inliers = cv2.findHomography(
        local_reference.reshape(-1, 1, 2),
        local_frame.reshape(-1, 1, 2),
        method=cv2.RANSAC,
        ransacReprojThreshold=_LOCAL_RANSAC_REPROJ_THRESHOLD,
    )

    if homography is None or inliers is None:
        return []

    if homography.shape != (3, 3) or not np.isfinite(homography).all():
        return []

    inlier_mask = inliers.reshape(-1).astype(bool)
    if int(inlier_mask.sum()) < 4:
        return []

    projected = cv2.perspectiveTransform(
        polygon_array.reshape(-1, 1, 2),
        homography.astype(np.float32),
    ).reshape(-1, 2)

    if not _projected_polygon_ok(
        source_polygon=polygon_array,
        projected=projected,
        data=data,
        require_global_consistency=require_global_consistency,
    ):
        return []

    if not _reprojection_quality_ok(
        reference_points=local_reference[inlier_mask],
        frame_points=local_frame[inlier_mask],
        transform=homography,
        perspective=True,
    ):
        return []

    return _to_points(projected)


def _try_local_affine(
    *,
    polygon_array: np.ndarray,
    local_reference: np.ndarray,
    local_frame: np.ndarray,
    data: LocalProjectionData,
    require_global_consistency: bool = True,
) -> PolygonPoints:
    affine, inliers = cv2.estimateAffinePartial2D(
        local_reference,
        local_frame,
        method=cv2.RANSAC,
        ransacReprojThreshold=_LOCAL_RANSAC_REPROJ_THRESHOLD,
        maxIters=800,
        confidence=0.99,
        refineIters=10,
    )

    if affine is None or inliers is None:
        return []

    if affine.shape != (2, 3) or not np.isfinite(affine).all():
        return []

    inlier_mask = inliers.reshape(-1).astype(bool)
    if int(inlier_mask.sum()) < _MIN_LOCAL_INLIERS:
        return []

    if not _affine_scale_ok(affine):
        return []

    projected = cv2.transform(
        polygon_array.reshape(-1, 1, 2),
        affine.astype(np.float32),
    ).reshape(-1, 2)

    if not _projected_polygon_ok(
        source_polygon=polygon_array,
        projected=projected,
        data=data,
        require_global_consistency=require_global_consistency,
    ):
        return []

    if not _reprojection_quality_ok(
        reference_points=local_reference[inlier_mask],
        frame_points=local_frame[inlier_mask],
        transform=affine,
        perspective=False,
    ):
        return []

    return _to_points(projected)


def _padding_for_bbox(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    return max(
        _PADDING_MIN_PX,
        min(_PADDING_MAX_PX, max(width, height) * _PADDING_SCALE),
    )


def _select_local_matches(
    *,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    padding = _padding_for_bbox(bbox)

    mask = (
        (reference_points[:, 0] >= x1 - padding)
        & (reference_points[:, 0] <= x2 + padding)
        & (reference_points[:, 1] >= y1 - padding)
        & (reference_points[:, 1] <= y2 + padding)
    )

    local_reference = reference_points[mask]
    local_frame = frame_points[mask]

    if len(local_reference) <= _MAX_LOCAL_POINTS:
        return local_reference, local_frame

    center = np.array([(x1 + x2) * 0.5, (y1 + y2) * 0.5], dtype=np.float32)
    distances = np.linalg.norm(local_reference - center, axis=1)
    indices = np.argsort(distances)[:_MAX_LOCAL_POINTS]

    return local_reference[indices], local_frame[indices]


def _affine_scale_ok(affine: np.ndarray) -> bool:
    matrix = affine[:, :2].astype(np.float64)
    scale_x = float(np.linalg.norm(matrix[:, 0]))
    scale_y = float(np.linalg.norm(matrix[:, 1]))

    return (
        _MIN_AFFINE_SCALE <= scale_x <= _MAX_AFFINE_SCALE
        and _MIN_AFFINE_SCALE <= scale_y <= _MAX_AFFINE_SCALE
    )


def _reprojection_quality_ok(
    *,
    reference_points: np.ndarray,
    frame_points: np.ndarray,
    transform: np.ndarray,
    perspective: bool,
) -> bool:
    if len(reference_points) == 0:
        return False

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
        return False

    errors = np.linalg.norm(projected - frame_points, axis=1)
    if len(errors) == 0:
        return False

    return float(np.median(errors)) <= _LOCAL_MAX_MEDIAN_ERROR_PX


def _projected_polygon_ok(
    *,
    source_polygon: np.ndarray,
    projected: np.ndarray,
    data: LocalProjectionData,
    require_global_consistency: bool = True,
) -> bool:
    if not np.isfinite(projected).all():
        return False

    if not _projected_polygon_visible(projected, data.frame_size):
        return False

    if not _polygon_area_ok(source_polygon, projected):
        return False

    if require_global_consistency and data.global_homography is not None:
        return _local_projection_consistent_with_global(
            source_polygon=source_polygon,
            local_projected=projected,
            global_homography=data.global_homography,
        )

    return True


def _polygon_area_ok(source_polygon: np.ndarray, projected: np.ndarray) -> bool:
    source_area = abs(float(cv2.contourArea(source_polygon.astype(np.float32))))
    projected_area = abs(float(cv2.contourArea(projected.astype(np.float32))))

    if source_area <= 1.0 or projected_area <= 1.0:
        return False

    ratio = projected_area / source_area
    return 0.05 <= ratio <= 20.0


def _local_projection_consistent_with_global(
    *,
    source_polygon: np.ndarray,
    local_projected: np.ndarray,
    global_homography: np.ndarray,
) -> bool:
    global_projected = _project_polygon_global(
        _to_points(source_polygon),
        global_homography,
    )

    global_array = _as_polygon_array(global_projected)
    if global_array is None:
        return True

    source_bbox = _bbox_from_array(source_polygon)
    if source_bbox is None:
        return True

    source_diag = _bbox_diag(source_bbox)
    max_drift = max(
        _MAX_CENTER_DRIFT_MIN_PX,
        source_diag * _MAX_CENTER_DRIFT_FACTOR,
    )

    local_center = _polygon_center(local_projected)
    global_center = _polygon_center(global_array)

    drift = float(np.linalg.norm(local_center - global_center))
    return drift <= max_drift


def _project_polygon_global(
    polygon: PolygonPoints,
    homography: np.ndarray,
) -> PolygonPoints:
    try:
        projected = project_polygon(polygon, homography)
    except Exception:
        return []

    if not projected or len(projected) < 3:
        return []

    if not all(len(point) >= 2 for point in projected):
        return []

    return [[float(point[0]), float(point[1])] for point in projected]


def _projected_polygon_visible(
    polygon: np.ndarray,
    frame_size: tuple[int, int] | None,
) -> bool:
    if frame_size is None:
        return True

    frame_width, frame_height = frame_size
    bbox = _bbox_from_array(polygon)
    if bbox is None:
        return False

    x1, y1, x2, y2 = bbox

    margin_x = max(80.0, frame_width * 0.15)
    margin_y = max(80.0, frame_height * 0.15)

    return not (
        x2 < -margin_x
        or y2 < -margin_y
        or x1 > frame_width + margin_x
        or y1 > frame_height + margin_y
    )


def _as_points(value: np.ndarray | None) -> np.ndarray | None:
    if value is None:
        return None

    array = np.asarray(value, dtype=np.float32).reshape(-1, 2)

    if len(array) == 0:
        return None

    if not np.isfinite(array).all():
        return None

    return array


def _as_polygon_array(polygon: PolygonPoints) -> np.ndarray | None:
    if len(polygon) < 3:
        return None

    array = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)

    if len(array) < 3:
        return None

    if not np.isfinite(array).all():
        return None

    return array


def _bbox_from_array(
    points: np.ndarray,
) -> tuple[float, float, float, float] | None:
    if len(points) == 0:
        return None

    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))

    if x2 <= x1 or y2 <= y1:
        return None

    return x1, y1, x2, y2


def _bbox_diag(bbox: tuple[float, float, float, float]) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1))


def _polygon_center(points: np.ndarray) -> np.ndarray:
    return np.mean(points.astype(np.float32), axis=0)


def _to_points(points: np.ndarray) -> PolygonPoints:
    return [[float(x), float(y)] for x, y in points.reshape(-1, 2)]
