from __future__ import annotations

import cv2
import numpy as np
from modules.yolo.inspection.domain.matcher_structs import BBox
from shapely.errors import GEOSException
from shapely.geometry import Polygon
from shapely.validation import make_valid


def bbox_from_polygon(polygon: list[list[float]] | None) -> BBox | None:
    if polygon is None or len(polygon) < 3:
        return None

    xs = [float(point[0]) for point in polygon]
    ys = [float(point[1]) for point in polygon]
    x1, x2 = min(xs), max(xs)
    y1, y2 = min(ys), max(ys)

    if x2 <= x1 or y2 <= y1:
        return None

    return (x1, y1, x2, y2)


def bbox_iou(a: BBox, b: BBox) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    if inter_area <= 0:
        return 0.0

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    union = area_a + area_b - inter_area
    if union <= 0:
        return 0.0

    return float(inter_area / union)


def bbox_area(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)


def bbox_center(bbox: BBox) -> tuple[float, float]:
    x1, y1, x2, y2 = bbox
    return ((x1 + x2) * 0.5, (y1 + y2) * 0.5)


def bbox_diag(bbox: BBox) -> float:
    x1, y1, x2, y2 = bbox
    return float(np.hypot(x2 - x1, y2 - y1))


def polygon_from_bbox(bbox: BBox) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def polygon_iou(
    polygon_a: list[list[float]],
    polygon_b: list[list[float]],
) -> float:
    try:
        shape_a = make_valid(Polygon(polygon_a))
        shape_b = make_valid(Polygon(polygon_b))
    except (ValueError, GEOSException):
        return 0.0

    if shape_a.is_empty or shape_b.is_empty:
        return 0.0

    try:
        intersection = shape_a.intersection(shape_b).area
        union = shape_a.union(shape_b).area
    except GEOSException:
        return 0.0

    if union <= 0:
        return 0.0
    return float(intersection / union)


def translate_bbox(bbox: BBox, *, dx: float, dy: float) -> BBox:
    x1, y1, x2, y2 = bbox
    return (x1 + dx, y1 + dy, x2 + dx, y2 + dy)



def polygon_has_usable_area(polygon: list[list[float]]) -> bool:
    if len(polygon) < 3:
        return False
    try:
        shape = make_valid(Polygon(polygon))
    except (ValueError, GEOSException):
        return False
    return not shape.is_empty and float(shape.area) > 1.0



def project_polygon_by_affine(
    polygon: list[list[float]],
    affine: np.ndarray,
) -> list[list[float]]:
    if len(polygon) < 3:
        return []

    source = np.asarray(polygon, dtype=np.float32).reshape(-1, 1, 2)
    try:
        projected = cv2.transform(source, affine.astype(np.float32)).reshape(-1, 2)
    except cv2.error:
        return []

    if not np.isfinite(projected).all():
        return []

    return [[float(x), float(y)] for x, y in projected.tolist()]


def project_points_with_homography(
    points: np.ndarray,
    homography: np.ndarray,
) -> np.ndarray | None:
    array = np.asarray(points, dtype=np.float32)
    if array.ndim != 2 or array.shape[1] < 2 or len(array) == 0:
        return None
    if not np.isfinite(array[:, :2]).all():
        return None
    try:
        projected = cv2.perspectiveTransform(
            array[:, :2].reshape(-1, 1, 2),
            homography.astype(np.float32),
        ).reshape(-1, 2)
    except cv2.error:
        return None
    if not np.isfinite(projected).all():
        return None
    return projected.astype(np.float32)


def translate_polygon(
    polygon: list[list[float]],
    *,
    dx: float,
    dy: float,
) -> list[list[float]]:
    if len(polygon) < 3:
        return []
    translated: list[list[float]] = []
    for point in polygon:
        if len(point) < 2:
            return []
        x = float(point[0]) + float(dx)
        y = float(point[1]) + float(dy)
        if not np.isfinite([x, y]).all():
            return []
        translated.append([x, y])
    return translated


def affine_reprojection_median_error(
    affine: np.ndarray,
    *,
    source: np.ndarray,
    target: np.ndarray,
) -> float | None:
    if len(source) == 0 or len(target) == 0:
        return None

    try:
        projected = cv2.transform(
            source.reshape(-1, 1, 2),
            affine.astype(np.float32),
        ).reshape(-1, 2)
    except cv2.error:
        return None

    if not np.isfinite(projected).all():
        return None

    errors = np.linalg.norm(projected - target, axis=1)
    if len(errors) == 0 or not np.isfinite(errors).all():
        return None

    return float(np.median(errors))


def bbox_area_similarity(a: BBox, b: BBox) -> float:
    area_a = max(1.0, bbox_area(a))
    area_b = max(1.0, bbox_area(b))
    ratio = area_a / area_b
    return float(max(0.0, min(1.0, min(ratio, 1.0 / ratio))))


def bbox_center_distance_factor(a: BBox, b: BBox) -> float:
    ax, ay = bbox_center(a)
    bx, by = bbox_center(b)
    distance = float(np.hypot(ax - bx, ay - by))
    return distance / max(1.0, bbox_diag(b))


def polygon_axis_delta(
    polygon_a: list[list[float]],
    polygon_b: list[list[float]],
) -> tuple[float | None, float | None]:
    metrics_a = _polygon_axis_metrics(polygon_a)
    metrics_b = _polygon_axis_metrics(polygon_b)
    if metrics_a is None or metrics_b is None:
        return None, None

    angle_a, major_a = metrics_a
    angle_b, major_b = metrics_b
    angle_delta = abs(angle_a - angle_b) % 180.0
    if angle_delta > 90.0:
        angle_delta = 180.0 - angle_delta

    if major_a <= 0.0 or major_b <= 0.0:
        return float(angle_delta), None

    return float(angle_delta), float(major_a / major_b)


def _polygon_axis_metrics(
    polygon: list[list[float]],
) -> tuple[float, float] | None:
    if len(polygon) < 3:
        return None

    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] < 2 or not np.isfinite(points).all():
        return None

    points = points[:, :2]
    centered = points - np.mean(points, axis=0, keepdims=True)
    if len(centered) < 2:
        return None

    try:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
    except np.linalg.LinAlgError:
        return None

    if vt.shape[0] == 0:
        return None

    major_axis = vt[0]
    angle = float(np.degrees(np.arctan2(major_axis[1], major_axis[0])))
    projection = centered @ major_axis
    major_length = float(np.max(projection) - np.min(projection))
    if major_length <= 0.0 or not np.isfinite(major_length):
        return None

    return angle, major_length


def empty_match_points() -> np.ndarray:
    return np.empty((0, 2), dtype=np.float32)


def as_match_points(points: np.ndarray | None) -> np.ndarray | None:
    if points is None:
        return None
    array = np.asarray(points, dtype=np.float32)
    if array.ndim == 3 and array.shape[1:] == (1, 2):
        array = array.reshape(-1, 2)
    if array.ndim != 2 or array.shape[1] < 2:
        return None
    if len(array) == 0:
        return None
    return array[:, :2]


def expand_bbox(
    bbox: BBox,
    *,
    factor: float,
    frame_size: tuple[int, int] | None = None,
) -> BBox:
    x1, y1, x2, y2 = bbox
    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)
    cx, cy = bbox_center(bbox)
    new_w = width * factor
    new_h = height * factor
    result = (
        cx - new_w * 0.5,
        cy - new_h * 0.5,
        cx + new_w * 0.5,
        cy + new_h * 0.5,
    )

    if frame_size is None:
        return result

    fw, fh = frame_size
    return (
        max(0.0, result[0]),
        max(0.0, result[1]),
        min(float(fw), result[2]),
        min(float(fh), result[3]),
    )


def bbox_containment(inner: BBox, outer: BBox) -> float:
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer

    inter_x1 = max(ix1, ox1)
    inter_y1 = max(iy1, oy1)
    inter_x2 = min(ix2, ox2)
    inter_y2 = min(iy2, oy2)

    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    inner_area = bbox_area(inner)
    if inner_area <= 0:
        return 0.0
    return float(inter_area / inner_area)


def point_in_bbox(point: np.ndarray, bbox: BBox) -> bool:
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = bbox
    return x1 <= x <= x2 and y1 <= y <= y2


def point_in_any_bbox(point: np.ndarray, bboxes: list[BBox]) -> bool:
    return any(point_in_bbox(point, bbox) for bbox in bboxes)



def point_outside_np_polygon_margin(
    point: np.ndarray,
    polygon: np.ndarray,
    *,
    margin: float,
) -> bool:
    if polygon.ndim != 2 or polygon.shape[0] < 3 or polygon.shape[1] < 2:
        return True
    if not np.isfinite(point).all() or not np.isfinite(polygon).all():
        return True

    try:
        signed_distance = cv2.pointPolygonTest(
            polygon[:, :2].astype(np.float32),
            (float(point[0]), float(point[1])),
            True,
        )
    except cv2.error:
        return True

    return float(signed_distance) < -max(0.0, float(margin))


def is_visible_in_frame(
    bbox: BBox,
    frame_size: tuple[int, int],
    *,
    min_visible_fraction: float = 0.20,
) -> bool:
    x1, y1, x2, y2 = bbox
    fw, fh = frame_size

    visible_x1 = max(0.0, x1)
    visible_y1 = max(0.0, y1)
    visible_x2 = min(float(fw), x2)
    visible_y2 = min(float(fh), y2)

    if visible_x2 <= visible_x1 or visible_y2 <= visible_y1:
        return False

    visible_area = (visible_x2 - visible_x1) * (visible_y2 - visible_y1)
    full_area = (x2 - x1) * (y2 - y1)
    if full_area <= 0:
        return False

    return visible_area / full_area >= min_visible_fraction



__all__ = [
    "affine_reprojection_median_error",
    "as_match_points",
    "bbox_area",
    "bbox_area_similarity",
    "bbox_center",
    "bbox_center_distance_factor",
    "bbox_containment",
    "bbox_diag",
    "bbox_from_polygon",
    "bbox_iou",
    "empty_match_points",
    "expand_bbox",
    "is_visible_in_frame",
    "point_in_any_bbox",
    "point_in_bbox",
    "point_outside_np_polygon_margin",
    "polygon_axis_delta",
    "_polygon_axis_metrics",
    "polygon_from_bbox",
    "polygon_has_usable_area",
    "polygon_iou",
    "project_points_with_homography",
    "project_polygon_by_affine",
    "translate_bbox",
    "translate_polygon",
]
