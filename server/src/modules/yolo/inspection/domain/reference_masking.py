from __future__ import annotations

import cv2
import numpy as np

PolygonPoints = list[list[float]]


def mask_reference_polygons(
    image: np.ndarray,
    polygons: list[PolygonPoints],
    *,
    padding_px: int = 3,
) -> np.ndarray:
    """Return a copy of reference image with expected object interiors erased.

    Global LightGlue alignment should rely on stable scene/context features, not on
    product interiors that may be absent on the checked photo.  The fill color is
    estimated from pixels outside all expected polygons to avoid introducing hard
    black/white artificial corners.
    """
    valid = [_polygon_to_int_array(polygon) for polygon in polygons]
    valid = [polygon for polygon in valid if polygon is not None and len(polygon) >= 3]
    if not valid:
        return image

    masked = image.copy()
    fill_color = _outside_median_color(image, valid)

    for polygon in valid:
        erase_polygon = polygon
        if padding_px > 0:
            erase_polygon = _expand_polygon(polygon, padding_px)
        cv2.fillPoly(masked, [erase_polygon], fill_color)

    return masked


def _polygon_to_int_array(polygon: PolygonPoints) -> np.ndarray | None:
    if not polygon or len(polygon) < 3:
        return None
    try:
        array = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    except (TypeError, ValueError):
        return None
    if array.shape[0] < 3 or not np.isfinite(array).all():
        return None
    return np.round(array).astype(np.int32)


def _outside_median_color(image: np.ndarray, polygons: list[np.ndarray]) -> tuple[int, int, int]:
    if image.size == 0:
        return (0, 0, 0)

    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, polygons, 255)
    outside = image[mask == 0]
    if outside.size == 0:
        outside = image.reshape(-1, image.shape[-1])

    median = np.median(outside.astype(np.float32), axis=0)
    return tuple(int(round(float(value))) for value in median[:3])


def _expand_polygon(polygon: np.ndarray, padding_px: int) -> np.ndarray:
    center = polygon.astype(np.float32).mean(axis=0)
    vectors = polygon.astype(np.float32) - center
    lengths = np.linalg.norm(vectors, axis=1)
    safe_lengths = np.where(lengths > 1e-6, lengths, 1.0)
    expanded = center + vectors * ((safe_lengths + float(padding_px)) / safe_lengths)[:, None]
    return np.round(expanded).astype(np.int32)
