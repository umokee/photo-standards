from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .types import SegmentMatch

COLOR_OK = (93, 155, 58)
COLOR_MISSING = (70, 70, 184)
COLOR_EXTRA = (0, 122, 200)
COLOR_UNMATCHED = (48, 154, 209)

MIN_LABEL_FONT_SIZE = 16
MAX_LABEL_FONT_SIZE = 34
MIN_LINE_THICKNESS = 3
MAX_LINE_THICKNESS = 8

POLYGON_OFFSCREEN_MARGIN_FRACTION = 0.15
POLYGON_MAX_BBOX_SIDE_FRACTION = 0.70
POLYGON_MIN_VISIBLE_BBOX_FRACTION = 0.45
POLYGON_MAX_EDGE_FRACTION = 0.45

ALPHA_FILL = 0.25
THICKNESS_LINE = 4
LABEL_FONT_SIZE = 16
LABEL_PADDING = 4

_FONT_PATH = (
    Path(__file__).resolve().parents[6]
    / "client"
    / "src"
    / "assets"
    / "fonts"
    / "ttf"
    / "JetBrainsMono-Bold.ttf"
)
_FONT: ImageFont.ImageFont | ImageFont.FreeTypeFont | None = None


def render_overlay(
    frame: np.ndarray,
    matches: list[SegmentMatch],
    *,
    fps: float | None = None,
    alignment_message: str | None = None,
    polygon_transform: np.ndarray | None = None,
) -> np.ndarray:
    del fps

    output = frame.copy()

    if alignment_message is not None and not matches:
        _draw_label(
            output,
            f"Не удалось совместить эталон: {alignment_message}",
            (16, 16),
            COLOR_MISSING,
        )
        return output

    _render_polygon_overlay(
        output,
        matches,
        polygon_transform=polygon_transform,
    )

    if alignment_message is not None:
        _draw_label(
            output,
            f"Совмещение: {alignment_message}",
            (16, 52),
            COLOR_MISSING,
        )

    return output


def _render_polygon_overlay(
    output: np.ndarray,
    matches: list[SegmentMatch],
    *,
    polygon_transform: np.ndarray | None,
) -> None:
    frame_shape = output.shape[:2]
    thickness = _line_thickness(frame_shape)
    label_font_size = _label_font_size(frame_shape)
    label_padding = _label_padding(label_font_size)

    draw_items: list[tuple[list[list[float]], tuple[int, int, int], str]] = []

    for match in matches:
        if match.status == "unmatched":
            if match.expected_polygon is not None:
                expected_polygon = _prepare_polygon_for_render(
                    match.expected_polygon,
                    polygon_transform=polygon_transform,
                    frame_shape=frame_shape,
                )
                if expected_polygon is not None:
                    draw_items.append(
                        (
                            expected_polygon,
                            COLOR_MISSING,
                            _expected_label_for_match(match),
                        )
                    )

            if match.detected_polygon is not None:
                detected_polygon = _prepare_polygon_for_render(
                    match.detected_polygon,
                    polygon_transform=polygon_transform,
                    frame_shape=frame_shape,
                )
                if detected_polygon is not None:
                    draw_items.append(
                        (
                            detected_polygon,
                            COLOR_UNMATCHED,
                            _label_for_match(match),
                        )
                    )

            continue

        polygon = _polygon_for_render(
            match,
            polygon_transform=polygon_transform,
            frame_shape=frame_shape,
        )
        if polygon is None:
            continue

        draw_items.append((polygon, _color_for_status(match.status), _label_for_match(match)))

    if not draw_items:
        return

    fill_overlay = output.copy()
    glow_overlay = output.copy()

    for polygon, color, _label in draw_items:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)

        cv2.fillPoly(fill_overlay, [pts], color)

        cv2.polylines(
            glow_overlay,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness + 4,
            lineType=cv2.LINE_AA,
        )

    cv2.addWeighted(fill_overlay, ALPHA_FILL, output, 1 - ALPHA_FILL, 0, output)
    cv2.addWeighted(glow_overlay, 0.35, output, 0.65, 0, output)

    for polygon, color, _label in draw_items:
        pts = np.array(polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(
            output,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness,
            lineType=cv2.LINE_AA,
        )

    for polygon, color, label in draw_items:
        _draw_label(
            output,
            label,
            _label_anchor(polygon),
            color,
            font_size=label_font_size,
            padding=label_padding,
        )


def _polygon_for_render(
    match: SegmentMatch,
    *,
    polygon_transform: np.ndarray | None,
    frame_shape: tuple[int, int],
) -> list[list[float]] | None:
    if match.status == "missing":
        polygon = match.expected_polygon
    elif match.status == "ok":
        polygon = match.detected_polygon or match.expected_polygon
    elif match.status == "extra":
        polygon = match.detected_polygon
    else:
        return None

    return _prepare_polygon_for_render(
        polygon,
        polygon_transform=polygon_transform,
        frame_shape=frame_shape,
    )


def _prepare_polygon_for_render(
    polygon: list[list[float]] | None,
    *,
    polygon_transform: np.ndarray | None,
    frame_shape: tuple[int, int],
) -> list[list[float]] | None:
    if polygon is None or len(polygon) < 3:
        return None

    if polygon_transform is not None:
        polygon = _transform_polygon(polygon, polygon_transform)
        if polygon is None or len(polygon) < 3:
            return None

    return _sanitize_polygon_for_render(polygon, frame_shape=frame_shape)


def _transform_polygon(
    polygon: list[list[float]],
    transform: np.ndarray,
) -> list[list[float]] | None:
    matrix = _normalize_transform(transform)
    if matrix is None:
        return None

    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None

    points = points[:, :2].reshape(-1, 1, 2)
    transformed = cv2.perspectiveTransform(points, matrix).reshape(-1, 2)

    if not np.isfinite(transformed).all():
        return None

    return [[float(x), float(y)] for x, y in transformed]


def _normalize_transform(transform: np.ndarray) -> np.ndarray | None:
    matrix = np.asarray(transform, dtype=np.float32)

    if matrix.shape == (2, 3):
        normalized = np.eye(3, dtype=np.float32)
        normalized[:2, :] = matrix
        return normalized

    if matrix.shape != (3, 3):
        return None
    if not np.isfinite(matrix).all():
        return None
    if abs(float(matrix[2, 2])) < 1e-9:
        return None

    matrix = matrix / matrix[2, 2]
    if not np.isfinite(matrix).all():
        return None

    return matrix.astype(np.float32)


def _color_for_status(status: str) -> tuple[int, int, int]:
    if status == "ok":
        return COLOR_OK
    if status == "missing":
        return COLOR_MISSING
    if status == "extra":
        return COLOR_EXTRA
    if status == "unmatched":
        return COLOR_UNMATCHED

    return (128, 128, 128)


def _label_for_match(match: SegmentMatch) -> str:
    name = match.name
    if len(name) > 22:
        name = name[:21] + "…"

    if match.status == "missing":
        return f"{name} · нет"

    if match.status == "unmatched":
        if match.confidence is not None:
            return f"{name} · не сопост. {int(match.confidence * 100)}%"
        return f"{name} · не сопост."

    if match.confidence is not None:
        return f"{name} {int(match.confidence * 100)}%"

    return name


def _expected_label_for_match(match: SegmentMatch) -> str:
    name = match.name
    if len(name) > 18:
        name = name[:17] + "…"

    return f"ожид. {name}"


def _label_anchor(polygon: list[list[float]]) -> tuple[int, int]:
    xs = [int(p[0]) for p in polygon]
    ys = [int(p[1]) for p in polygon]
    return (min(xs), min(ys))


@lru_cache(maxsize=32)
def _get_font(font_size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(_FONT_PATH), font_size)
    except OSError:
        return ImageFont.load_default()


@lru_cache(maxsize=1024)
def _render_label_plate(
    text: str,
    color: tuple[int, int, int],
    font_size: int,
    padding: int,
) -> np.ndarray:
    font = _get_font(font_size)
    bbox = font.getbbox(text)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    box_w = text_w + padding * 2
    box_h = text_h + padding * 2

    plate = Image.new("RGB", (box_w, box_h), (color[2], color[1], color[0]))
    draw = ImageDraw.Draw(plate)
    draw.text(
        (padding, padding - bbox[1]),
        text,
        fill=(255, 255, 255),
        font=font,
    )
    return cv2.cvtColor(np.array(plate), cv2.COLOR_RGB2BGR)


def _draw_label(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    color: tuple[int, int, int],
    *,
    font_size: int | None = None,
    padding: int | None = None,
) -> None:
    if font_size is None:
        font_size = _label_font_size(image.shape[:2])

    if padding is None:
        padding = _label_padding(font_size)

    plate_bgr = _render_label_plate(text, color, font_size, padding)
    h, w = plate_bgr.shape[:2]

    x, y = anchor
    box_x = max(0, min(x, image.shape[1] - w))
    box_y = max(0, y - h - 2)

    y2 = min(box_y + h, image.shape[0])
    x2 = min(box_x + w, image.shape[1])

    if y2 > box_y and x2 > box_x:
        image[box_y:y2, box_x:x2] = plate_bgr[: y2 - box_y, : x2 - box_x]


def _sanitize_polygon_for_render(
    polygon: list[list[float]],
    *,
    frame_shape: tuple[int, int],
) -> list[list[float]] | None:
    frame_h, frame_w = frame_shape

    points = np.asarray(polygon, dtype=np.float32)
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None

    points = points[:, :2]

    if not np.isfinite(points).all():
        return None

    points = _remove_near_duplicate_points(points)
    if points.shape[0] < 3:
        return None

    xs = points[:, 0]
    ys = points[:, 1]

    margin = max(frame_w, frame_h) * POLYGON_OFFSCREEN_MARGIN_FRACTION

    if (
        np.any(xs < -margin)
        or np.any(xs > frame_w + margin)
        or np.any(ys < -margin)
        or np.any(ys > frame_h + margin)
    ):
        return None

    bbox_w = float(np.max(xs) - np.min(xs))
    bbox_h = float(np.max(ys) - np.min(ys))

    if bbox_w < 2.0 or bbox_h < 2.0:
        return None

    if bbox_w > frame_w * POLYGON_MAX_BBOX_SIDE_FRACTION:
        return None

    if bbox_h > frame_h * POLYGON_MAX_BBOX_SIDE_FRACTION:
        return None

    if not _has_enough_visible_bbox(points, frame_w=frame_w, frame_h=frame_h):
        return None

    if (
        _max_edge_length(points)
        > np.hypot(frame_w, frame_h) * POLYGON_MAX_EDGE_FRACTION
    ):
        return None

    area = abs(float(cv2.contourArea(points.reshape(-1, 1, 2))))
    if area < 4.0:
        return None

    return [[float(x), float(y)] for x, y in points]


def _remove_near_duplicate_points(points: np.ndarray) -> np.ndarray:
    cleaned: list[np.ndarray] = []

    for point in points:
        if cleaned and float(np.linalg.norm(point - cleaned[-1])) < 1.0:
            continue
        cleaned.append(point)

    if len(cleaned) >= 2 and float(np.linalg.norm(cleaned[0] - cleaned[-1])) < 1.0:
        cleaned.pop()

    if not cleaned:
        return np.empty((0, 2), dtype=np.float32)

    return np.asarray(cleaned, dtype=np.float32)


def _has_enough_visible_bbox(
    points: np.ndarray,
    *,
    frame_w: int,
    frame_h: int,
) -> bool:
    x1 = float(np.min(points[:, 0]))
    y1 = float(np.min(points[:, 1]))
    x2 = float(np.max(points[:, 0]))
    y2 = float(np.max(points[:, 1]))

    bbox_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if bbox_area <= 0:
        return False

    visible_x1 = max(0.0, x1)
    visible_y1 = max(0.0, y1)
    visible_x2 = min(float(frame_w), x2)
    visible_y2 = min(float(frame_h), y2)

    visible_area = max(0.0, visible_x2 - visible_x1) * max(0.0, visible_y2 - visible_y1)

    return visible_area / bbox_area >= POLYGON_MIN_VISIBLE_BBOX_FRACTION


def _max_edge_length(points: np.ndarray) -> float:
    if points.shape[0] < 2:
        return 0.0

    rolled = np.roll(points, shift=-1, axis=0)
    edges = np.linalg.norm(points - rolled, axis=1)
    return float(np.max(edges))


def _label_font_size(frame_shape: tuple[int, int]) -> int:
    frame_h, frame_w = frame_shape
    base = max(frame_w, frame_h)
    return int(np.clip(round(base * 0.014), MIN_LABEL_FONT_SIZE, MAX_LABEL_FONT_SIZE))


def _line_thickness(frame_shape: tuple[int, int]) -> int:
    frame_h, frame_w = frame_shape
    base = max(frame_w, frame_h)
    return int(np.clip(round(base * 0.0032), MIN_LINE_THICKNESS, MAX_LINE_THICKNESS))


def _label_padding(font_size: int) -> int:
    return max(4, round(font_size * 0.28))
