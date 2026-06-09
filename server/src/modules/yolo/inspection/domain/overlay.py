from __future__ import annotations

from dataclasses import dataclass
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
COLOR_UNCONFIRMED = (96, 96, 176)
COLOR_HALO = (10, 10, 10)
COLOR_CONNECTOR_HALO = (255, 255, 255)

MIN_LABEL_FONT_SIZE = 15
MAX_LABEL_FONT_SIZE = 30
MIN_LINE_THICKNESS = 3
MAX_LINE_THICKNESS = 7

POLYGON_OFFSCREEN_MARGIN_FRACTION = 0.15
POLYGON_MAX_BBOX_SIDE_FRACTION = 0.70
POLYGON_MIN_VISIBLE_BBOX_FRACTION = 0.45
POLYGON_MAX_EDGE_FRACTION = 0.45

ALPHA_FILL_OK = 0.18
ALPHA_FILL_PROBLEM = 0.28
THICKNESS_LINE = 4
LABEL_FONT_SIZE = 16
LABEL_PADDING = 4

# v20_overlay_polish: keep problem labels readable first; dense scenes get compact
# OK labels, stronger polygon halo, and a wider label-placement search.
MAX_FULL_OK_LABELS_DENSE = 10
DENSE_LABEL_COUNT = 14
LABEL_CANDIDATE_RINGS = 4
LABEL_MIN_GAP = 3

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


@dataclass(slots=True)
class _OverlayItem:
    polygon: list[list[float]]
    color: tuple[int, int, int]
    label: str
    status: str
    priority: int
    compact_label: str | None = None


def render_overlay(
    frame: np.ndarray,
    matches: list[SegmentMatch],
    *,
    fps: float | None = None,
    alignment_message: str | None = None,
    polygon_transform: np.ndarray | None = None,
    debug_projection: bool = True,
    pose_method: str | None = None,
) -> np.ndarray:
    del fps, debug_projection

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

    if pose_method is not None:
        method_label = (
            "Фолбэк alignment"
            if pose_method == "feature_slot_fallback"
            else f"Метод: {pose_method}"
        )
        _draw_label(
            output,
            method_label,
            (16, 124 if projection_label is not None else 88),
            COLOR_EXTRA,
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

    normal_items: list[_OverlayItem] = []
    unconfirmed_items: list[_OverlayItem] = []

    for match in matches:
        render_item = _polygon_for_render(
            match,
            polygon_transform=polygon_transform,
            frame_shape=frame_shape,
        )
        if render_item is None:
            continue

        polygon, unconfirmed = render_item
        if unconfirmed:
            unconfirmed_items.append(
                _OverlayItem(
                    polygon=polygon,
                    color=COLOR_UNCONFIRMED,
                    label=_unconfirmed_label_for_match(match),
                    compact_label=_compact_label_for_match(match),
                    status="unconfirmed",
                    priority=1,
                )
            )
        else:
            status = str(match.status)
            normal_items.append(
                _OverlayItem(
                    polygon=polygon,
                    color=_color_for_status(status),
                    label=_label_for_match(match),
                    compact_label=_compact_label_for_match(match),
                    status=status,
                    priority=_label_priority(status),
                )
            )

    if normal_items:
        _draw_normal_polygon_items(
            output,
            normal_items,
            thickness=thickness,
            label_font_size=label_font_size,
            label_padding=label_padding,
        )

    if unconfirmed_items:
        _draw_unconfirmed_polygon_items(
            output,
            unconfirmed_items,
            thickness=max(2, thickness - 1),
            label_font_size=label_font_size,
            label_padding=label_padding,
        )


def _polygon_for_render(
    match: SegmentMatch,
    *,
    polygon_transform: np.ndarray | None,
    frame_shape: tuple[int, int],
) -> tuple[list[list[float]], bool] | None:
    unconfirmed = False
    if match.status == "missing":
        polygon = match.expected_polygon
        unconfirmed = _is_unconfirmed_missing_projection(match)
        if polygon is None:
            polygon = _debug_shadow_polygon(match)
            unconfirmed = polygon is not None
    elif match.status == "ok":
        polygon = match.detected_polygon or match.expected_polygon or _bbox_polygon_for_match(match)
    elif match.status == "unmatched":
        polygon = match.expected_polygon or match.detected_polygon or _bbox_polygon_for_match(match)
    elif match.status == "extra":
        polygon = match.detected_polygon or _bbox_polygon_for_match(match)
    else:
        return None

    prepared = _prepare_polygon_for_render(
        polygon,
        polygon_transform=polygon_transform,
        frame_shape=frame_shape,
        relaxed_expected=match.status in {"missing", "unmatched"},
    )
    if prepared is None:
        return None
    return prepared, unconfirmed


def _prepare_polygon_for_render(
    polygon: list[list[float]] | None,
    *,
    polygon_transform: np.ndarray | None,
    frame_shape: tuple[int, int],
    relaxed_expected: bool = False,
) -> list[list[float]] | None:
    if polygon is None or len(polygon) < 3:
        return None

    if polygon_transform is not None:
        polygon = _transform_polygon(polygon, polygon_transform)
        if polygon is None or len(polygon) < 3:
            return None

    return _sanitize_polygon_for_render(
        polygon,
        frame_shape=frame_shape,
        relaxed_expected=relaxed_expected,
    )


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


def _label_priority(status: str) -> int:
    if status == "missing":
        return 0
    if status == "unmatched":
        return 1
    if status == "extra":
        return 2
    if status == "unconfirmed":
        return 3
    return 10


def _short_name(name: str, *, limit: int = 22) -> str:
    if len(name) > limit:
        return name[: max(1, limit - 1)] + "…"
    return name


def _label_for_match(match: SegmentMatch) -> str:
    name = _short_name(match.name)

    if match.status == "missing":
        return f"{name} · отсутствует"
    if match.status == "unmatched":
        return f"{name} · проверить"
    if match.status == "extra":
        if match.confidence is not None:
            return f"{name} · лишнее {int(match.confidence * 100)}%"
        return f"{name} · лишнее"
    if match.confidence is not None:
        return f"{name} {int(match.confidence * 100)}%"
    return name


def _compact_label_for_match(match: SegmentMatch) -> str | None:
    name = _short_name(match.name, limit=16)
    if match.status == "ok" and match.confidence is not None:
        return f"{name} {int(match.confidence * 100)}%"
    if match.status == "ok":
        return name
    return None


def _unconfirmed_label_for_match(match: SegmentMatch) -> str:
    return f"{_short_name(match.name)} · зона?"


def _is_unconfirmed_missing_projection(match: SegmentMatch) -> bool:
    if match.status != "missing" or not isinstance(match.debug, dict):
        return False

    safety = str(match.debug.get("missing_polygon_projection_safety") or "")
    if safety == "unsafe_hidden":
        return True

    action = str(match.debug.get("missing_polygon_candidate_recommended_action") or "")
    if action in {
        "keep_hidden_or_require_more_evidence",
        "prefer_uncertain_or_hidden_in_ui",
        "render_as_unconfirmed_expected_zone",
    }:
        return True

    projection = str(
        match.debug.get("missing_polygon_projection")
        or match.debug.get("projection")
        or ""
    )
    return projection in {
        "expected_slot",
        "expected_slot_global_fallback",
        "expected_slot_global_fallback_hidden_release",
        "expected_slot_agreement_hidden_release",
        "none",
    }


def _debug_shadow_polygon(match: SegmentMatch) -> list[list[float]] | None:
    if not isinstance(match.debug, dict):
        return None
    if not match.debug.get("missing_polygon_hidden_shadow_available"):
        return None
    return _debug_polygon_points(match.debug.get("missing_polygon_hidden_shadow_polygon"))


def _debug_polygon_points(value: object) -> list[list[float]] | None:
    if not isinstance(value, list) or len(value) < 3:
        return None
    points: list[list[float]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) < 2:
            return None
        try:
            x = float(point[0])
            y = float(point[1])
        except (TypeError, ValueError):
            return None
        if not np.isfinite([x, y]).all():
            return None
        points.append([x, y])
    return points


def _bbox_polygon_for_match(match: SegmentMatch) -> list[list[float]] | None:
    bbox = match.detected_bbox
    if not isinstance(bbox, dict):
        return None

    for keys in (
        ("x1", "y1", "x2", "y2"),
        ("xmin", "ymin", "xmax", "ymax"),
        ("left", "top", "right", "bottom"),
    ):
        if all(key in bbox for key in keys):
            try:
                x1, y1, x2, y2 = (float(bbox[key]) for key in keys)
            except (TypeError, ValueError):
                return None
            break
    else:
        if all(key in bbox for key in ("x", "y", "width", "height")):
            try:
                x1 = float(bbox["x"])
                y1 = float(bbox["y"])
                x2 = x1 + float(bbox["width"])
                y2 = y1 + float(bbox["height"])
            except (TypeError, ValueError):
                return None
        else:
            return None

    if not np.isfinite([x1, y1, x2, y2]).all():
        return None
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 - x1 < 2.0 or y2 - y1 < 2.0:
        return None
    return [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]


def _label_anchor(polygon: list[list[float]]) -> tuple[int, int]:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return (int(round((min(xs) + max(xs)) / 2.0)), int(round(min(ys))))


def _polygon_center(polygon: list[list[float]]) -> tuple[int, int]:
    xs = [float(p[0]) for p in polygon]
    ys = [float(p[1]) for p in polygon]
    return (int(round(sum(xs) / len(xs))), int(round(sum(ys) / len(ys))))


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
    text_w = int(round(bbox[2] - bbox[0]))
    text_h = int(round(bbox[3] - bbox[1]))
    border = max(1, round(font_size * 0.10))
    box_w = text_w + padding * 2 + border * 2
    box_h = text_h + padding * 2 + border * 2

    plate = Image.new("RGB", (box_w, box_h), (0, 0, 0))
    draw = ImageDraw.Draw(plate)
    draw.rectangle(
        [border, border, box_w - border - 1, box_h - border - 1],
        fill=(color[2], color[1], color[0]),
    )
    draw.text(
        (padding + border, padding + border - bbox[1]),
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
    rect = _label_rect_for_anchor(image.shape[:2], anchor, w, h)
    _paste_label_plate(image, plate_bgr, rect)


def _draw_labels_avoid_overlap(
    image: np.ndarray,
    items: list[_OverlayItem],
    *,
    label_font_size: int,
    label_padding: int,
) -> None:
    dense = len(items) >= DENSE_LABEL_COUNT
    occupied: list[tuple[int, int, int, int]] = []
    ok_labels_used = 0
    ordered_items = sorted(
        items,
        key=lambda item: (
            item.priority,
            _label_anchor(item.polygon)[1],
            _label_anchor(item.polygon)[0],
        ),
    )
    for item in ordered_items:
        label = item.label
        font_size = label_font_size
        padding = label_padding
        if dense and item.status == "ok":
            if ok_labels_used >= MAX_FULL_OK_LABELS_DENSE:
                continue
            if item.compact_label:
                label = item.compact_label
                font_size = max(MIN_LABEL_FONT_SIZE, label_font_size - 2)
                padding = max(3, label_padding - 1)
            ok_labels_used += 1

        _draw_label_avoiding(
            image,
            label,
            _label_anchor(item.polygon),
            _polygon_center(item.polygon),
            item.color,
            occupied=occupied,
            font_size=font_size,
            padding=padding,
        )


def _draw_label_avoiding(
    image: np.ndarray,
    text: str,
    anchor: tuple[int, int],
    connector_target: tuple[int, int],
    color: tuple[int, int, int],
    *,
    occupied: list[tuple[int, int, int, int]],
    font_size: int,
    padding: int,
) -> None:
    plate_bgr = _render_label_plate(text, color, font_size, padding)
    h, w = plate_bgr.shape[:2]
    candidates = _label_candidate_rects(image.shape[:2], anchor, w, h)

    best_rect: tuple[int, int, int, int] | None = None
    best_score: float | None = None
    for rect in candidates:
        overlap_count = sum(1 for other in occupied if _rects_overlap_with_gap(rect, other, LABEL_MIN_GAP))
        overlap_area = sum(_rect_overlap_area(rect, other) for other in occupied)
        distance = _rect_anchor_distance(rect, anchor)
        score = overlap_count * 1_000_000.0 + overlap_area * 100.0 + distance
        if best_score is None or score < best_score:
            best_score = score
            best_rect = rect
        if overlap_count == 0:
            break

    if best_rect is None:
        return

    _draw_label_connector(image, best_rect, connector_target, color)
    _paste_label_plate(image, plate_bgr, best_rect)
    occupied.append(best_rect)


def _label_candidate_rects(
    frame_shape: tuple[int, int],
    anchor: tuple[int, int],
    w: int,
    h: int,
) -> list[tuple[int, int, int, int]]:
    ax, ay = anchor
    step_x = max(12, round(w * 0.55))
    step_y = max(8, h + 6)
    offsets: list[tuple[int, int]] = [(0, 0), (0, step_y), (0, -step_y)]
    for ring in range(1, LABEL_CANDIDATE_RINGS + 1):
        offsets.extend(
            [
                (-step_x * ring, 0),
                (step_x * ring, 0),
                (-step_x * ring, step_y * ring),
                (step_x * ring, step_y * ring),
                (-step_x * ring, -step_y * ring),
                (step_x * ring, -step_y * ring),
                (0, step_y * (ring + 1)),
                (0, -step_y * (ring + 1)),
            ]
        )
    # Additional left/right lanes help camera-wide industrial frames.
    offsets.extend(
        [
            (-step_x * 2, step_y),
            (step_x * 2, step_y),
            (-step_x * 3, 0),
            (step_x * 3, 0),
            (-step_x * 3, step_y * 2),
            (step_x * 3, step_y * 2),
        ]
    )
    rects: list[tuple[int, int, int, int]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for dx, dy in offsets:
        rect = _label_rect_for_anchor(frame_shape, (ax + dx, ay + dy), w, h)
        if rect not in seen:
            seen.add(rect)
            rects.append(rect)
    return rects


def _label_rect_for_anchor(
    frame_shape: tuple[int, int],
    anchor: tuple[int, int],
    w: int,
    h: int,
) -> tuple[int, int, int, int]:
    frame_h, frame_w = frame_shape
    x, y = anchor
    box_x = max(0, min(int(round(x - w / 2)), max(0, frame_w - w)))
    box_y = int(y) - h - 3
    if box_y < 0:
        box_y = int(y) + 3
    box_y = max(0, min(box_y, max(0, frame_h - h)))
    return box_x, box_y, box_x + w, box_y + h


def _draw_label_connector(
    image: np.ndarray,
    rect: tuple[int, int, int, int],
    anchor: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x1, y1, x2, y2 = rect
    ax = max(0, min(int(anchor[0]), image.shape[1] - 1))
    ay = max(0, min(int(anchor[1]), image.shape[0] - 1))
    if ay >= y2:
        label_point = (int((x1 + x2) / 2), y2 - 1)
    elif ay <= y1:
        label_point = (int((x1 + x2) / 2), y1)
    else:
        label_point = (x1 if ax < x1 else x2 - 1, int((y1 + y2) / 2))
    cv2.line(image, label_point, (ax, ay), COLOR_HALO, thickness=3, lineType=cv2.LINE_AA)
    cv2.line(image, label_point, (ax, ay), color, thickness=1, lineType=cv2.LINE_AA)
    cv2.circle(image, (ax, ay), 3, COLOR_HALO, -1, lineType=cv2.LINE_AA)
    cv2.circle(image, (ax, ay), 2, color, -1, lineType=cv2.LINE_AA)


def _paste_label_plate(
    image: np.ndarray,
    plate_bgr: np.ndarray,
    rect: tuple[int, int, int, int],
) -> None:
    x1, y1, x2, y2 = rect
    y2 = min(y2, image.shape[0])
    x2 = min(x2, image.shape[1])
    if y2 > y1 and x2 > x1:
        image[y1:y2, x1:x2] = plate_bgr[: y2 - y1, : x2 - x1]


def _rects_overlap_with_gap(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
    gap: int,
) -> bool:
    return not (
        a[2] + gap <= b[0]
        or b[2] + gap <= a[0]
        or a[3] + gap <= b[1]
        or b[3] + gap <= a[1]
    )


def _rect_overlap_area(
    a: tuple[int, int, int, int],
    b: tuple[int, int, int, int],
) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def _rect_anchor_distance(
    rect: tuple[int, int, int, int],
    anchor: tuple[int, int],
) -> float:
    cx = (rect[0] + rect[2]) / 2.0
    cy = (rect[1] + rect[3]) / 2.0
    return float(np.hypot(cx - anchor[0], cy - anchor[1]))


def _sanitize_polygon_for_render(
    polygon: list[list[float]],
    *,
    frame_shape: tuple[int, int],
    relaxed_expected: bool = False,
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

    if relaxed_expected:
        if (
            float(np.max(xs)) < 0.0
            or float(np.min(xs)) > float(frame_w)
            or float(np.max(ys)) < 0.0
            or float(np.min(ys)) > float(frame_h)
        ):
            return None
        points = points.copy()
        points[:, 0] = np.clip(points[:, 0], 0, max(0, frame_w - 1))
        points[:, 1] = np.clip(points[:, 1], 0, max(0, frame_h - 1))
        points = _remove_near_duplicate_points(points)
        if points.shape[0] < 3:
            return None
        xs = points[:, 0]
        ys = points[:, 1]
    else:
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

    if not relaxed_expected:
        if bbox_w > frame_w * POLYGON_MAX_BBOX_SIDE_FRACTION:
            return None
        if bbox_h > frame_h * POLYGON_MAX_BBOX_SIDE_FRACTION:
            return None
        if not _has_enough_visible_bbox(points, frame_w=frame_w, frame_h=frame_h):
            return None
        if _max_edge_length(points) > np.hypot(frame_w, frame_h) * POLYGON_MAX_EDGE_FRACTION:
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


def _has_enough_visible_bbox(points: np.ndarray, *, frame_w: int, frame_h: int) -> bool:
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
    return int(np.clip(round(base * 0.013), MIN_LABEL_FONT_SIZE, MAX_LABEL_FONT_SIZE))


def _line_thickness(frame_shape: tuple[int, int]) -> int:
    frame_h, frame_w = frame_shape
    base = max(frame_w, frame_h)
    return int(np.clip(round(base * 0.0030), MIN_LINE_THICKNESS, MAX_LINE_THICKNESS))


def _label_padding(font_size: int) -> int:
    return max(4, round(font_size * 0.26))


def _draw_unconfirmed_polygon_items(
    output: np.ndarray,
    items: list[_OverlayItem],
    *,
    thickness: int,
    label_font_size: int,
    label_padding: int,
) -> None:
    for item in items:
        pts = np.array(item.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [pts], True, COLOR_HALO, thickness + 2, lineType=cv2.LINE_AA)
        cv2.polylines(output, [pts], True, item.color, thickness, lineType=cv2.LINE_AA)
    _draw_labels_avoid_overlap(
        output,
        items,
        label_font_size=label_font_size,
        label_padding=label_padding,
    )


def _draw_normal_polygon_items(
    output: np.ndarray,
    items: list[_OverlayItem],
    *,
    thickness: int,
    label_font_size: int,
    label_padding: int,
) -> None:
    ok_items = [item for item in items if item.status == "ok"]
    problem_items = [item for item in items if item.status != "ok"]

    for subset, alpha in ((ok_items, ALPHA_FILL_OK), (problem_items, ALPHA_FILL_PROBLEM)):
        if not subset:
            continue
        fill_overlay = output.copy()
        for item in subset:
            pts = np.array(item.polygon, dtype=np.int32).reshape(-1, 1, 2)
            cv2.fillPoly(fill_overlay, [pts], item.color)
        cv2.addWeighted(fill_overlay, alpha, output, 1 - alpha, 0, output)

    for item in items:
        pts = np.array(item.polygon, dtype=np.int32).reshape(-1, 1, 2)
        cv2.polylines(output, [pts], True, COLOR_HALO, thickness + 3, lineType=cv2.LINE_AA)
        cv2.polylines(output, [pts], True, item.color, thickness, lineType=cv2.LINE_AA)

    _draw_labels_avoid_overlap(
        output,
        items,
        label_font_size=label_font_size,
        label_padding=label_padding,
    )
