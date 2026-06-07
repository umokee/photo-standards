from __future__ import annotations

import time
from collections import defaultdict
from concurrent.futures import Executor, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from app.config import settings
from infra.storage.file_storage import resolve_storage_path
from modules.core.standards.reference_constants import (
    SUPERPOINT_PHOTO_GRID_COLS,
    SUPERPOINT_PHOTO_GRID_ROWS,
    SUPERPOINT_PHOTO_MAX_KEYPOINTS,
    SUPERPOINT_PHOTO_MAX_SIDE,
)
from modules.yolo.inspection.adapters.context import InspectionContext
from modules.yolo.inspection.adapters.features import align_frame, load_image
from modules.yolo.inspection.adapters.yolo import run_inference
from modules.yolo.inspection.constants import inspections as inspections_constants
from modules.yolo.inspection.domain.alignment import (
    FrameAlignment,
    LocalProjectionData,
    alignment_message,
    failed_alignment,
)
from modules.yolo.inspection.domain.matcher import (
    all_ok,
    build_expected_segments,
    build_missing_matches,
    match_segments,
    match_segments_by_count,
    summarize,
)
from modules.yolo.inspection.domain.yolo_anchor_pose import (
    estimate_yolo_anchor_alignment,
)
from modules.yolo.inspection.domain.overlay import render_overlay
from modules.yolo.inspection.domain.types import (
    ExpectedSegment,
    SegmentMatch,
    YoloDetection,
)

VERIFICATION_MODE_ALIGNMENT = "alignment"
VERIFICATION_MODE_YOLO_COUNT = "yolo_count"


@dataclass(slots=True)
class DetectionResult:
    detections: list[YoloDetection]
    raw_class_counts: dict[str, int]


@dataclass(slots=True)
class InspectionFrameResult:
    image_path: Path | None
    detections: list[YoloDetection]
    expected_segments: list[ExpectedSegment]
    matches: list[SegmentMatch]

    total: int
    matched: int
    missing: list[str]
    inspection_status: str

    alignment: FrameAlignment
    alignment_message: str
    raw_class_counts: dict[str, int]
    rendered_frame: np.ndarray | None = None
    profile: dict[str, float] = field(default_factory=dict)
    verification_mode: str = "alignment"


def inspect_image_path(
    *,
    context: InspectionContext,
    image_path: Path,
    render: bool = False,
    profile_enabled: bool = False,
) -> InspectionFrameResult:
    frame = load_image(image_path)
    return inspect_frame(
        context=context,
        frame=frame,
        image_path=image_path,
        render=render,
        profile_enabled=profile_enabled,
        alignment_max_side=SUPERPOINT_PHOTO_MAX_SIDE,
        alignment_max_keypoints=SUPERPOINT_PHOTO_MAX_KEYPOINTS,
        yolo_conf=settings.YOLO_CONF_THRESHOLD,
    )


def inspect_frame(
    *,
    context: InspectionContext,
    frame: np.ndarray,
    image_path: Path | None = None,
    alignment: FrameAlignment | None = None,
    alignment_display_message: str | None = None,
    render: bool = False,
    fps: float | None = None,
    skip_detection_when_alignment_failed: bool = False,
    expected_segments: list[ExpectedSegment] | None = None,
    profile_enabled: bool = False,
    alignment_max_side: int | None = SUPERPOINT_PHOTO_MAX_SIDE,
    alignment_max_keypoints: int = SUPERPOINT_PHOTO_MAX_KEYPOINTS,
    alignment_selection_grid: tuple[int, int] | None = (
        SUPERPOINT_PHOTO_GRID_ROWS,
        SUPERPOINT_PHOTO_GRID_COLS,
    ),
    yolo_conf: float | None = None,
) -> InspectionFrameResult:
    profile: dict[str, float] = {}
    verification_mode = _verification_mode()

    expected = _resolve_expected_segments(
        context=context,
        expected_segments=expected_segments,
        profile=profile,
        profile_enabled=profile_enabled,
    )

    if verification_mode == VERIFICATION_MODE_YOLO_COUNT:
        alignment = _skipped_alignment()
        message = alignment_display_message or "Совмещение отключено: YOLO count"
        if profile_enabled:
            profile["inspect_alignment_ms"] = 0.0
            started_at = time.perf_counter()
            detection_result = _detect_segments(
                context=context,
                image=frame,
                conf=yolo_conf,
            )
            profile["inspect_detection_ms"] = _elapsed_ms(started_at)
        else:
            detection_result = _detect_segments(
                context=context,
                image=frame,
                conf=yolo_conf,
            )

        return _compose_frame_result(
            context=context,
            frame=frame,
            image_path=image_path,
            expected=expected,
            detection_result=detection_result,
            alignment=alignment,
            alignment_display_message=message,
            render=render,
            fps=fps,
            profile=profile,
            profile_enabled=profile_enabled,
            verification_mode=verification_mode,
        )

    if alignment is None:
        if profile_enabled:
            started_at = time.perf_counter()
            alignment = align_frame(
                context=context,
                frame=frame,
                max_side=alignment_max_side,
                max_keypoints=alignment_max_keypoints,
                selection_grid=alignment_selection_grid,
            )
            profile["inspect_alignment_ms"] = _elapsed_ms(started_at)
        else:
            alignment = align_frame(
                context=context,
                frame=frame,
                max_side=alignment_max_side,
                max_keypoints=alignment_max_keypoints,
                selection_grid=alignment_selection_grid,
            )

    message = alignment_display_message or alignment_message(alignment)

    if skip_detection_when_alignment_failed and not alignment.is_success:
        detection_result = DetectionResult(detections=[], raw_class_counts={})
        if profile_enabled:
            profile["inspect_detection_ms"] = 0.0
    elif profile_enabled:
        started_at = time.perf_counter()
        detection_result = _detect_segments(
            context=context,
            image=frame,
            conf=yolo_conf,
        )
        profile["inspect_detection_ms"] = _elapsed_ms(started_at)
    else:
        detection_result = _detect_segments(
            context=context,
            image=frame,
            conf=yolo_conf,
        )

    return _compose_frame_result(
        context=context,
        frame=frame,
        image_path=image_path,
        expected=expected,
        detection_result=detection_result,
        alignment=alignment,
        alignment_display_message=message,
        render=render,
        fps=fps,
        profile=profile,
        profile_enabled=profile_enabled,
        verification_mode=verification_mode,
    )


def inspect_frame_parallel(
    *,
    context: InspectionContext,
    frame: np.ndarray,
    image_path: Path | None = None,
    alignment: FrameAlignment | None = None,
    alignment_display_message: str | None = None,
    render: bool = False,
    fps: float | None = None,
    skip_detection_when_alignment_failed: bool = False,
    expected_segments: list[ExpectedSegment] | None = None,
    profile_enabled: bool = False,
    executor: Executor | None = None,
    alignment_max_side: int | None = SUPERPOINT_PHOTO_MAX_SIDE,
    alignment_max_keypoints: int = SUPERPOINT_PHOTO_MAX_KEYPOINTS,
    alignment_selection_grid: tuple[int, int] | None = (
        SUPERPOINT_PHOTO_GRID_ROWS,
        SUPERPOINT_PHOTO_GRID_COLS,
    ),
    yolo_conf: float | None = None,
) -> InspectionFrameResult:
    profile: dict[str, float] = {}
    verification_mode = _verification_mode()

    expected = _resolve_expected_segments(
        context=context,
        expected_segments=expected_segments,
        profile=profile,
        profile_enabled=profile_enabled,
    )

    if verification_mode == VERIFICATION_MODE_YOLO_COUNT:
        alignment = _skipped_alignment()
        message = alignment_display_message or "Совмещение отключено: YOLO count"
        if profile_enabled:
            profile["inspect_alignment_ms"] = 0.0
            started_at = time.perf_counter()
            detection_result = _detect_segments(
                context=context,
                image=frame,
                conf=yolo_conf,
            )
            profile["inspect_detection_ms"] = _elapsed_ms(started_at)
            profile["inspect_parallel_wait_ms"] = profile["inspect_detection_ms"]
        else:
            detection_result = _detect_segments(
                context=context,
                image=frame,
                conf=yolo_conf,
            )

        return _compose_frame_result(
            context=context,
            frame=frame,
            image_path=image_path,
            expected=expected,
            detection_result=detection_result,
            alignment=alignment,
            alignment_display_message=message,
            render=render,
            fps=fps,
            profile=profile,
            profile_enabled=profile_enabled,
            verification_mode=verification_mode,
        )

    owned_executor: ThreadPoolExecutor | None = None

    try:
        if alignment is None:
            active_executor: Executor
            if executor is None:
                owned_executor = ThreadPoolExecutor(
                    max_workers=2,
                    thread_name_prefix="inspect-parallel",
                )
                active_executor = owned_executor
            else:
                active_executor = executor

            started_at = time.perf_counter()

            alignment_future = active_executor.submit(
                _timed_align_frame,
                context,
                frame.copy(),
                alignment_max_side,
                alignment_max_keypoints,
                alignment_selection_grid,
            )
            detection_future = active_executor.submit(
                _timed_detect_segments,
                context,
                frame.copy(),
                yolo_conf,
            )

            alignment, alignment_ms = alignment_future.result()
            detection_result, detection_ms = detection_future.result()

            if profile_enabled:
                profile["inspect_alignment_ms"] = alignment_ms
                profile["inspect_detection_ms"] = detection_ms
                profile["inspect_parallel_wait_ms"] = _elapsed_ms(started_at)

        else:
            message = alignment_display_message or alignment_message(alignment)

            if skip_detection_when_alignment_failed and not alignment.is_success:
                detection_result = DetectionResult(detections=[], raw_class_counts={})
                if profile_enabled:
                    profile["inspect_detection_ms"] = 0.0
            elif profile_enabled:
                started_at = time.perf_counter()
                detection_result = _detect_segments(
                    context=context,
                    image=frame,
                    conf=yolo_conf,
                )
                profile["inspect_detection_ms"] = _elapsed_ms(started_at)
            else:
                detection_result = _detect_segments(
                    context=context,
                    image=frame,
                    conf=yolo_conf,
                )

            return _compose_frame_result(
                context=context,
                frame=frame,
                image_path=image_path,
                expected=expected,
                detection_result=detection_result,
                alignment=alignment,
                alignment_display_message=message,
                render=render,
                fps=fps,
                profile=profile,
                profile_enabled=profile_enabled,
                verification_mode=verification_mode,
            )
    finally:
        if owned_executor is not None:
            owned_executor.shutdown(wait=False, cancel_futures=True)

    message = alignment_display_message or alignment_message(alignment)

    return _compose_frame_result(
        context=context,
        frame=frame,
        image_path=image_path,
        expected=expected,
        detection_result=detection_result,
        alignment=alignment,
        alignment_display_message=message,
        render=render,
        fps=fps,
        profile=profile,
        profile_enabled=profile_enabled,
        verification_mode=verification_mode,
    )


def _select_effective_alignment(
    *,
    context: InspectionContext,
    expected: list[ExpectedSegment],
    detections: list[YoloDetection],
    alignment: FrameAlignment,
    alignment_display_message: str,
    frame: np.ndarray,
    profile: dict[str, float],
    profile_enabled: bool,
) -> tuple[FrameAlignment, str]:
    """Choose the final pose source.

    The previous "auto" behaved like "YOLO exists -> trust YOLO-anchor pose".
    That can be worse than the LightGlue feature fallback when synthetic/real
    detections are noisy or repeated. Auto must be an arbiter: compute both
    candidates when possible and keep the safest confirmed pose.
    """

    mode = str(getattr(settings, "INSPECTION_YOLO_ANCHOR_POSE_MODE", "auto") or "auto")
    if mode == "off" or not detections:
        _merge_alignment_extra_debug(
            alignment,
            {
                "pose_arbiter": {
                    "mode": mode,
                    "selected": "feature_slot_fallback",
                    "reason": "yolo_anchor_disabled_or_no_detections",
                    "yolo_detection_count": len(detections),
                }
            },
        )
        return alignment, alignment_display_message

    fallback_projection = _build_projection_data(
        context=context,
        alignment=alignment,
        frame=frame,
    )
    fallback_can_project = _can_project_segments(
        alignment=alignment,
        projection_data=fallback_projection,
    )
    fallback_strong = _is_strong_feature_pose(alignment) and fallback_can_project

    if mode == "fallback" and fallback_can_project:
        _merge_alignment_extra_debug(
            alignment,
            {
                "pose_arbiter": {
                    "mode": mode,
                    "selected": "feature_slot_fallback",
                    "reason": "fallback_mode_keeps_lightglue_feature_pose",
                    "fallback_strong": fallback_strong,
                    "yolo_detection_count": len(detections),
                }
            },
        )
        return alignment, alignment_display_message

    started_at = time.perf_counter()
    anchor_debug: dict[str, object] = {}
    try:
        anchor_alignment = estimate_yolo_anchor_alignment(
            expected,
            detections,
            frame_size=(frame.shape[1], frame.shape[0]),
            debug_out=anchor_debug,
        )
    except TypeError:
        # Compatibility with older yolo_anchor_pose.py before debug_out existed.
        anchor_alignment = estimate_yolo_anchor_alignment(
            expected,
            detections,
            frame_size=(frame.shape[1], frame.shape[0]),
        )
    if profile_enabled:
        profile["inspect_yolo_anchor_pose_ms"] = _elapsed_ms(started_at)

    if anchor_debug:
        _merge_alignment_extra_debug(alignment, anchor_debug)

    if anchor_alignment is None or not anchor_alignment.is_success:
        _merge_alignment_extra_debug(
            alignment,
            {
                "pose_arbiter": {
                    "mode": mode,
                    "selected": "feature_slot_fallback",
                    "reason": "yolo_anchor_pose_rejected",
                    "fallback_strong": fallback_strong,
                    "fallback_can_project": fallback_can_project,
                    "yolo_detection_count": len(detections),
                }
            },
        )
        return alignment, alignment_display_message

    anchor_strong = _is_strong_yolo_anchor_pose(anchor_alignment, anchor_debug)
    disagreement_px = _median_pose_disagreement_px(
        expected,
        alignment.homography,
        anchor_alignment.homography,
    )

    if mode == "prefer":
        _merge_alignment_extra_debug(
            anchor_alignment,
            {
                "pose_arbiter": {
                    "mode": mode,
                    "selected": "yolo_anchor_pose",
                    "reason": "prefer_mode_uses_accepted_yolo_anchor_pose",
                    "anchor_strong": anchor_strong,
                    "fallback_strong": fallback_strong,
                    "pose_disagreement_px": disagreement_px,
                    "yolo_detection_count": len(detections),
                }
            },
        )
        return anchor_alignment, "Совмещение по видимым YOLO-объектам"

    selected, message, reason = _choose_auto_pose_candidate(
        fallback_alignment=alignment,
        fallback_message=alignment_display_message,
        anchor_alignment=anchor_alignment,
        fallback_strong=fallback_strong,
        fallback_can_project=fallback_can_project,
        anchor_strong=anchor_strong,
        disagreement_px=disagreement_px,
    )
    _merge_alignment_extra_debug(
        selected,
        {
            "pose_arbiter": {
                "mode": mode,
                "selected": selected.method,
                "reason": reason,
                "anchor_strong": anchor_strong,
                "fallback_strong": fallback_strong,
                "fallback_can_project": fallback_can_project,
                "pose_disagreement_px": disagreement_px,
                "yolo_detection_count": len(detections),
            }
        },
    )
    return selected, message


def _choose_auto_pose_candidate(
    *,
    fallback_alignment: FrameAlignment,
    fallback_message: str,
    anchor_alignment: FrameAlignment,
    fallback_strong: bool,
    fallback_can_project: bool,
    anchor_strong: bool,
    disagreement_px: float | None,
) -> tuple[FrameAlignment, str, str]:
    # If LightGlue/fallback cannot project a scene, the accepted YOLO-anchor
    # pose is the only usable candidate.
    if not fallback_can_project or not fallback_alignment.is_success:
        return (
            anchor_alignment,
            "Совмещение по видимым YOLO-объектам",
            "fallback_unavailable_yolo_anchor_used",
        )

    # If both are usable but disagree a lot, prefer the strong LightGlue pose.
    # This fixes the observed auto regression where a noisy accepted YOLO pose
    # overrode an actually better feature fallback.
    if disagreement_px is not None and disagreement_px > 14.0:
        if fallback_strong:
            return (
                fallback_alignment,
                fallback_message,
                "feature_fallback_strong_yolo_disagrees",
            )
        if not anchor_strong:
            return (
                fallback_alignment,
                fallback_message,
                "both_candidates_weak_or_disagree_keep_safer_fallback",
            )

    # If YOLO is clearly strong and either fallback is weak or both agree,
    # prefer YOLO-anchor pose for multi-object missing cases.
    if anchor_strong and (not fallback_strong or disagreement_px is None or disagreement_px <= 8.0):
        return (
            anchor_alignment,
            "Совмещение по видимым YOLO-объектам",
            "yolo_anchor_strong_and_consistent",
        )

    # Default safe auto behavior: do not let a merely accepted YOLO candidate
    # replace a projectable LightGlue fallback.
    return (
        fallback_alignment,
        fallback_message,
        "feature_fallback_preferred_over_weak_yolo_anchor",
    )


def _is_strong_feature_pose(alignment: FrameAlignment) -> bool:
    if not alignment.is_success or alignment.homography is None:
        return False
    raw = int(alignment.raw_match_count or 0)
    inliers = int(alignment.inlier_count or 0)
    ratio = inliers / max(1, raw)
    if raw < 24 or inliers < 14 or ratio < 0.24:
        return False
    if alignment.median_error is not None and float(alignment.median_error) > 9.0:
        return False
    return True


def _is_strong_yolo_anchor_pose(
    alignment: FrameAlignment,
    anchor_debug: dict[str, object] | None,
) -> bool:
    if not alignment.is_success or alignment.homography is None:
        return False
    payload = _yolo_anchor_payload(alignment, anchor_debug)
    if payload and payload.get("accepted") is False:
        return False

    anchor_count = _int_payload(payload, "anchor_count", alignment.reference_feature_count)
    inlier_anchors = _int_payload(payload, "inlier_anchor_count", None)
    point_ratio = _float_payload(payload, "point_inlier_ratio", None)
    median_error = _float_payload(payload, "median_center_error_px", alignment.median_error)
    p90_error = _float_payload(payload, "p90_center_error_px", None)
    spread = _float_payload(payload, "anchor_spread_score", None)
    condition = _float_payload(payload, "homography_condition", None)

    if anchor_count is not None and anchor_count < 3:
        # Two anchors can form a transform but are fragile with repeated parts.
        return False
    if inlier_anchors is not None and inlier_anchors < 3:
        return False
    if point_ratio is not None and point_ratio < 0.48:
        return False
    if median_error is not None and median_error > 8.0:
        return False
    if p90_error is not None and p90_error > 18.0:
        return False
    if spread is not None and spread < 0.16:
        return False
    if condition is not None and condition > 5.0e7:
        return False
    return True


def _yolo_anchor_payload(
    alignment: FrameAlignment,
    anchor_debug: dict[str, object] | None,
) -> dict[str, object]:
    if anchor_debug:
        candidate = anchor_debug.get("yolo_anchor_pose")
        if isinstance(candidate, dict):
            return candidate
    extra = getattr(alignment, "extra_debug", None)
    if isinstance(extra, dict):
        candidate = extra.get("yolo_anchor_pose")
        if isinstance(candidate, dict):
            return candidate
    return {}


def _int_payload(payload: dict[str, object], key: str, default: object) -> int | None:
    value = payload.get(key, default)
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_payload(payload: dict[str, object], key: str, default: object) -> float | None:
    value = payload.get(key, default)
    try:
        if value is None:
            return None
        out = float(value)
    except (TypeError, ValueError):
        return None
    if not np.isfinite(out):
        return None
    return out


def _median_pose_disagreement_px(
    expected: list[ExpectedSegment],
    fallback_homography: np.ndarray | None,
    anchor_homography: np.ndarray | None,
) -> float | None:
    if fallback_homography is None or anchor_homography is None:
        return None
    distances: list[float] = []
    for item in expected[:80]:
        center = _reference_polygon_center(item)
        if center is None:
            continue
        a = _project_xy(center, fallback_homography)
        b = _project_xy(center, anchor_homography)
        if a is None or b is None:
            continue
        distances.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
    if not distances:
        return None
    return float(np.median(distances))


def _reference_polygon_center(item: ExpectedSegment) -> tuple[float, float] | None:
    try:
        points = np.asarray(item.reference_polygon, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if points.ndim != 2 or points.shape[0] < 3 or points.shape[1] < 2:
        return None
    center = np.mean(points[:, :2], axis=0)
    if not np.isfinite(center).all():
        return None
    return float(center[0]), float(center[1])


def _project_xy(
    point: tuple[float, float],
    homography: np.ndarray,
) -> tuple[float, float] | None:
    try:
        matrix = np.asarray(homography, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (3, 3):
        return None
    source = np.asarray([point[0], point[1], 1.0], dtype=np.float64)
    projected = matrix @ source
    if abs(float(projected[2])) < 1e-9:
        return None
    x = float(projected[0] / projected[2])
    y = float(projected[1] / projected[2])
    if not np.isfinite([x, y]).all():
        return None
    return x, y


def _merge_alignment_extra_debug(
    alignment: FrameAlignment,
    extra: dict[str, object],
) -> None:
    current = getattr(alignment, "extra_debug", None)
    merged = dict(current) if isinstance(current, dict) else {}
    for key, value in extra.items():
        if key not in merged:
            merged[key] = value
        elif isinstance(merged[key], dict) and isinstance(value, dict):
            nested = dict(merged[key])
            nested.update(value)
            merged[key] = nested
    try:
        alignment.extra_debug = merged
    except (AttributeError, TypeError):
        # Older FrameAlignment without extra_debug: keep the arbiter fail-safe,
        # only the debug attachment is skipped.
        return


def _compose_frame_result(
    *,
    context: InspectionContext,
    frame: np.ndarray,
    image_path: Path | None,
    expected: list[ExpectedSegment],
    detection_result: DetectionResult,
    alignment: FrameAlignment,
    alignment_display_message: str,
    render: bool,
    fps: float | None,
    profile: dict[str, float],
    profile_enabled: bool,
    verification_mode: str,
) -> InspectionFrameResult:
    if verification_mode == VERIFICATION_MODE_YOLO_COUNT:
        if profile_enabled:
            started_at = time.perf_counter()
            matches = match_segments_by_count(expected, detection_result.detections)
            profile["inspect_matching_ms"] = _elapsed_ms(started_at)
        else:
            matches = match_segments_by_count(expected, detection_result.detections)

        _, matched, missing = summarize(matches)
        inspection_status = (
            inspections_constants.statuses.passed
            if all_ok(matches, expected_total=len(expected))
            else inspections_constants.statuses.failed
        )
        rendered_frame = None
        if render:
            if profile_enabled:
                started_at = time.perf_counter()
                rendered_frame = render_overlay(frame, matches, fps=fps)
                profile["inspect_render_ms"] = _elapsed_ms(started_at)
            else:
                rendered_frame = render_overlay(frame, matches, fps=fps)

        return InspectionFrameResult(
            image_path=image_path,
            detections=detection_result.detections,
            expected_segments=expected,
            matches=matches,
            total=len(expected),
            matched=matched,
            missing=missing,
            inspection_status=inspection_status,
            alignment=alignment,
            alignment_message=alignment_display_message,
            raw_class_counts=detection_result.raw_class_counts,
            rendered_frame=rendered_frame,
            profile=profile,
            verification_mode=verification_mode,
        )

    alignment, alignment_display_message = _select_effective_alignment(
        context=context,
        expected=expected,
        detections=detection_result.detections,
        alignment=alignment,
        alignment_display_message=alignment_display_message,
        frame=frame,
        profile=profile,
        profile_enabled=profile_enabled,
    )

    projection_data = _build_projection_data(
        context=context,
        alignment=alignment,
        frame=frame,
    )
    if _can_project_segments(alignment=alignment, projection_data=projection_data):
        if profile_enabled:
            started_at = time.perf_counter()
            matches = match_segments(
                expected,
                detection_result.detections,
                alignment.homography,
                frame_size=(frame.shape[1], frame.shape[0]),
                projection_data=projection_data,
            )
            profile["inspect_matching_ms"] = _elapsed_ms(started_at)
        else:
            matches = match_segments(
                expected,
                detection_result.detections,
                alignment.homography,
                frame_size=(frame.shape[1], frame.shape[0]),
                projection_data=projection_data,
            )

        _, matched, missing = summarize(matches)
        inspection_status = (
            inspections_constants.statuses.passed
            if all_ok(matches, expected_total=len(expected))
            else inspections_constants.statuses.failed
        )
    else:
        if profile_enabled:
            profile["inspect_matching_ms"] = 0.0

        if _failsafe_requires_confirmed_pose():
            matches = _build_scene_unconfirmed_matches(
                expected,
                alignment=alignment,
                reason="pose_not_confirmed",
            )
            missing = []
            alignment_display_message = _scene_unconfirmed_message(alignment)
        else:
            matches = build_missing_matches(
                expected,
                alignment.homography,
                frame_size=(frame.shape[1], frame.shape[0]),
                projection_data=projection_data,
            )
            missing = [item.name for item in expected]
        matched = 0
        inspection_status = inspections_constants.statuses.failed

    rendered_frame = None

    if render:
        if profile_enabled:
            started_at = time.perf_counter()
            rendered_frame = render_overlay(
                frame,
                matches,
                fps=fps,
                alignment_message=(
                    alignment_display_message if not alignment.is_success else None
                ),
            )
            profile["inspect_render_ms"] = _elapsed_ms(started_at)
        else:
            rendered_frame = render_overlay(
                frame,
                matches,
                fps=fps,
                alignment_message=(
                    alignment_display_message if not alignment.is_success else None
                ),
            )

    return InspectionFrameResult(
        image_path=image_path,
        detections=detection_result.detections,
        expected_segments=expected,
        matches=matches,
        total=len(expected),
        matched=matched,
        missing=missing,
        inspection_status=inspection_status,
        alignment=alignment,
        alignment_message=alignment_display_message,
        raw_class_counts=detection_result.raw_class_counts,
        rendered_frame=rendered_frame,
        profile=profile,
        verification_mode=verification_mode,
    )


def _failsafe_requires_confirmed_pose() -> bool:
    return bool(getattr(settings, "INSPECTION_FAILSAFE_REQUIRE_CONFIRMED_POSE", True))


def _scene_unconfirmed_message(alignment: FrameAlignment) -> str:
    if alignment.reason:
        return f"Сцена не подтверждена: {alignment.reason}"
    if alignment.stage:
        return f"Сцена не подтверждена: {alignment.stage}"
    return "Сцена не подтверждена: нет надёжной позы для переноса слотов"


def _build_scene_unconfirmed_matches(
    expected: list[ExpectedSegment],
    *,
    alignment: FrameAlignment,
    reason: str,
) -> list[SegmentMatch]:
    debug = {
        "reason": "Сцена не подтверждена, missing-полигон не рисуется",
        "reason_code": "scene_pose_unconfirmed",
        "projection": "none",
        "missing_polygon_projection": "hidden_unconfirmed_pose",
        "missing_polygon_projection_safety": "safe_hidden",
        "pose_failsafe": True,
        "pose_failsafe_reason": reason,
        "alignment_method": alignment.method,
        "alignment_status": alignment.status.value,
        "alignment_stage": alignment.stage,
        "alignment_reason": alignment.reason,
    }

    return [
        SegmentMatch(
            annotation_id=item.annotation_id,
            segment_class_id=item.segment_class_id,
            class_key=item.class_key,
            name=item.name,
            hue=item.hue,
            status="unmatched",
            iou=None,
            confidence=None,
            expected_polygon=None,
            detected_polygon=None,
            detected_bbox=None,
            debug=dict(debug),
        )
        for item in expected
    ]


def _merge_alignment_extra_debug(
    alignment: FrameAlignment,
    extra: dict[str, object],
) -> None:
    existing = alignment.extra_debug if isinstance(alignment.extra_debug, dict) else {}
    merged = dict(existing)
    for key, value in extra.items():
        if key not in merged:
            merged[key] = value
    alignment.extra_debug = merged


def _resolve_expected_segments(
    *,
    context: InspectionContext,
    expected_segments: list[ExpectedSegment] | None,
    profile: dict[str, float],
    profile_enabled: bool,
) -> list[ExpectedSegment]:
    if expected_segments is not None:
        return expected_segments

    if profile_enabled:
        started_at = time.perf_counter()
        expected = build_expected_segments(
            context.reference_image,
            {sc.id for sc in context.selected_classes},
        )
        profile["inspect_expected_ms"] = _elapsed_ms(started_at)
        return expected

    return build_expected_segments(
        context.reference_image,
        {sc.id for sc in context.selected_classes},
    )


def _timed_align_frame(
    context: InspectionContext,
    frame: np.ndarray,
    max_side: int | None,
    max_keypoints: int,
    selection_grid: tuple[int, int] | None,
) -> tuple[FrameAlignment, float]:
    started_at = time.perf_counter()
    alignment = align_frame(
        context=context,
        frame=frame,
        max_side=max_side,
        max_keypoints=max_keypoints,
        selection_grid=selection_grid,
    )
    return alignment, _elapsed_ms(started_at)


def _timed_detect_segments(
    context: InspectionContext,
    frame: np.ndarray,
    conf: float | None,
) -> tuple[DetectionResult, float]:
    started_at = time.perf_counter()
    detection_result = _detect_segments(context=context, image=frame, conf=conf)
    return detection_result, _elapsed_ms(started_at)


def _build_projection_data(
    *,
    context: InspectionContext,
    alignment: FrameAlignment,
    frame: np.ndarray,
) -> LocalProjectionData | None:
    reference_points = _prefer_points(
        alignment.reference_matches,
        alignment.reference_inliers,
    )
    frame_points = _prefer_points(
        alignment.frame_matches,
        alignment.frame_inliers,
    )

    if (
        alignment.homography is None
        and reference_points is None
        and frame_points is None
    ):
        return None

    return LocalProjectionData(
        global_homography=alignment.homography,
        reference_points=reference_points,
        frame_points=frame_points,
        frame_size=(frame.shape[1], frame.shape[0]),
        frame=frame,
        reference_frame=_load_projection_reference_frame(context),
        reference_feature_count=alignment.reference_feature_count,
        frame_feature_count=alignment.frame_feature_count,
        frame_max_keypoints=alignment.frame_max_keypoints,
        frame_keypoint_grid=alignment.frame_keypoint_grid,
        masked_alignment_used=alignment.masked_alignment_used,
        original_reference_feature_count=alignment.original_reference_feature_count,
        masked_reference_feature_count=alignment.masked_reference_feature_count,
    )




def _load_projection_reference_frame(context: InspectionContext) -> np.ndarray | None:
    try:
        return load_image(resolve_storage_path(context.reference_image.image_path))
    except Exception:
        return None

def _can_project_segments(
    *,
    alignment: FrameAlignment,
    projection_data: LocalProjectionData | None,
) -> bool:
    if alignment.homography is not None:
        return True

    if projection_data is None:
        return False

    return projection_data.has_local_points


def _verification_mode() -> str:
    if settings.INSPECTION_VERIFICATION_MODE == VERIFICATION_MODE_YOLO_COUNT:
        return VERIFICATION_MODE_YOLO_COUNT
    return VERIFICATION_MODE_ALIGNMENT


def _skipped_alignment() -> FrameAlignment:
    alignment = failed_alignment()
    alignment.method = "skipped"
    alignment.stage = "verification_mode"
    alignment.reason = "alignment skipped by INSPECTION_VERIFICATION_MODE=yolo_count"
    return alignment


def _prefer_points(
    primary: np.ndarray | None,
    fallback: np.ndarray | None,
) -> np.ndarray | None:
    if primary is not None and len(primary) > 0:
        return primary

    return fallback


def _elapsed_ms(started_at: float) -> float:
    return round((time.perf_counter() - started_at) * 1000, 1)


def _detect_segments(
    *,
    context: InspectionContext,
    image: np.ndarray | None = None,
    conf: float | None = None,
) -> DetectionResult:
    selected_class_keys = {str(item.id) for item in context.selected_classes}
    native_to_internal = context.native_to_internal

    raw_detections = run_inference(
        weights_path=resolve_storage_path(context.model.weights_path),
        image=image,
        conf=settings.YOLO_CONF_THRESHOLD if conf is None else conf,
        iou=settings.YOLO_NMS_IOU,
        imgsz=context.model.imgsz or None,
    )

    detections: list[YoloDetection] = []
    raw_counts: dict[str, int] = defaultdict(int)

    for detection in raw_detections:
        internal_key = native_to_internal.get(detection.class_key)
        if internal_key is None:
            continue

        raw_counts[internal_key] += 1

        if internal_key not in selected_class_keys:
            continue

        detections.append(
            YoloDetection(
                class_key=internal_key,
                confidence=detection.confidence,
                bbox=detection.bbox,
                polygon=detection.polygon,
            )
        )

    return DetectionResult(
        detections=detections,
        raw_class_counts=dict(raw_counts),
    )
