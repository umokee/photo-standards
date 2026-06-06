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
    SUPERPOINT_OFFLINE_MAX_KEYPOINTS,
    SUPERPOINT_OFFLINE_MAX_SIDE,
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
        alignment_max_side=SUPERPOINT_OFFLINE_MAX_SIDE,
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
    alignment_max_side: int | None = SUPERPOINT_OFFLINE_MAX_SIDE,
    alignment_max_keypoints: int = SUPERPOINT_OFFLINE_MAX_KEYPOINTS,
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
            )
            profile["inspect_alignment_ms"] = _elapsed_ms(started_at)
        else:
            alignment = align_frame(
                context=context,
                frame=frame,
                max_side=alignment_max_side,
                max_keypoints=alignment_max_keypoints,
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
    alignment_max_side: int | None = SUPERPOINT_OFFLINE_MAX_SIDE,
    alignment_max_keypoints: int = SUPERPOINT_OFFLINE_MAX_KEYPOINTS,
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

    projection_data = _build_projection_data(
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
                yolo_anchor_rescue=True,
            )
            profile["inspect_matching_ms"] = _elapsed_ms(started_at)
        else:
            matches = match_segments(
                expected,
                detection_result.detections,
                alignment.homography,
                frame_size=(frame.shape[1], frame.shape[0]),
                projection_data=projection_data,
                yolo_anchor_rescue=True,
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
) -> tuple[FrameAlignment, float]:
    started_at = time.perf_counter()
    alignment = align_frame(
        context=context,
        frame=frame,
        max_side=max_side,
        max_keypoints=max_keypoints,
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
    )


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
