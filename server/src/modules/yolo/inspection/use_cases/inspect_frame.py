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
from modules.yolo.inspection.adapters.context import InspectionContext, ReferenceView
from modules.yolo.inspection.adapters.features import align_frame, load_image
from modules.yolo.inspection.adapters.yolo import run_inference
from modules.yolo.inspection.constants import inspections as inspections_constants
from modules.yolo.inspection.domain.alignment import (
    FrameAlignment,
    LocalProjectionData,
    alignment_message,
    failed_alignment,
    project_polygon,
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


@dataclass(slots=True)
class ReferenceSelectionResult:
    context: InspectionContext
    alignment: FrameAlignment
    scores: list[dict[str, object]]


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
    active_context = context

    if verification_mode == VERIFICATION_MODE_YOLO_COUNT:
        expected = _resolve_expected_segments(
            context=active_context,
            expected_segments=expected_segments,
            profile=profile,
            profile_enabled=profile_enabled,
        )
        alignment = _skipped_alignment()
        message = alignment_display_message or "Совмещение отключено: YOLO count"
        if profile_enabled:
            profile["inspect_alignment_ms"] = 0.0
            started_at = time.perf_counter()
            detection_result = _detect_segments(
                context=active_context,
                image=frame,
                conf=yolo_conf,
            )
            profile["inspect_detection_ms"] = _elapsed_ms(started_at)
        else:
            detection_result = _detect_segments(
                context=active_context,
                image=frame,
                conf=yolo_conf,
            )

        return _compose_frame_result(
            context=active_context,
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
            selection = _select_reference_context_and_align(
                context=context,
                frame=frame,
                max_side=alignment_max_side,
                max_keypoints=alignment_max_keypoints,
                selection_grid=alignment_selection_grid,
            )
            profile["inspect_alignment_ms"] = _elapsed_ms(started_at)
            if len(context.reference_views) > 1:
                profile["inspect_reference_candidates"] = float(len(selection.scores))
        else:
            selection = _select_reference_context_and_align(
                context=context,
                frame=frame,
                max_side=alignment_max_side,
                max_keypoints=alignment_max_keypoints,
                selection_grid=alignment_selection_grid,
            )
        active_context = selection.context
        alignment = selection.alignment
    else:
        active_context = context

    expected = _resolve_expected_segments(
        context=active_context,
        expected_segments=expected_segments,
        profile=profile,
        profile_enabled=profile_enabled,
    )

    message = alignment_display_message or alignment_message(alignment)

    if skip_detection_when_alignment_failed and not alignment.is_success:
        detection_result = DetectionResult(detections=[], raw_class_counts={})
        if profile_enabled:
            profile["inspect_detection_ms"] = 0.0
    elif profile_enabled:
        started_at = time.perf_counter()
        detection_result = _detect_segments(
            context=active_context,
            image=frame,
            conf=yolo_conf,
        )
        profile["inspect_detection_ms"] = _elapsed_ms(started_at)
    else:
        detection_result = _detect_segments(
            context=active_context,
            image=frame,
            conf=yolo_conf,
        )

    return _compose_frame_result(
        context=active_context,
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
    active_context = context

    if verification_mode == VERIFICATION_MODE_YOLO_COUNT:
        expected = _resolve_expected_segments(
            context=active_context,
            expected_segments=expected_segments,
            profile=profile,
            profile_enabled=profile_enabled,
        )
        alignment = _skipped_alignment()
        message = alignment_display_message or "Совмещение отключено: YOLO count"
        if profile_enabled:
            profile["inspect_alignment_ms"] = 0.0
            started_at = time.perf_counter()
            detection_result = _detect_segments(
                context=active_context,
                image=frame,
                conf=yolo_conf,
            )
            profile["inspect_detection_ms"] = _elapsed_ms(started_at)
            profile["inspect_parallel_wait_ms"] = profile["inspect_detection_ms"]
        else:
            detection_result = _detect_segments(
                context=active_context,
                image=frame,
                conf=yolo_conf,
            )

        return _compose_frame_result(
            context=active_context,
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
                _timed_select_reference_context_and_align,
                context,
                frame,
                alignment_max_side,
                alignment_max_keypoints,
                alignment_selection_grid,
            )
            detection_future = active_executor.submit(
                _timed_detect_segments,
                context,
                frame,
                yolo_conf,
            )

            selection, alignment_ms = alignment_future.result()
            active_context = selection.context
            alignment = selection.alignment
            detection_result, detection_ms = detection_future.result()

            if profile_enabled:
                profile["inspect_alignment_ms"] = alignment_ms
                profile["inspect_detection_ms"] = detection_ms
                profile["inspect_parallel_wait_ms"] = _elapsed_ms(started_at)
                if len(context.reference_views) > 1:
                    profile["inspect_reference_candidates"] = float(len(selection.scores))

        else:
            active_context = context
            message = alignment_display_message or alignment_message(alignment)

            if skip_detection_when_alignment_failed and not alignment.is_success:
                detection_result = DetectionResult(detections=[], raw_class_counts={})
                if profile_enabled:
                    profile["inspect_detection_ms"] = 0.0
            elif profile_enabled:
                started_at = time.perf_counter()
                detection_result = _detect_segments(
                    context=active_context,
                    image=frame,
                    conf=yolo_conf,
                )
                profile["inspect_detection_ms"] = _elapsed_ms(started_at)
            else:
                detection_result = _detect_segments(
                    context=active_context,
                    image=frame,
                    conf=yolo_conf,
                )

            expected = _resolve_expected_segments(
                context=active_context,
                expected_segments=expected_segments,
                profile=profile,
                profile_enabled=profile_enabled,
            )

            return _compose_frame_result(
                context=active_context,
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

    expected = _resolve_expected_segments(
        context=active_context,
        expected_segments=expected_segments,
        profile=profile,
        profile_enabled=profile_enabled,
    )
    message = alignment_display_message or alignment_message(alignment)

    return _compose_frame_result(
        context=active_context,
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
        context=context,
        alignment=alignment,
        frame=frame,
    )
    if alignment.homography is not None:
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
    alignment.extra_debug = _compact_alignment_extra_debug(merged)


def _compact_alignment_extra_debug(extra: dict[str, object]) -> dict[str, object]:
    compacted = dict(extra)

    reference_scores = compacted.get("reference_scores")
    if isinstance(reference_scores, list):
        limit = _debug_int_setting("INSPECTION_DEBUG_MAX_REFERENCE_SCORES", 8)
        total = len(reference_scores)
        compacted["reference_scores_total_count"] = total
        compacted["reference_scores"] = [
            _compact_reference_score(item)
            for item in reference_scores[:limit]
            if isinstance(item, dict)
        ]
        if total > limit:
            compacted["reference_scores_truncated_count"] = total - limit

    return compacted


def _compact_reference_score(item: dict[object, object]) -> dict[str, object]:
    result = _copy_debug_keys(
        item,
        {
            "rank",
            "reference_image_id",
            "is_primary",
            "status",
            "method",
            "stage",
            "reason",
            "raw_match_count",
            "inlier_count",
            "inlier_ratio",
            "median_error",
            "score",
            "confidence",
            "low_confidence",
            "guard_reasons",
            "slot_coverage",
        },
    )
    image_path = item.get("image_path")
    if isinstance(image_path, str):
        result["image_path_tail"] = image_path.rstrip("/").split("/")[-1]
    return result


def _copy_debug_keys(item: dict[object, object], keys: set[str]) -> dict[str, object]:
    copied: dict[str, object] = {}
    for key in keys:
        if key in item:
            copied[key] = item[key]
    return copied


def _debug_int_setting(name: str, default: int) -> int:
    raw = getattr(settings, name, default)
    try:
        value = int(raw)
    except (TypeError, ValueError):
        return default
    return max(1, value)

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


def _timed_select_reference_context_and_align(
    context: InspectionContext,
    frame: np.ndarray,
    max_side: int | None,
    max_keypoints: int,
    selection_grid: tuple[int, int] | None,
) -> tuple[ReferenceSelectionResult, float]:
    started_at = time.perf_counter()
    selection = _select_reference_context_and_align(
        context=context,
        frame=frame,
        max_side=max_side,
        max_keypoints=max_keypoints,
        selection_grid=selection_grid,
    )
    return selection, _elapsed_ms(started_at)


def _select_reference_context_and_align(
    *,
    context: InspectionContext,
    frame: np.ndarray,
    max_side: int | None,
    max_keypoints: int,
    selection_grid: tuple[int, int] | None,
) -> ReferenceSelectionResult:
    views = context.reference_views or [
        ReferenceView(
            image=context.reference_image,
            features=context.reference_features,
            is_primary=True,
        )
    ]

    if len(views) <= 1 or not settings.INSPECTION_MULTI_REFERENCE_ENABLED:
        active_context = context.with_reference_view(views[0])
        alignment = align_frame(
            context=active_context,
            frame=frame,
            max_side=max_side,
            max_keypoints=max_keypoints,
            selection_grid=selection_grid,
        )
        slot_coverage = _reference_slot_coverage_payload(
            context=active_context,
            alignment=alignment,
            frame=frame,
        )
        _merge_alignment_extra_debug(
            alignment,
            {
                "multi_reference_enabled": bool(
                    settings.INSPECTION_MULTI_REFERENCE_ENABLED
                ),
                "selected_reference_id": str(active_context.reference_image.id),
                "selected_reference_rank": 1,
                "reference_candidate_count": len(views),
                "selected_reference_confidence": _reference_quality_guard_payload(
                    alignment=alignment,
                    slot_coverage=slot_coverage,
                )["confidence"],
                "low_confidence_reference_selection": _reference_quality_guard_payload(
                    alignment=alignment,
                    slot_coverage=slot_coverage,
                )["low_confidence"],
                "reference_selection_guard_reasons": _reference_quality_guard_payload(
                    alignment=alignment,
                    slot_coverage=slot_coverage,
                )["reasons"],
                "reference_scores": [
                    _reference_score_payload(
                        views[0],
                        alignment,
                        1,
                        slot_coverage=slot_coverage,
                    )
                ],
            },
        )
        return ReferenceSelectionResult(
            context=active_context,
            alignment=alignment,
            scores=alignment.extra_debug.get("reference_scores", [])
            if isinstance(alignment.extra_debug, dict)
            else [],
        )

    scored: list[
        tuple[
            float,
            ReferenceView,
            InspectionContext,
            FrameAlignment,
            dict[str, object],
        ]
    ] = []
    score_payloads: list[dict[str, object]] = []

    for view in views:
        candidate_context = context.with_reference_view(view)
        alignment = align_frame(
            context=candidate_context,
            frame=frame,
            max_side=max_side,
            max_keypoints=max_keypoints,
            selection_grid=selection_grid,
        )
        slot_coverage = _reference_slot_coverage_payload(
            context=candidate_context,
            alignment=alignment,
            frame=frame,
        )
        score = _reference_alignment_score(view, alignment, slot_coverage)
        scored.append((score, view, candidate_context, alignment, slot_coverage))

    scored.sort(key=lambda item: item[0], reverse=True)
    for rank, (
        score,
        view,
        _candidate_context,
        alignment,
        slot_coverage,
    ) in enumerate(scored, start=1):
        score_payload = _reference_score_payload(
            view,
            alignment,
            rank,
            slot_coverage=slot_coverage,
        )
        score_payload["score"] = round(score, 3)
        score_payloads.append(score_payload)

    _score, best_view, best_context, best_alignment, _best_slot_coverage = scored[0]
    best_guard = _reference_quality_guard_payload(
        alignment=best_alignment,
        slot_coverage=_best_slot_coverage,
    )
    _merge_alignment_extra_debug(
        best_alignment,
        {
            "multi_reference_enabled": True,
            "selected_reference_id": str(best_view.image.id),
            "selected_reference_rank": 1,
            "selected_reference_is_primary": best_view.is_primary,
            "reference_candidate_count": len(views),
            "selected_reference_confidence": best_guard["confidence"],
            "low_confidence_reference_selection": best_guard["low_confidence"],
            "reference_selection_guard_reasons": best_guard["reasons"],
            "reference_selection_guard_thresholds": best_guard["thresholds"],
            "reference_scores": score_payloads,
        },
    )
    return ReferenceSelectionResult(
        context=best_context,
        alignment=best_alignment,
        scores=score_payloads,
    )


def _reference_alignment_score(
    view: ReferenceView,
    alignment: FrameAlignment,
    slot_coverage: dict[str, object] | None = None,
) -> float:
    median_error = alignment.median_error
    error_penalty = min(float(median_error), 50.0) if median_error is not None else 25.0
    primary_bonus = 0.01 if view.is_primary else 0.0
    coverage_score = _coverage_score(slot_coverage)

    if alignment.is_success:
        inlier_ratio = alignment.inlier_count / max(alignment.raw_match_count, 1)
        return (
            alignment.inlier_count * 3.0
            + alignment.raw_match_count * 0.15
            + inlier_ratio * 25.0
            + coverage_score * 1.25
            - error_penalty * 2.0
            + primary_bonus
        )

    # Failed alignments are still ranked, so debug shows why every candidate lost.
    return (
        alignment.inlier_count * 1.0
        + alignment.raw_match_count * 0.05
        + coverage_score * 0.25
        - error_penalty * 2.0
        + primary_bonus
    )


def _reference_score_payload(
    view: ReferenceView,
    alignment: FrameAlignment,
    rank: int,
    *,
    slot_coverage: dict[str, object] | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "rank": rank,
        "reference_image_id": str(view.image.id),
        "is_primary": view.is_primary,
        "image_path": view.image.image_path,
        "status": alignment.status.value,
        "method": alignment.method,
        "stage": alignment.stage,
        "reason": alignment.reason,
        "raw_match_count": alignment.raw_match_count,
        "inlier_count": alignment.inlier_count,
        "inlier_ratio": round(
            alignment.inlier_count / max(alignment.raw_match_count, 1),
            4,
        ),
        "median_error": alignment.median_error,
        "reference_feature_count": alignment.reference_feature_count,
        "frame_feature_count": alignment.frame_feature_count,
    }
    if slot_coverage is not None:
        payload["slot_coverage"] = slot_coverage

    guard = _reference_quality_guard_payload(
        alignment=alignment,
        slot_coverage=slot_coverage,
    )
    payload["confidence"] = guard["confidence"]
    payload["low_confidence"] = guard["low_confidence"]
    payload["guard_reasons"] = guard["reasons"]
    payload["guard_thresholds"] = guard["thresholds"]
    return payload


def _reference_quality_guard_payload(
    *,
    alignment: FrameAlignment,
    slot_coverage: dict[str, object] | None,
) -> dict[str, object]:
    thresholds = {
        "min_inliers": int(
            getattr(settings, "INSPECTION_MULTI_REFERENCE_WARN_MIN_INLIERS", 8)
        ),
        "min_inlier_ratio": float(
            getattr(settings, "INSPECTION_MULTI_REFERENCE_WARN_MIN_INLIER_RATIO", 0.08)
        ),
        "max_median_error": float(
            getattr(settings, "INSPECTION_MULTI_REFERENCE_WARN_MAX_MEDIAN_ERROR", 25.0)
        ),
        "min_visible_ratio": float(
            getattr(settings, "INSPECTION_MULTI_REFERENCE_WARN_MIN_VISIBLE_RATIO", 0.35)
        ),
    }

    reasons: list[str] = []
    raw_matches = max(int(alignment.raw_match_count or 0), 0)
    inliers = max(int(alignment.inlier_count or 0), 0)
    inlier_ratio = inliers / max(raw_matches, 1)

    if not alignment.is_success:
        reasons.append(f"alignment_status:{alignment.status.value}")

    if inliers < thresholds["min_inliers"]:
        reasons.append(
            f"too_few_inliers:{inliers}<"
            f"{thresholds['min_inliers']}"
        )

    if inlier_ratio < thresholds["min_inlier_ratio"]:
        reasons.append(
            f"low_inlier_ratio:{inlier_ratio:.3f}<"
            f"{thresholds['min_inlier_ratio']:.3f}"
        )

    if alignment.median_error is None:
        reasons.append("missing_median_error")
    elif float(alignment.median_error) > thresholds["max_median_error"]:
        reasons.append(
            f"high_median_error:{float(alignment.median_error):.2f}>"
            f"{thresholds['max_median_error']:.2f}"
        )

    visible_ratio = _slot_visible_ratio(slot_coverage)
    if visible_ratio is not None and visible_ratio < thresholds["min_visible_ratio"]:
        reasons.append(
            f"low_slot_visible_ratio:{visible_ratio:.3f}<"
            f"{thresholds['min_visible_ratio']:.3f}"
        )

    low_confidence = bool(reasons)
    return {
        "confidence": "low" if low_confidence else "high",
        "low_confidence": low_confidence,
        "reasons": reasons,
        "thresholds": thresholds,
    }


def _slot_visible_ratio(slot_coverage: dict[str, object] | None) -> float | None:
    if not slot_coverage:
        return None

    raw_total = slot_coverage.get("total")
    try:
        total = int(raw_total)
    except (TypeError, ValueError):
        total = 0

    if total <= 0:
        return None

    raw_ratio = slot_coverage.get("visible_ratio")
    if isinstance(raw_ratio, (int, float)):
        return float(raw_ratio)

    raw_visible = slot_coverage.get("visible")
    try:
        visible = int(raw_visible)
    except (TypeError, ValueError):
        return None

    return visible / max(total, 1)

def _coverage_score(slot_coverage: dict[str, object] | None) -> float:
    if not slot_coverage:
        return 0.0
    raw_score = slot_coverage.get("score")
    if isinstance(raw_score, (int, float)):
        return float(raw_score)
    return 0.0


def _reference_slot_coverage_payload(
    *,
    context: InspectionContext,
    alignment: FrameAlignment,
    frame: np.ndarray,
) -> dict[str, object]:
    expected = build_expected_segments(
        context.reference_image,
        {sc.id for sc in context.selected_classes},
    )
    total = len(expected)
    if total == 0:
        return {
            "total": 0,
            "projected": 0,
            "visible": 0,
            "avg_containment": None,
            "score": 0.0,
            "reason": "no_expected_segments",
        }

    if alignment.homography is None:
        return {
            "total": total,
            "projected": 0,
            "visible": 0,
            "avg_containment": 0.0,
            "score": 0.0,
            "reason": "no_homography",
        }

    frame_height, frame_width = frame.shape[:2]
    projected = 0
    visible = 0
    containments: list[float] = []

    for item in expected:
        containment = _projected_polygon_containment(
            item.reference_polygon,
            alignment.homography,
            frame_width=frame_width,
            frame_height=frame_height,
        )
        if containment is None:
            continue
        projected += 1
        containments.append(containment)
        if containment >= 0.35:
            visible += 1

    avg_containment = (sum(containments) / len(containments)) if containments else 0.0
    visible_ratio = visible / max(total, 1)
    projected_ratio = projected / max(total, 1)
    score = (visible_ratio * 70.0) + (projected_ratio * 15.0) + (avg_containment * 15.0)

    return {
        "total": total,
        "projected": projected,
        "visible": visible,
        "visible_ratio": round(visible_ratio, 4),
        "projected_ratio": round(projected_ratio, 4),
        "avg_containment": round(avg_containment, 4),
        "score": round(score, 3),
        "reason": "ok" if projected else "no_projected_polygons",
    }


def _projected_polygon_containment(
    polygon: list[list[float]],
    homography: np.ndarray,
    *,
    frame_width: int,
    frame_height: int,
) -> float | None:
    if not polygon:
        return None

    try:
        projected = np.asarray(project_polygon(polygon, homography), dtype=np.float32)
    except Exception:
        return None

    if projected.ndim != 2 or projected.shape[1] != 2 or not np.isfinite(projected).all():
        return None

    min_x = float(np.min(projected[:, 0]))
    max_x = float(np.max(projected[:, 0]))
    min_y = float(np.min(projected[:, 1]))
    max_y = float(np.max(projected[:, 1]))
    width = max_x - min_x
    height = max_y - min_y
    if width <= 1.0 or height <= 1.0:
        return None

    polygon_area = width * height
    clipped_min_x = max(0.0, min_x)
    clipped_max_x = min(float(frame_width - 1), max_x)
    clipped_min_y = max(0.0, min_y)
    clipped_max_y = min(float(frame_height - 1), max_y)
    clipped_width = max(0.0, clipped_max_x - clipped_min_x)
    clipped_height = max(0.0, clipped_max_y - clipped_min_y)
    return float((clipped_width * clipped_height) / max(polygon_area, 1.0))


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
