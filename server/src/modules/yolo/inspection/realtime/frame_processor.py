from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from uuid import UUID

import numpy as np
from app.config import settings
from modules.core.standards.reference_constants import (
    SUPERPOINT_VIDEO_GRID_COLS,
    SUPERPOINT_VIDEO_GRID_ROWS,
    SUPERPOINT_VIDEO_MAX_KEYPOINTS,
    SUPERPOINT_VIDEO_MAX_SIDE,
)
from modules.yolo.inspection.adapters.context import InspectionContext
from modules.yolo.inspection.adapters.entities import build_matches_from_details
from modules.yolo.inspection.adapters.payloads import build_result_item
from modules.yolo.inspection.domain.matcher import build_expected_segments
from modules.yolo.inspection.domain.overlay import render_overlay
from modules.yolo.inspection.domain.types import ExpectedSegment
from modules.yolo.inspection.use_cases.inspect_frame import (
    InspectionFrameResult,
    inspect_frame_parallel,
)

from .constants import (
    EMPTY_SCENE_INSPECTION_EVERY_N_FRAMES,
    INSPECTION_EVERY_N_FRAMES,
)
from .frame_result import (
    FrameResult,
    rebuild_frame_result_from_details,
)
from .overlay_motion import OverlayMotionTracker
from .profiler import FrameProfiler


@dataclass(slots=True)
class ProcessedRealtimeFrame:
    rendered: np.ndarray
    result: FrameResult


class RealtimeFrameProcessor:
    def __init__(
        self,
        *,
        session_id: UUID,
        context: InspectionContext,
    ) -> None:
        self._session_id = session_id
        self._context = context
        self._active_inspection_interval = max(1, int(INSPECTION_EVERY_N_FRAMES))
        self._empty_scene_inspection_interval = max(
            self._active_inspection_interval,
            int(EMPTY_SCENE_INSPECTION_EVERY_N_FRAMES),
        )
        self._inspection_interval = self._active_inspection_interval

        self._frame_counter = 0
        self._last_full_result: InspectionFrameResult | None = None

        self._expected_segments = build_expected_segments(
            context.reference_image,
            {sc.id for sc in context.selected_classes},
        )

        self._executor = ThreadPoolExecutor(
            max_workers=2,
            thread_name_prefix=f"rt-inspect-{session_id.hex[:8]}",
        )

        self._motion_tracker = OverlayMotionTracker()
        self._fps_counter = _FpsCounter()

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)

    def process(self, frame: np.ndarray) -> ProcessedRealtimeFrame | None:
        profile_enabled = FrameProfiler.is_enabled()
        profiler = FrameProfiler(str(self._session_id))

        self._frame_counter += 1
        display_fps = self._fps_counter.tick()

        needs_full_pipeline = self._needs_full_pipeline()
        polygon_transform: np.ndarray | None = None

        if needs_full_pipeline:
            frame_result, pipeline_mode = self._process_full_frame(
                frame=frame,
                display_fps=display_fps,
                profiler=profiler,
                profile_enabled=profile_enabled,
            )
        else:
            if self._last_full_result is None:
                raise RuntimeError(
                    "_last_full_result is None on non-full-pipeline frame"
                )

            frame_result = self._last_full_result
            pipeline_mode = "cached"

            with profiler.stage("track_cached_overlay"):
                polygon_transform = self._motion_tracker.track(frame)

        with profiler.stage("build_result"):
            result = self._build_frame_result(frame_result)

        with profiler.stage("render_overlay"):
            rendered = self._render_frame_from_result(
                frame=frame,
                result=result,
                fps=display_fps,
                polygon_transform=polygon_transform,
            )

        profiler.commit(
            extra={
                "fps": round(display_fps, 1) if display_fps is not None else None,
                "status": result.status,
                "alignment_status": result.alignment_status,
                "detections_count": len(frame_result.detections),
                "expected_count": len(frame_result.expected_segments),
                "matches_count": len(frame_result.matches),
                "pipeline_mode": pipeline_mode,
                "verification_mode": frame_result.verification_mode,
                "inspection_interval": self._inspection_interval,
                "result_matched": result.matched,
                "result_total": result.total,
                "result_details_count": len(result.details),
                "result_ok_count": sum(
                    1 for detail in result.details if str(detail.get("status")) == "ok"
                ),
                "result_missing_count": sum(
                    1
                    for detail in result.details
                    if str(detail.get("status")) == "missing"
                ),
            }
        )

        return ProcessedRealtimeFrame(
            rendered=rendered,
            result=result,
        )

    def _render_frame_from_result(
        self,
        *,
        frame: np.ndarray,
        result: FrameResult,
        fps: float | None,
        polygon_transform: np.ndarray | None = None,
    ) -> np.ndarray:
        if result.verification_mode == "yolo_count":
            matches = build_matches_from_details(result.details)
            return render_overlay(
                frame,
                matches,
                fps=fps,
                alignment_message=None,
                polygon_transform=polygon_transform,
            )

        if result.alignment_db_status != "success":
            return render_overlay(
                frame,
                [],
                fps=fps,
                alignment_message=result.alignment_status,
                polygon_transform=None,
            )

        matches = build_matches_from_details(result.details)

        return render_overlay(
            frame,
            matches,
            fps=fps,
            alignment_message=None,
            polygon_transform=polygon_transform,
        )

    def _needs_full_pipeline(self) -> bool:
        return (
            self._last_full_result is None
            or self._frame_counter % self._inspection_interval == 0
        )

    def _process_full_frame(
        self,
        *,
        frame: np.ndarray,
        display_fps: float | None,
        profiler: FrameProfiler,
        profile_enabled: bool,
    ) -> tuple[InspectionFrameResult, str]:
        with profiler.stage("inspect_parallel"):
            frame_result = inspect_frame_parallel(
                context=self._context,
                frame=frame,
                render=False,
                fps=display_fps,
                expected_segments=self._expected_segments,
                profile_enabled=profile_enabled,
                executor=self._executor,
                alignment_max_side=SUPERPOINT_VIDEO_MAX_SIDE,
                alignment_max_keypoints=SUPERPOINT_VIDEO_MAX_KEYPOINTS,
                alignment_selection_grid=(
                    SUPERPOINT_VIDEO_GRID_ROWS,
                    SUPERPOINT_VIDEO_GRID_COLS,
                ),
                yolo_conf=settings.YOLO_REALTIME_CONF_THRESHOLD,
            )

        for stage_name, stage_ms in frame_result.profile.items():
            profiler.record(stage_name, stage_ms)

        if frame_result.verification_mode == "yolo_count":
            self._last_full_result = frame_result
            self._motion_tracker.reset(frame)
            if self._should_slow_down_for_empty_scene(frame_result):
                self._inspection_interval = self._empty_scene_inspection_interval
                return frame_result, "empty_scene"

            self._inspection_interval = self._active_inspection_interval
            return frame_result, "yolo_count"

        self._last_full_result = frame_result
        self._motion_tracker.reset(frame)

        if self._should_slow_down_for_empty_scene(frame_result):
            self._inspection_interval = self._empty_scene_inspection_interval
            return frame_result, "empty_scene"

        self._inspection_interval = self._active_inspection_interval

        if frame_result.alignment.is_success:
            return frame_result, "full_parallel"

        return frame_result, "alignment_failure"

    def _should_slow_down_for_empty_scene(
        self,
        frame_result: InspectionFrameResult,
    ) -> bool:
        if frame_result.detections:
            return False

        if frame_result.verification_mode == "yolo_count":
            return True

        return not frame_result.alignment.is_success

    def _build_frame_result(self, frame_result: InspectionFrameResult) -> FrameResult:
        details = [build_result_item(match) for match in frame_result.matches]
        details = self._complete_expected_details(
            details=details,
            expected_segments=frame_result.expected_segments,
        )

        base_result = FrameResult(
            matched=0,
            total=len(frame_result.expected_segments),
            missing=[],
            status=frame_result.inspection_status,
            passed=False,
            alignment_status=frame_result.alignment_message,
            alignment_db_status=(
                None
                if frame_result.verification_mode == "yolo_count"
                else frame_result.alignment.status.value
            ),
            alignment_inlier_count=(
                None
                if frame_result.verification_mode == "yolo_count"
                else frame_result.alignment.inlier_count
            ),
            alignment_raw_match_count=(
                None
                if frame_result.verification_mode == "yolo_count"
                else frame_result.alignment.raw_match_count
            ),
            homography=(
                None
                if frame_result.verification_mode == "yolo_count"
                else (
                    frame_result.alignment.homography.tolist()
                    if frame_result.alignment.homography is not None
                    else None
                )
            ),
            alignment_debug=frame_result.alignment.to_debug_payload(),
            verification_mode=frame_result.verification_mode,
            captured_at=datetime.now(UTC),
            details=details,
        )

        return rebuild_frame_result_from_details(
            base_result,
            details,
            fallback_total=len(frame_result.expected_segments),
        )

    def _complete_expected_details(
        self,
        *,
        details: list[dict],
        expected_segments: list[ExpectedSegment],
    ) -> list[dict]:
        completed = [dict(detail) for detail in details]

        present_annotation_ids = {
            str(detail.get("annotation_id"))
            for detail in completed
            if detail.get("annotation_id")
            and str(detail.get("status") or "") in {"ok", "missing"}
        }

        for item in expected_segments:
            annotation_id = str(item.annotation_id)

            if annotation_id in present_annotation_ids:
                continue

            completed.append(
                {
                    "annotation_id": annotation_id,
                    "segment_class_id": str(item.segment_class_id),
                    "class_key": item.class_key,
                    "name": item.name,
                    "hue": item.hue,
                    "status": "missing",
                    "iou": None,
                    "confidence": None,
                    "expected_polygon": None,
                    "detected_polygon": None,
                    "detected_bbox": None,
                }
            )

        return completed


class _FpsCounter:
    def __init__(self) -> None:
        self._window_started = time.perf_counter()
        self._frames_in_window = 0
        self._displayed_fps = 0.0

    def tick(self) -> float:
        self._frames_in_window += 1
        elapsed = time.perf_counter() - self._window_started

        if elapsed < 1.0:
            return self._displayed_fps

        self._displayed_fps = self._frames_in_window / elapsed
        self._frames_in_window = 0
        self._window_started = time.perf_counter()
        return self._displayed_fps
