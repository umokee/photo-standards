from modules.yolo.inspection.adapters.payloads import build_model_name
from modules.yolo.inspection.models import InspectionResult
from modules.yolo.inspection.use_cases.start_inspection import InspectionStartResult

from .schemas import (
    InspectionHistoryItemResponse,
    InspectionResultResponse,
    InspectionSaveResponse,
    InspectionSegmentResultResponse,
    InspectionStartResponse,
)


def start_result(result: InspectionStartResult) -> InspectionStartResponse:
    return InspectionStartResponse(
        kind=result.kind,
        task_id=result.task_id,
        session_id=result.session_id,
        status=result.status,
        message=result.message,
    )


def save_result(
    inspection: InspectionResult,
    *,
    message: str,
) -> InspectionSaveResponse:
    return InspectionSaveResponse(
        inspection_id=inspection.id,
        status=inspection.status,
        message=message,
    )


def history_item(inspection: InspectionResult) -> InspectionHistoryItemResponse:
    return build_inspection_history_item_response(inspection)


def inspection_result(inspection: InspectionResult) -> InspectionResultResponse:
    return build_inspection_result_response(inspection)


def build_inspection_history_item_response(
    inspection: InspectionResult,
) -> InspectionHistoryItemResponse:
    standard = inspection.standard
    model = inspection.ml_model
    camera = inspection.camera

    return InspectionHistoryItemResponse(
        id=inspection.id,
        group_id=(
            standard.group_id
            if standard is not None
            else model.group_id
            if model is not None
            else None
        ),
        standard_id=inspection.standard_id,
        standard_name=standard.name if standard is not None else None,
        standard_reference_path=_get_inspection_reference_path(inspection),
        model_id=inspection.model_id,
        model_name=build_model_name(model) if model is not None else None,
        camera_id=inspection.camera_id,
        camera_name=camera.name if camera is not None else None,
        mode=inspection.mode,
        status=inspection.status,
        image_path=inspection.image_path,
        result_image_path=inspection.result_image_path,
        total_segments=inspection.total_segments,
        matched_segments=inspection.matched_segments,
        final_pose_source=_get_final_pose_source(inspection),
        final_pose_reason=_get_final_pose_reason(inspection),
        notes=inspection.notes,
        inspected_at=inspection.inspected_at,
    )


def build_inspection_result_response(
    inspection: InspectionResult,
) -> InspectionResultResponse:
    standard = inspection.standard
    model = inspection.ml_model
    camera = inspection.camera
    user = inspection.user

    return InspectionResultResponse(
        id=inspection.id,
        group_id=(
            standard.group_id
            if standard is not None
            else model.group_id
            if model is not None
            else None
        ),
        standard_id=inspection.standard_id,
        standard_name=standard.name if standard is not None else None,
        standard_reference_path=_get_inspection_reference_path(inspection),
        model_id=inspection.model_id,
        model_name=build_model_name(model) if model is not None else None,
        camera_id=inspection.camera_id,
        camera_name=camera.name if camera is not None else None,
        user_id=inspection.user_id,
        user_name=user.full_name if user is not None else None,
        mode=inspection.mode,
        status=inspection.status,
        image_path=inspection.image_path,
        result_image_path=inspection.result_image_path,
        total_segments=inspection.total_segments,
        matched_segments=inspection.matched_segments,
        alignment_status=inspection.alignment_status,
        alignment_inlier_count=inspection.alignment_inlier_count,
        alignment_raw_match_count=inspection.alignment_raw_match_count,
        homography=inspection.homography,
        notes=inspection.notes,
        debug_payload=inspection.debug_payload,
        pose_pipeline=_get_pose_pipeline(inspection),
        inspected_at=inspection.inspected_at,
        segment_results=[
            InspectionSegmentResultResponse.model_validate(item)
            for item in inspection.segment_results
        ],
    )


def _get_inspection_reference_path(inspection: InspectionResult) -> str | None:
    standard = inspection.standard
    selected_reference_id = _find_debug_value(
        inspection.debug_payload,
        "selected_reference_id",
    )

    if selected_reference_id and standard is not None:
        selected_reference_id_text = str(selected_reference_id)
        for image in getattr(standard, "images", []):
            if str(getattr(image, "id", "")) == selected_reference_id_text:
                return getattr(image, "image_path", None)

    return _get_standard_reference_path(standard)


def _find_debug_value(payload: object, key: str) -> object | None:
    if isinstance(payload, dict):
        if key in payload:
            return payload[key]
        for value in payload.values():
            found = _find_debug_value(value, key)
            if found is not None:
                return found
    if isinstance(payload, list):
        for value in payload:
            found = _find_debug_value(value, key)
            if found is not None:
                return found
    return None


def _get_standard_reference_path(standard: object | None) -> str | None:
    if standard is None:
        return None

    for image in getattr(standard, "images", []):
        if getattr(image, "is_reference", False):
            return image.image_path

    return None


def _get_pose_pipeline(inspection: InspectionResult) -> dict | None:
    payload = inspection.debug_payload
    if not isinstance(payload, dict):
        return None

    pose_pipeline = payload.get("pose_pipeline")
    if isinstance(pose_pipeline, dict):
        return pose_pipeline

    return None


def _get_final_pose_source(inspection: InspectionResult) -> str | None:
    pose_pipeline = _get_pose_pipeline(inspection)
    if pose_pipeline is None:
        return None
    value = pose_pipeline.get("final_pose_source")
    return str(value) if value is not None else None


def _get_final_pose_reason(inspection: InspectionResult) -> str | None:
    pose_pipeline = _get_pose_pipeline(inspection)
    if pose_pipeline is None:
        return None
    value = pose_pipeline.get("final_pose_reason")
    return str(value) if value is not None else None
