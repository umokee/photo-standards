from __future__ import annotations

from modules.yolo.interop.api import schemas
from modules.yolo.interop.use_cases.create_import_task import ModelImportStartResult
from modules.yolo.interop.use_cases.preview_import_model import ModelImportPreviewResult


def preview_result(
    result: ModelImportPreviewResult,
) -> schemas.ModelImportPreviewResponse:
    return schemas.ModelImportPreviewResponse(
        native_classes=[
            schemas.ImportedNativeClassResponse(
                index=item.index,
                key=item.key,
                suggested=(
                    schemas.ImportedClassSuggestionResponse(
                        segment_class_id=item.suggested.segment_class_id,
                        segment_class_name=item.suggested.segment_class_name,
                        match_reason=item.suggested.match_reason,
                    )
                    if item.suggested is not None
                    else None
                ),
            )
            for item in result.native_classes
        ]
    )


def import_start_result(
    result: ModelImportStartResult,
) -> schemas.ModelImportStartResponse:
    return schemas.ModelImportStartResponse(
        task_id=result.task_id,
        model_id=result.model_id,
    )
