from __future__ import annotations

from uuid import UUID

from app.dependencies import DbSession
from fastapi import APIRouter, File, Form, UploadFile
from fastapi.responses import FileResponse
from modules.yolo.interop.api import presenters, schemas
from modules.yolo.interop.use_cases.build_model_export_archive import (
    build_model_export_archive,
)
from modules.yolo.interop.use_cases.create_import_task import create_import_task
from modules.yolo.interop.use_cases.preview_import_model import preview_import_model
from modules.yolo.training.constants import training
from starlette.background import BackgroundTask

router = APIRouter()


@router.post(
    "/import/preview",
    response_model=schemas.ModelImportPreviewResponse,
    status_code=200,
)
async def preview_model_import_route(
    db: DbSession,
    group_id: UUID = Form(...),  # noqa: B008
    weights: UploadFile = File(...),  # noqa: B008
) -> schemas.ModelImportPreviewResponse:
    data = schemas.parse_import_preview_params(
        group_id=group_id,
        weights=weights,
    )
    result = await preview_import_model(
        db,
        data=data,
    )
    return presenters.preview_result(result)


@router.post(
    "/import", response_model=schemas.ModelImportStartResponse, status_code=202
)
async def import_model_route(
    db: DbSession,
    group_id: UUID = Form(...),  # noqa: B008
    architecture: str = Form(...),
    imgsz: int = Form(...),
    batch_size: int = Form(training.batch_size.default),
    activate: bool = Form(False),
    mappings_json: str = Form(...),
    weights: UploadFile = File(...),  # noqa: B008
) -> schemas.ModelImportStartResponse:
    data = schemas.parse_import_create_params(
        group_id=group_id,
        architecture=architecture,
        imgsz=imgsz,
        batch_size=batch_size,
        activate=activate,
        mappings_json=mappings_json,
        weights=weights,
    )
    result = await create_import_task(
        db,
        data=data,
    )
    return presenters.import_start_result(result)


@router.get("/{model_id}/export")
async def export_model_route(
    db: DbSession,
    model_id: UUID,
) -> FileResponse:
    archive = await build_model_export_archive(
        db,
        model_id=model_id,
    )
    return FileResponse(
        archive.path,
        filename=archive.filename,
        media_type="application/zip",
        background=BackgroundTask(archive.path.unlink, missing_ok=True),
    )
