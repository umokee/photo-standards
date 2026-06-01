from uuid import UUID

from app.dependencies import DbSession
from fastapi import APIRouter
from modules.yolo.inspection.api.router import router as inspection_router
from modules.yolo.interop.api.router import router as interop_router
from modules.yolo.training.api.presenters import model_response
from modules.yolo.training.api.router import router as training_router
from modules.yolo.training.api.schemas import MlModelResponse
from modules.yolo.training.use_cases.activate_model import activate_model
from modules.yolo.training.use_cases.delete_model import delete_model
from modules.yolo.training.use_cases.get_model import get_model
from modules.yolo.training.use_cases.list_models import list_models

router = APIRouter(prefix="/yolo", tags=["yolo"])
router.include_router(interop_router)
router.include_router(inspection_router)
router.include_router(training_router)


@router.get("/{model_id}", response_model=MlModelResponse)
async def get_model_route(
    db: DbSession,
    model_id: UUID,
) -> MlModelResponse:
    model = await get_model(db, model_id=model_id)
    return model_response(model)


@router.get("", response_model=list[MlModelResponse])
async def list_models_route(
    db: DbSession,
    group_id: UUID,
) -> list[MlModelResponse]:
    models = await list_models(db, group_id=group_id)
    return [model_response(model) for model in models]


@router.put("/{model_id}/activate", response_model=MlModelResponse)
async def activate_model_route(
    db: DbSession,
    model_id: UUID,
) -> MlModelResponse:
    model = await activate_model(db, model_id=model_id)
    return model_response(model)


@router.delete("/{model_id}", status_code=204)
async def delete_model_route(
    db: DbSession,
    model_id: UUID,
) -> None:
    await delete_model(db, model_id=model_id)
