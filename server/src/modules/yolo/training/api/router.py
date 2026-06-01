from __future__ import annotations

from app.dependencies import DbSession
from fastapi import APIRouter
from modules.yolo.training.api import presenters, schemas
from modules.yolo.training.use_cases import start_training

router = APIRouter(prefix="/training", tags=["yolo"])


@router.post("/run", response_model=schemas.TrainingStartResponse, status_code=202)
async def start_training_route(
    db: DbSession,
    data: schemas.TrainRequest,
) -> schemas.TrainingStartResponse:
    result = await start_training.start_training(
        db,
        group_id=data.group_id,
        architecture=data.architecture,
        epochs=data.epochs,
        imgsz=data.imgsz,
        batch_size=data.batch_size,
        train_ratio=data.train_ratio,
        val_ratio=data.val_ratio,
    )
    return presenters.training_start_result(result)
