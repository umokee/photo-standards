from __future__ import annotations

from modules.yolo.training.api.schemas import MlModelResponse, TrainingStartResponse
from modules.yolo.training.models import MlModel
from modules.yolo.training.use_cases.start_training import TrainingStartResult


def training_start_result(result: TrainingStartResult) -> TrainingStartResponse:
    return TrainingStartResponse(
        task_id=result.task.id,
        model_id=result.model.id,
    )


def model_response(model: MlModel) -> MlModelResponse:
    return MlModelResponse.model_validate(model)
