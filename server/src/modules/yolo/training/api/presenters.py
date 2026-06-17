from __future__ import annotations

from modules.yolo.training.adapters import checkpoint_metrics, storage
from modules.yolo.training.api.schemas import MlModelResponse, TrainingStartResponse
from modules.yolo.training.models import MlModel
from modules.yolo.training.use_cases.start_training import TrainingStartResult


def training_start_result(result: TrainingStartResult) -> TrainingStartResponse:
    return TrainingStartResponse(
        task_id=result.task.id,
        model_id=result.model.id,
    )


def model_response(model: MlModel) -> MlModelResponse:
    response = MlModelResponse.model_validate(model)
    checkpoint = checkpoint_metrics.load_checkpoint_metrics(
        storage.resolve_model_weights_path(model.weights_path)
    )
    if checkpoint is not None and checkpoint.summary:
        response.metrics = checkpoint.summary
    return response
