from __future__ import annotations

from uuid import UUID

from modules.tasks.constants import tasks as tasks_constants
from modules.yolo.training.adapters import checkpoint_metrics, metrics_artifacts, repository, storage
from modules.yolo.training.api.schemas import TrainingMetricsHistoryResponse
from sqlalchemy.ext.asyncio import AsyncSession


async def get_model_metrics_history(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> TrainingMetricsHistoryResponse:
    model = await repository.get_model(db, model_id=model_id)
    live_task = await repository.get_latest_training_task_for_model(
        db,
        model_id=model.id,
        statuses=[*tasks_constants.statuses.active, tasks_constants.statuses.paused],
    )

    if live_task is not None:
        live_results_path = storage.resolve_run_results_path(live_task.run_dir)
        if live_results_path is not None and live_results_path.is_file():
            epochs, series = metrics_artifacts.parse_results_csv(live_results_path)
            return TrainingMetricsHistoryResponse(
                model_id=model.id,
                task_id=live_task.id,
                source="live",
                epochs=epochs,
                series=series,
            )

    artifact_results_path = storage.resolve_metrics_results_path(model.weights_path)
    checkpoint = checkpoint_metrics.load_checkpoint_metrics(
        storage.resolve_model_weights_path(model.weights_path)
    )
    if checkpoint is not None and checkpoint.series:
        max_length = max((len(values) for values in checkpoint.series.values()), default=0)
        return TrainingMetricsHistoryResponse(
            model_id=model.id,
            task_id=None,
            source="checkpoint",
            epochs=list(range(1, max_length + 1)),
            series=checkpoint.series,
        )

    if artifact_results_path is not None and artifact_results_path.is_file():
        epochs, series = metrics_artifacts.parse_results_csv(artifact_results_path)
        return TrainingMetricsHistoryResponse(
            model_id=model.id,
            task_id=None,
            source="artifact",
            epochs=epochs,
            series=series,
        )

    return TrainingMetricsHistoryResponse(
        model_id=model.id,
        task_id=live_task.id if live_task is not None else None,
        source="empty",
        epochs=[],
        series={},
    )
