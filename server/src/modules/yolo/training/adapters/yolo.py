from __future__ import annotations

import shutil
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog
from app.observability import log_event

logger = structlog.get_logger(__name__)


class TrainingInterrupted(RuntimeError):
    pass


@dataclass(slots=True)
class TrainingRunConfig:
    yaml_path: Path
    base_weights_path: Path
    checkpoint_path: Path
    best_checkpoint_path: Path
    output_checkpoint_path: Path
    output_weights_path: Path
    run_dir: Path
    epochs: int
    imgsz: int
    batch_size: int
    resume: bool = False
    save_period: int = 1
    on_status: Callable[[str], None] | None = None
    on_epoch_end: Callable[[int, int, dict[str, float | None]], None] | None = None
    on_model_save: Callable[[Path | None, Path | None], None] | None = None
    on_heartbeat: Callable[[], None] | None = None
    should_stop: Callable[[], bool] | None = None


@dataclass(slots=True)
class TrainingRunResult:
    output_weights_path: Path
    metrics: dict[str, float | None]
    raw_results: Any | None = None


def run_training_sync(
    config: TrainingRunConfig,
) -> TrainingRunResult:
    source_weights = (
        config.checkpoint_path
        if config.resume and config.checkpoint_path.exists()
        else config.base_weights_path
    )

    device = _training_device()

    log_event(
        logger,
        "info",
        "training.runtime.started",
        resume=config.resume,
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch_size=config.batch_size,
        device=device,
        yaml_path=config.yaml_path,
        source_weights=source_weights,
        run_dir=config.run_dir,
    )

    if config.on_status:
        config.on_status("training")

    from ultralytics import YOLO

    yolo = YOLO(str(source_weights))

    if config.on_epoch_end or config.should_stop:

        def _on_fit_epoch_end(trainer):
            current = trainer.epoch + 1
            total = trainer.epochs
            metrics = _extract_metrics(getattr(trainer, "metrics", {}) or {})

            if config.on_epoch_end:
                config.on_epoch_end(current, total, metrics)

            if config.should_stop and config.should_stop():
                raise TrainingInterrupted("Training stop requested")

        yolo.add_callback("on_fit_epoch_end", _on_fit_epoch_end)

    if config.on_heartbeat or config.should_stop:

        def _on_batch_end(trainer):
            if config.on_heartbeat:
                config.on_heartbeat()
            if config.should_stop and config.should_stop():
                raise TrainingInterrupted("Training stop requested")

        yolo.add_callback("on_train_batch_end", _on_batch_end)

    if config.on_model_save:
        yolo.add_callback(
            "on_model_save",
            lambda trainer: config.on_model_save(
                config.checkpoint_path if config.checkpoint_path.exists() else None,
                config.best_checkpoint_path
                if config.best_checkpoint_path.exists()
                else None,
            ),
        )

    config.run_dir.mkdir(parents=True, exist_ok=True)
    config.output_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    config.output_weights_path.parent.mkdir(parents=True, exist_ok=True)

    project_dir = config.run_dir.parent
    run_name = config.run_dir.name

    results = yolo.train(
        data=str(config.yaml_path),
        epochs=config.epochs,
        imgsz=config.imgsz,
        batch=config.batch_size,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
        save=True,
        save_period=config.save_period,
        resume=config.resume,
        device=device,
        workers=2,
        cache=False,
        patience=30,
        plots=False,
    )

    if config.on_status:
        config.on_status("saving")

    if not config.best_checkpoint_path.exists():
        raise FileNotFoundError(f"best.pt не найден: {config.best_checkpoint_path}")

    shutil.copy2(
        str(config.best_checkpoint_path),
        str(config.output_checkpoint_path),
    )

    if config.on_status:
        config.on_status("saving")

    trained_pt_path = config.output_checkpoint_path
    if trained_pt_path.resolve() != config.output_weights_path.resolve():
        shutil.copy2(
            str(trained_pt_path),
            str(config.output_weights_path),
        )

    if not config.output_weights_path.exists():
        raise FileNotFoundError(
            f"Финальный .pt не найден: {config.output_weights_path}"
        )

    results_dict = getattr(results, "results_dict", {}) or {}
    metrics = _extract_metrics(results_dict)

    log_event(
        logger,
        "info",
        "training.runtime.finished",
        output_checkpoint_path=config.output_checkpoint_path,
        output_weights_path=config.output_weights_path,
        metrics=metrics,
    )

    return TrainingRunResult(
        output_weights_path=config.output_weights_path,
        metrics=metrics,
        raw_results=results,
    )


def _training_device() -> str:
    import torch

    return "cuda:0" if torch.cuda.is_available() else "cpu"


def _metric(
    results_dict: dict[str, Any],
    *keys: str,
) -> float | None:
    for key in keys:
        value = results_dict.get(key)
        if value is not None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


def _extract_metrics(
    results_dict: dict[str, Any],
) -> dict[str, float | None]:
    return {
        "mAP50": _metric(results_dict, "metrics/mAP50(B)", "metrics/mAP50(M)"),
        "mAP50_95": _metric(results_dict, "metrics/mAP50-95(B)", "metrics/mAP50-95(M)"),
        "precision": _metric(
            results_dict, "metrics/precision(B)", "metrics/precision(M)"
        ),
        "recall": _metric(results_dict, "metrics/recall(B)", "metrics/recall(M)"),
    }
