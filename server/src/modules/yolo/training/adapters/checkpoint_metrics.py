from __future__ import annotations

import contextlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_SERIES_ALIASES: dict[str, tuple[str, ...]] = {
    "precision": ("metrics/precision(B)", "metrics/precision(M)", "precision"),
    "recall": ("metrics/recall(B)", "metrics/recall(M)", "recall"),
    "mAP50": ("metrics/mAP50(B)", "metrics/mAP50(M)", "mAP50"),
    "mAP50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "mAP50_95", "mAP50-95"),
    "train_box_loss": ("train/box_loss", "box_loss", "train_box_loss"),
    "train_cls_loss": ("train/cls_loss", "cls_loss", "train_cls_loss"),
    "train_dfl_loss": ("train/dfl_loss", "dfl_loss", "train_dfl_loss"),
    "val_box_loss": ("val/box_loss", "val_box_loss"),
    "val_cls_loss": ("val/cls_loss", "val_cls_loss"),
    "val_dfl_loss": ("val/dfl_loss", "val_dfl_loss"),
    "lr0": ("lr/pg0", "lr0", "lr_pg0"),
    "lr1": ("lr/pg1", "lr1", "lr_pg1"),
    "lr2": ("lr/pg2", "lr2", "lr_pg2"),
}

_SUMMARY_ALIASES: dict[str, tuple[str, ...]] = {
    "precision": _SERIES_ALIASES["precision"],
    "recall": _SERIES_ALIASES["recall"],
    "mAP50": _SERIES_ALIASES["mAP50"],
    "mAP50_95": _SERIES_ALIASES["mAP50_95"],
}


@dataclass(slots=True)
class CheckpointMetrics:
    summary: dict[str, float | None]
    series: dict[str, list[float]]


def load_checkpoint_metrics(weights_path: Path | None) -> CheckpointMetrics | None:
    if weights_path is None or not weights_path.is_file():
        return None

    try:
        import torch
    except Exception:
        return None

    try:
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=False)
    except TypeError:
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")
        except Exception:
            return None
    except Exception:
        return None

    if not isinstance(checkpoint, dict):
        return None

    train_results = checkpoint.get("train_results")
    train_metrics = checkpoint.get("train_metrics")

    series = _normalize_series(train_results)
    summary = _normalize_summary(train_metrics)

    if not summary and series:
        summary = {
            key: values[-1]
            for key, values in series.items()
            if key in _SUMMARY_ALIASES and values
        }

    if not summary and not series:
        return None

    return CheckpointMetrics(summary=summary, series=series)


def _normalize_series(raw: Any) -> dict[str, list[float]]:
    if not isinstance(raw, dict):
        return {}

    normalized: dict[str, list[float]] = {}
    for target_key, aliases in _SERIES_ALIASES.items():
        values = _extract_series(raw, aliases)
        if values:
            normalized[target_key] = values

    return normalized


def _normalize_summary(raw: Any) -> dict[str, float | None]:
    if not isinstance(raw, dict):
        return {}

    summary: dict[str, float | None] = {}
    for target_key, aliases in _SUMMARY_ALIASES.items():
        value = _extract_scalar(raw, aliases)
        if value is not None:
            summary[target_key] = value

    return summary


def _extract_series(source: dict[str, Any], aliases: tuple[str, ...]) -> list[float]:
    for alias in aliases:
        values = _coerce_float_list(source.get(alias))
        if values:
            return values
    return []


def _extract_scalar(source: dict[str, Any], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        with contextlib.suppress(TypeError, ValueError):
            value = source.get(alias)
            if value is not None:
                return float(value)
    return None


def _coerce_float_list(value: Any) -> list[float]:
    if not isinstance(value, list):
        return []

    result: list[float] = []
    for item in value:
        with contextlib.suppress(TypeError, ValueError):
            result.append(float(item))
    return result
