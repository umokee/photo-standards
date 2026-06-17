from __future__ import annotations

import csv
from pathlib import Path

_SERIES_ALIASES: dict[str, tuple[str, ...]] = {
    "precision": ("metrics/precision(B)", "metrics/precision(M)", "precision"),
    "recall": ("metrics/recall(B)", "metrics/recall(M)", "recall"),
    "mAP50": ("metrics/mAP50(B)", "metrics/mAP50(M)", "mAP50"),
    "mAP50_95": ("metrics/mAP50-95(B)", "metrics/mAP50-95(M)", "mAP50_95", "mAP50-95"),
    "train_box_loss": ("train/box_loss", "train_box_loss"),
    "train_cls_loss": ("train/cls_loss", "train_cls_loss"),
    "train_dfl_loss": ("train/dfl_loss", "train_dfl_loss"),
    "val_box_loss": ("val/box_loss", "val_box_loss"),
    "val_cls_loss": ("val/cls_loss", "val_cls_loss"),
    "val_dfl_loss": ("val/dfl_loss", "val_dfl_loss"),
    "lr0": ("lr/pg0", "lr0", "lr_pg0"),
    "lr1": ("lr/pg1", "lr1", "lr_pg1"),
    "lr2": ("lr/pg2", "lr2", "lr_pg2"),
}


def parse_results_csv(csv_path: Path) -> tuple[list[int], dict[str, list[float]]]:
    if not csv_path.is_file():
        return [], {}

    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)

    if not rows:
        return [], {}

    epochs = [_epoch_value(row, index) for index, row in enumerate(rows)]
    series: dict[str, list[float]] = {}

    for key, aliases in _SERIES_ALIASES.items():
        values = [_metric_value(row, aliases) for row in rows]
        compact = [value for value in values if value is not None]
        if compact:
            series[key] = compact

    return epochs, series


def _metric_value(row: dict[str, str | None], aliases: tuple[str, ...]) -> float | None:
    for alias in aliases:
        raw = row.get(alias)
        if raw is None:
            continue

        normalized = raw.strip()
        if not normalized:
            continue

        try:
            return float(normalized)
        except ValueError:
            continue

    return None


def _epoch_value(row: dict[str, str | None], index: int) -> int:
    raw = row.get("epoch")
    if raw is None:
        return index + 1

    normalized = raw.strip()
    if not normalized:
        return index + 1

    try:
        parsed = int(float(normalized))
    except ValueError:
        return index + 1

    return parsed + 1 if parsed == index else parsed
