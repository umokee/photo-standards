from __future__ import annotations

from pathlib import Path
from typing import Any

from app.exception import ValidationError
from modules.yolo.interop.domain.types import ImportedNativeClassRef


def read_native_classes(weights_path: Path) -> list[ImportedNativeClassRef]:
    model = _load_yolo_model(weights_path)
    _ensure_segmentation_model(model)

    names = getattr(model, "names", None) or getattr(
        getattr(model, "model", None), "names", None
    )
    if not names:
        raise ValidationError("В модели отсутствует список классов")

    result: list[ImportedNativeClassRef] = []
    seen: set[str] = set()

    for index, name in _ordered_names(names):
        key = str(name).strip()
        if not key:
            raise ValidationError("Один из классов модели пустой")
        if key in seen:
            raise ValidationError(f"Класс '{key}' повторяется в модели")
        seen.add(key)
        result.append(ImportedNativeClassRef(index=index, key=key))

    if not result:
        raise ValidationError("В модели нет классов")

    return result


def _load_yolo_model(weights_path: Path) -> Any:
    if not weights_path.is_file():
        raise ValidationError(f"Файл модели не найден: {weights_path}")
    try:
        from ultralytics import YOLO

        return YOLO(str(weights_path))
    except Exception as exc:  # noqa: BLE001
        raise ValidationError("Не удалось открыть модель YOLO") from exc


def _ensure_segmentation_model(model: Any) -> None:
    task = getattr(model, "task", None) or getattr(
        getattr(model, "model", None), "task", None
    )
    if str(task or "").strip().lower() != "segment":
        raise ValidationError("Выберите сегментационную YOLO-модель в формате .pt")


def _ordered_names(names: Any) -> list[tuple[int, str]]:
    if isinstance(names, dict):
        return [
            (int(index), str(name))
            for index, name in sorted(names.items(), key=lambda item: int(item[0]))
        ]
    if isinstance(names, list):
        return list(enumerate(str(name) for name in names))
    raise ValidationError("Не удалось прочитать классы модели")
