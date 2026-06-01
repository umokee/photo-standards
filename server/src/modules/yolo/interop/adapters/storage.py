from __future__ import annotations

import json
import os
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from uuid import UUID

from app.exception import ValidationError
from infra.storage.file_storage import ensure_parent_dir, resolve_storage_path
from modules.yolo.interop.api import schemas as interop_schemas
from modules.yolo.interop.constants import (
    ALLOWED_IMPORT_SUFFIXES,
    CLASS_MANIFEST_SUFFIX,
    EXPORT_ARCHIVE_SUFFIX,
)
from modules.yolo.interop.domain.types import ImportMappingDraft, ImportTaskPayloadData
from modules.yolo.training.models import MlModel


@dataclass(slots=True)
class ModelArtifactPaths:
    pt: Path
    pt_rel: str
    manifest: Path
    manifest_rel: str


@dataclass(slots=True)
class ImportTaskPaths:
    task_root: Path
    task_root_rel: str
    source_pt: Path
    source_pt_rel: str

    @classmethod
    def from_payload(cls, payload: ImportTaskPayloadData) -> ImportTaskPaths:
        return cls(
            task_root=resolve_storage_path(payload.task_root),
            task_root_rel=payload.task_root,
            source_pt=resolve_storage_path(payload.source_pt_path),
            source_pt_rel=payload.source_pt_path,
        )


@dataclass(slots=True)
class ExportArchive:
    path: Path
    filename: str


def normalized_suffix(filename: str | None) -> str:
    suffix = Path(filename).suffix.lower() if filename else ".pt"
    suffix = suffix or ".pt"

    if suffix not in ALLOWED_IMPORT_SUFFIXES:
        raise ValidationError(
            "После отказа от ONNX можно импортировать только .pt модели"
        )

    return suffix


def build_temporary_import_path(*, suffix: str) -> Path:
    fd, tmp_name = tempfile.mkstemp(prefix="yolo-import-", suffix=suffix)
    os.close(fd)
    tmp_path = Path(tmp_name)
    tmp_path.unlink(missing_ok=True)
    return tmp_path


def build_import_task_paths(
    *,
    group_id: UUID,
    task_id: UUID,
    suffix: str,
) -> ImportTaskPaths:
    task_root_rel = (
        Path("models") / str(group_id) / "tasks" / str(task_id) / "import"
    ).as_posix()
    source_pt_rel = (Path(task_root_rel) / f"source{suffix}").as_posix()

    return ImportTaskPaths(
        task_root=resolve_storage_path(task_root_rel),
        task_root_rel=task_root_rel,
        source_pt=resolve_storage_path(source_pt_rel),
        source_pt_rel=source_pt_rel,
    )


def build_model_artifact_paths(
    *,
    group_id: UUID,
    version: int,
) -> ModelArtifactPaths:
    base = Path("models") / str(group_id)
    pt_rel = (base / f"v{version}.pt").as_posix()
    manifest_rel = (base / f"v{version}{CLASS_MANIFEST_SUFFIX}").as_posix()

    return ModelArtifactPaths(
        pt=resolve_storage_path(pt_rel),
        pt_rel=pt_rel,
        manifest=resolve_storage_path(manifest_rel),
        manifest_rel=manifest_rel,
    )


def write_bytes(path: Path, data: bytes) -> None:
    ensure_parent_dir(path)
    path.write_bytes(data)


def move_file(*, source: Path, target: Path) -> None:
    ensure_parent_dir(target)
    shutil.move(str(source), str(target))


def unlink_path(path: Path) -> None:
    path.unlink(missing_ok=True)


def remove_tree(path: Path) -> None:
    shutil.rmtree(path, ignore_errors=True)


def serialize_import_task_payload(payload: ImportTaskPayloadData) -> dict:
    schema = interop_schemas.ModelImportTaskPayload(
        group_id=payload.group_id,
        model_id=payload.model_id,
        version=payload.version,
        architecture=payload.architecture,
        imgsz=payload.imgsz,
        batch_size=payload.batch_size,
        activate=payload.activate,
        task_root=payload.task_root,
        source_pt_path=payload.source_pt_path,
        mappings=[_serialize_mapping(mapping) for mapping in payload.mappings],
    )
    return schema.model_dump(mode="json")


def parse_import_task_payload(raw: dict) -> ImportTaskPayloadData:
    schema = interop_schemas.ModelImportTaskPayload.model_validate(raw)
    return ImportTaskPayloadData(
        group_id=schema.group_id,
        model_id=schema.model_id,
        version=schema.version,
        architecture=schema.architecture,
        imgsz=schema.imgsz,
        batch_size=schema.batch_size,
        activate=schema.activate,
        task_root=schema.task_root,
        source_pt_path=schema.source_pt_path,
        mappings=[_to_mapping_draft(mapping) for mapping in schema.mappings],
    )


def write_class_manifest(
    *,
    model: MlModel,
    artifact_paths: ModelArtifactPaths,
) -> Path:
    if model.version is None:
        raise ValidationError("Нельзя сформировать manifest для модели без version")

    class_meta = list(model.class_meta or [])
    if not class_meta:
        raise ValidationError("У модели отсутствует class_meta")

    payload = {
        "format": "photo-standards-db.yolo-class-manifest",
        "format_version": 1,
        "model": {
            "id": str(model.id),
            "group_id": str(model.group_id),
            "version": model.version,
            "architecture": model.architecture,
            "imgsz": model.imgsz,
            "pt_path": artifact_paths.pt.name,
        },
        "classes": [
            _build_manifest_class_entry(index, item)
            for index, item in enumerate(class_meta)
        ],
    }

    ensure_parent_dir(artifact_paths.manifest)
    artifact_paths.manifest.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return artifact_paths.manifest


def build_export_archive(
    *,
    model: MlModel,
    artifact_paths: ModelArtifactPaths,
) -> ExportArchive:
    if model.version is None:
        raise ValidationError("Модель ещё не готова к экспорту")

    if not artifact_paths.pt.is_file():
        raise ValidationError("Файл .pt для экспорта не найден")

    write_class_manifest(model=model, artifact_paths=artifact_paths)

    with tempfile.NamedTemporaryFile(
        prefix="yolo-model-",
        suffix=EXPORT_ARCHIVE_SUFFIX,
        delete=False,
    ) as tmp:
        archive_path = Path(tmp.name)

    with zipfile.ZipFile(
        archive_path,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
    ) as archive:
        archive.write(artifact_paths.pt, arcname=artifact_paths.pt.name)
        archive.write(artifact_paths.manifest, arcname=artifact_paths.manifest.name)

    return ExportArchive(
        path=archive_path,
        filename=f"yolo-model-v{model.version}{EXPORT_ARCHIVE_SUFFIX}",
    )


def _serialize_mapping(mapping: ImportMappingDraft) -> dict[str, Any]:
    data: dict[str, Any] = {
        "mode": mapping.mode,
        "native_key": mapping.native_key,
    }

    if mapping.mode == "existing":
        data["segment_class_id"] = mapping.segment_class_id
    else:
        data["new_class_name"] = mapping.new_class_name
        data["new_class_hue"] = mapping.new_class_hue
        data["new_class_group_id"] = mapping.new_class_group_id

    return data


def _to_mapping_draft(mapping: Any) -> ImportMappingDraft:
    return ImportMappingDraft(
        mode=mapping.mode,
        native_key=mapping.native_key,
        segment_class_id=getattr(mapping, "segment_class_id", None),
        new_class_name=getattr(mapping, "new_class_name", None),
        new_class_hue=getattr(mapping, "new_class_hue", None),
        new_class_group_id=getattr(mapping, "new_class_group_id", None),
    )


def _build_manifest_class_entry(index: int, item: dict[str, Any]) -> dict[str, Any]:
    model_key = _str_or_none(item.get("native_key")) or _str_or_none(item.get("key"))
    segment_class_id = _str_or_none(item.get("id")) or _str_or_none(item.get("key"))

    return {
        "index": _int_or_default(item.get("index"), index),
        "model_key": model_key or "",
        "model_name": _str_or_none(item.get("name")) or model_key or "",
        "segment_class_id": segment_class_id,
        "segment_class_name": _str_or_none(item.get("name")),
        "class_group_id": _str_or_none(item.get("class_group_id")),
        "hue": item.get("hue"),
        "native_index": item.get("native_index"),
        "native_key": _str_or_none(item.get("native_key")),
        "source": _str_or_none(item.get("source")),
    }


def _str_or_none(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _int_or_default(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
