from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import UUID

from app.exception import ValidationError as AppValidationError
from fastapi import UploadFile
from modules.core.segments.constants import segments
from modules.yolo.interop.constants import ALLOWED_IMPORT_SUFFIXES
from modules.yolo.training.constants import training
from pydantic import (
    AfterValidator,
    BaseModel,
    ConfigDict,
    Field,
    StringConstraints,
    TypeAdapter,
    field_validator,
)
from pydantic import (
    ValidationError as PydanticValidationError,
)

NonEmptyStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=255),
]

MappingsJsonStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1),
]

StoragePathStr = Annotated[
    str,
    StringConstraints(strip_whitespace=True, min_length=1, max_length=500),
]

HueValue = Annotated[
    int,
    Field(ge=segments.hue.min, le=segments.hue.max),
]

BatchSizeValue = Annotated[
    int,
    Field(ge=training.batch_size.min, le=training.batch_size.max),
]


def _check_architecture(value: str) -> str:
    if value not in training.architectures:
        raise ValueError(
            f"Архитектура должна быть одной из: {', '.join(training.architectures)}"
        )
    return value


def _check_imgsz(value: int) -> int:
    if value not in training.image_size:
        raise ValueError(
            f"Размер изображения должен быть одним из: "
            f"{', '.join(map(str, training.image_size))}"
        )
    return value


def _check_import_file(weights: UploadFile) -> UploadFile:
    suffix = Path(weights.filename or "").suffix.lower()
    if suffix not in ALLOWED_IMPORT_SUFFIXES:
        raise ValueError("Поддерживается импорт только .pt моделей YOLO")
    return weights


Architecture = Annotated[NonEmptyStr, AfterValidator(_check_architecture)]
Imgsz = Annotated[int, AfterValidator(_check_imgsz)]


class ImportedClassSuggestionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    segment_class_id: UUID
    segment_class_name: NonEmptyStr
    match_reason: Literal["uuid", "name"]


class ImportedNativeClass(BaseModel):
    model_config = ConfigDict(extra="forbid")

    index: int = Field(ge=0)
    key: NonEmptyStr


class ImportedNativeClassResponse(ImportedNativeClass):
    suggested: ImportedClassSuggestionResponse | None = None


class ModelImportPreviewResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    native_classes: list[ImportedNativeClassResponse] = Field(default_factory=list)


class ModelImportStartResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    task_id: UUID
    model_id: UUID


class ExistingImportedClassMappingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["existing"]
    native_key: NonEmptyStr
    segment_class_id: UUID


class NewImportedClassMappingDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")

    mode: Literal["new"]
    native_key: NonEmptyStr
    new_class_name: NonEmptyStr
    new_class_hue: HueValue = segments.hue.default
    new_class_group_id: UUID | None = None


ImportedClassMappingDraft = Annotated[
    ExistingImportedClassMappingDraft | NewImportedClassMappingDraft,
    Field(discriminator="mode"),
]


class ModelImportPreviewParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    group_id: UUID
    weights: UploadFile

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: UploadFile) -> UploadFile:
        return _check_import_file(value)


class ModelImportCreateParams(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    group_id: UUID
    architecture: Architecture
    imgsz: Imgsz
    batch_size: BatchSizeValue = training.batch_size.default
    activate: bool = False
    mappings_json: MappingsJsonStr
    weights: UploadFile

    @field_validator("weights")
    @classmethod
    def validate_weights(cls, value: UploadFile) -> UploadFile:
        return _check_import_file(value)


class ModelImportTaskPayload(BaseModel):
    model_config = ConfigDict(extra="forbid")

    group_id: UUID
    model_id: UUID
    version: int = Field(ge=1)
    architecture: Architecture
    imgsz: Imgsz
    batch_size: BatchSizeValue = training.batch_size.default
    activate: bool = False
    task_root: StoragePathStr
    source_pt_path: StoragePathStr
    mappings: list[ImportedClassMappingDraft] = Field(min_length=1)


_IMPORT_MAPPINGS_ADAPTER = TypeAdapter(list[ImportedClassMappingDraft])


def parse_import_preview_params(
    *,
    group_id: UUID,
    weights: UploadFile,
) -> ModelImportPreviewParams:
    try:
        return ModelImportPreviewParams.model_validate(
            {
                "group_id": group_id,
                "weights": weights,
            }
        )
    except PydanticValidationError as exc:
        raise AppValidationError(_first_pydantic_error(exc)) from exc


def parse_import_create_params(
    *,
    group_id: UUID,
    architecture: str,
    imgsz: int,
    batch_size: int,
    activate: bool,
    mappings_json: str,
    weights: UploadFile,
) -> ModelImportCreateParams:
    try:
        return ModelImportCreateParams.model_validate(
            {
                "group_id": group_id,
                "architecture": architecture,
                "imgsz": imgsz,
                "batch_size": batch_size,
                "activate": activate,
                "mappings_json": mappings_json,
                "weights": weights,
            }
        )
    except PydanticValidationError as exc:
        raise AppValidationError(_first_pydantic_error(exc)) from exc


def parse_import_mappings(
    raw: str | list[dict[str, Any]] | list[ImportedClassMappingDraft],
) -> list[ImportedClassMappingDraft]:
    if isinstance(raw, str):
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise AppValidationError("Некорректный JSON в mappings_json") from exc
    else:
        data = raw

    try:
        return _IMPORT_MAPPINGS_ADAPTER.validate_python(data)
    except PydanticValidationError as exc:
        raise AppValidationError("Некорректная структура mappings_json") from exc


def _first_pydantic_error(exc: PydanticValidationError) -> str:
    issue = exc.errors()[0] if exc.errors() else None
    message = issue.get("msg") if issue else None
    return str(message or "Проверьте введенные данные")
