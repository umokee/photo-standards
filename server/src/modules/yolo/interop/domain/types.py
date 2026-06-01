from __future__ import annotations

from dataclasses import dataclass
from typing import Literal
from uuid import UUID


@dataclass(slots=True, frozen=True)
class ImportedClassSuggestion:
    segment_class_id: UUID
    segment_class_name: str
    match_reason: Literal["uuid", "name"]


@dataclass(slots=True, frozen=True)
class ImportedNativeClassRef:
    index: int
    key: str


@dataclass(slots=True, frozen=True)
class ImportedNativeClassPreview:
    index: int
    key: str
    suggested: ImportedClassSuggestion | None = None


@dataclass(slots=True, frozen=True)
class ImportMappingDraft:
    mode: Literal["existing", "new"]
    native_key: str
    segment_class_id: UUID | None = None
    new_class_name: str | None = None
    new_class_hue: int | None = None
    new_class_group_id: UUID | None = None


@dataclass(slots=True, frozen=True)
class SegmentClassRef:
    id: UUID
    name: str
    hue: int
    class_group_id: UUID | None = None


@dataclass(slots=True, frozen=True)
class SegmentClassGroupRef:
    id: UUID
    segment_classes: list[SegmentClassRef]


@dataclass(slots=True, frozen=True)
class ResolvedImportMapping:
    class_keys: list[str]
    class_meta: list[dict]
    ignored_native_classes: list[str]


@dataclass(slots=True, frozen=True)
class ImportTaskPayloadData:
    group_id: UUID
    model_id: UUID
    version: int
    architecture: str
    imgsz: int
    batch_size: int
    activate: bool
    task_root: str
    source_pt_path: str
    mappings: list[ImportMappingDraft]
