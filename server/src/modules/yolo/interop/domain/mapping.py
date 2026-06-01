from __future__ import annotations

from uuid import UUID

from modules.yolo.interop.domain.types import (
    ImportedClassSuggestion,
    ImportedNativeClassPreview,
    ImportedNativeClassRef,
    ImportMappingDraft,
    ResolvedImportMapping,
    SegmentClassGroupRef,
    SegmentClassRef,
)


def build_preview_native_classes(
    *,
    native_classes: list[ImportedNativeClassRef],
    categories: list[SegmentClassGroupRef],
    ungrouped_classes: list[SegmentClassRef],
) -> list[ImportedNativeClassPreview]:
    all_classes = _flatten_segment_classes(categories, ungrouped_classes)
    by_id = {str(item.id): item for item in all_classes}
    by_name = _build_name_lookup(all_classes)

    return [
        ImportedNativeClassPreview(
            index=native.index,
            key=native.key,
            suggested=_find_suggested_segment_class(
                native_key=native.key,
                by_id=by_id,
                by_name=by_name,
            ),
        )
        for native in native_classes
    ]


def validate_import_mappings(
    *,
    mappings: list[ImportMappingDraft],
    native_classes: list[ImportedNativeClassRef],
    categories: list[SegmentClassGroupRef],
    ungrouped_classes: list[SegmentClassRef],
) -> None:
    if not mappings:
        raise ValueError("Нужно передать хотя бы одно сопоставление классов")

    native_keys = {item.key for item in native_classes}
    all_classes = _flatten_segment_classes(categories, ungrouped_classes)
    allowed_segment_ids = {item.id for item in all_classes}
    allowed_group_ids = {item.id for item in categories}
    existing_names_norm = {_norm(item.name) for item in all_classes}

    pending_new_names_norm: set[str] = set()
    seen_native: set[str] = set()
    seen_existing_segment_ids: set[UUID] = set()

    for item in mappings:
        if item.native_key not in native_keys:
            raise ValueError(
                f"Класс модели '{item.native_key}' отсутствует в загруженной модели"
            )

        if item.native_key in seen_native:
            raise ValueError(f"Класс модели '{item.native_key}' сопоставлен дважды")
        seen_native.add(item.native_key)

        if item.mode == "existing":
            if item.segment_class_id not in allowed_segment_ids:
                raise ValueError("Один из выбранных классов не принадлежит группе")
            if item.segment_class_id in seen_existing_segment_ids:
                raise ValueError(
                    "Один и тот же существующий класс нельзя выбрать дважды"
                )
            seen_existing_segment_ids.add(item.segment_class_id)
            continue

        if (
            item.new_class_group_id is not None
            and item.new_class_group_id not in allowed_group_ids
        ):
            raise ValueError("Выбранная группа классов не принадлежит текущей группе")

        if item.new_class_name is None:
            raise ValueError("Для нового класса нужно указать название")

        new_name_norm = _norm(item.new_class_name)
        if new_name_norm in existing_names_norm:
            raise ValueError(
                f"Класс '{item.new_class_name}' уже существует в текущей группе"
            )
        if new_name_norm in pending_new_names_norm:
            raise ValueError(
                f"Новый класс '{item.new_class_name}' указан несколько раз"
            )
        pending_new_names_norm.add(new_name_norm)


def build_resolved_import_mapping(
    *,
    native_classes: list[ImportedNativeClassRef],
    target_by_native: dict[str, SegmentClassRef],
) -> ResolvedImportMapping:
    class_keys: list[str] = []
    class_meta: list[dict] = []

    for native in native_classes:
        segment_class = target_by_native.get(native.key)
        if segment_class is None:
            continue

        internal_key = str(segment_class.id)
        class_keys.append(internal_key)
        class_meta.append(
            {
                "id": internal_key,
                "key": internal_key,
                "name": segment_class.name,
                "index": len(class_meta),
                "class_group_id": (
                    str(segment_class.class_group_id)
                    if segment_class.class_group_id is not None
                    else None
                ),
                "hue": segment_class.hue,
                "native_key": native.key,
                "native_index": native.index,
                "source": "imported",
            }
        )

    if not class_keys:
        raise ValueError("Нужно сопоставить хотя бы один класс модели")

    return ResolvedImportMapping(
        class_keys=class_keys,
        class_meta=class_meta,
        ignored_native_classes=[
            native.key
            for native in native_classes
            if native.key not in target_by_native
        ],
    )


def _flatten_segment_classes(
    categories: list[SegmentClassGroupRef],
    ungrouped_classes: list[SegmentClassRef],
) -> list[SegmentClassRef]:
    return [
        *(item for category in categories for item in category.segment_classes),
        *ungrouped_classes,
    ]


def _build_name_lookup(
    classes: list[SegmentClassRef],
) -> dict[str, SegmentClassRef | None]:
    by_name: dict[str, SegmentClassRef | None] = {}

    for item in classes:
        key = _norm(item.name)
        by_name[key] = item if key not in by_name else None

    return by_name


def _find_suggested_segment_class(
    *,
    native_key: str,
    by_id: dict[str, SegmentClassRef],
    by_name: dict[str, SegmentClassRef | None],
) -> ImportedClassSuggestion | None:
    parsed = _parse_uuid(native_key)
    if parsed is not None and (found := by_id.get(str(parsed))) is not None:
        return ImportedClassSuggestion(
            segment_class_id=found.id,
            segment_class_name=found.name,
            match_reason="uuid",
        )

    if (found_by_name := by_name.get(_norm(native_key))) is not None:
        return ImportedClassSuggestion(
            segment_class_id=found_by_name.id,
            segment_class_name=found_by_name.name,
            match_reason="name",
        )

    return None


def _parse_uuid(value: str) -> UUID | None:
    try:
        return UUID(value.strip())
    except (TypeError, ValueError, AttributeError):
        return None


def _norm(value: str) -> str:
    return value.strip().casefold()
