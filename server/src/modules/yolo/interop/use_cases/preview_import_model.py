from __future__ import annotations

import asyncio
from dataclasses import dataclass

from modules.yolo.interop.adapters import repository, storage, yolo
from modules.yolo.interop.api.schemas import ModelImportPreviewParams
from modules.yolo.interop.domain import mapping as mapping_domain
from modules.yolo.interop.domain.types import ImportedNativeClassPreview
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(slots=True, frozen=True)
class ModelImportPreviewResult:
    native_classes: list[ImportedNativeClassPreview]


async def preview_import_model(
    db: AsyncSession,
    *,
    data: ModelImportPreviewParams,
) -> ModelImportPreviewResult:
    await repository.ensure_group_exists(db, group_id=data.group_id)

    tmp_path = storage.build_temporary_import_path(
        suffix=storage.normalized_suffix(data.weights.filename)
    )
    try:
        storage.write_bytes(tmp_path, await data.weights.read())

        native_classes = await asyncio.to_thread(yolo.read_native_classes, tmp_path)
        categories, ungrouped_classes = await repository.load_group_segment_tree(
            db,
            group_id=data.group_id,
        )

        return ModelImportPreviewResult(
            native_classes=mapping_domain.build_preview_native_classes(
                native_classes=native_classes,
                categories=categories,
                ungrouped_classes=ungrouped_classes,
            )
        )
    finally:
        storage.unlink_path(tmp_path)
