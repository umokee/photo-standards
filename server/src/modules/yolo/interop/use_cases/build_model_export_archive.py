from __future__ import annotations

import asyncio
from uuid import UUID

from app.exception import ValidationError
from modules.yolo.interop.adapters import repository, storage
from sqlalchemy.ext.asyncio import AsyncSession


async def build_model_export_archive(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> storage.ExportArchive:
    model = await repository.get_model(db, model_id=model_id)

    if model.version is None:
        raise ValidationError("Экспорт доступен только для готовой версии модели")

    artifact_paths = storage.build_model_artifact_paths(
        group_id=model.group_id,
        version=model.version,
    )

    return await asyncio.to_thread(
        storage.build_export_archive,
        model=model,
        artifact_paths=artifact_paths,
    )
