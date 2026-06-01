from __future__ import annotations

import asyncio
from dataclasses import dataclass
from uuid import UUID, uuid4

from app.exception import ValidationError
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.service import create_task
from modules.yolo.interop.adapters import queue, repository, storage, yolo
from modules.yolo.interop.api.schemas import (
    ModelImportCreateParams,
    parse_import_mappings,
)
from modules.yolo.interop.domain import mapping as mapping_domain
from modules.yolo.interop.domain.types import ImportMappingDraft, ImportTaskPayloadData
from sqlalchemy.ext.asyncio import AsyncSession


@dataclass(slots=True, frozen=True)
class ModelImportStartResult:
    task_id: UUID
    model_id: UUID


async def create_import_task(
    db: AsyncSession,
    *,
    data: ModelImportCreateParams,
) -> ModelImportStartResult:
    await repository.ensure_group_exists(db, group_id=data.group_id)

    mappings = _to_mapping_drafts(parse_import_mappings(data.mappings_json))
    task_id = uuid4()
    paths = storage.build_import_task_paths(
        group_id=data.group_id,
        task_id=task_id,
        suffix=storage.normalized_suffix(data.weights.filename),
    )

    try:
        storage.write_bytes(paths.source_pt, await data.weights.read())

        native_classes = await asyncio.to_thread(
            yolo.read_native_classes, paths.source_pt
        )
        categories, ungrouped_classes = await repository.load_group_segment_tree(
            db,
            group_id=data.group_id,
        )

        try:
            mapping_domain.validate_import_mappings(
                mappings=mappings,
                native_classes=native_classes,
                categories=categories,
                ungrouped_classes=ungrouped_classes,
            )
        except ValueError as exc:
            raise ValidationError(str(exc)) from exc

        version = await repository.get_next_model_version(db, group_id=data.group_id)
        model = await repository.create_import_model(
            db,
            group_id=data.group_id,
            architecture=data.architecture,
            imgsz=data.imgsz,
            batch_size=data.batch_size,
        )

        payload = ImportTaskPayloadData(
            group_id=data.group_id,
            model_id=model.id,
            version=version,
            architecture=data.architecture,
            imgsz=data.imgsz,
            batch_size=data.batch_size,
            activate=data.activate,
            task_root=paths.task_root_rel,
            source_pt_path=paths.source_pt_rel,
            mappings=mappings,
        )

        task = await create_task(
            db,
            type=tasks_constants.types.model_import,
            status=tasks_constants.statuses.pending,
            queue=tasks_constants.queues.gpu,
            priority=tasks_constants.priorities.model_import,
            task_id=task_id,
            payload=storage.serialize_import_task_payload(payload),
            entity_type="ml_model",
            entity_id=model.id,
            group_id=data.group_id,
            auto_resume=False,
            run_dir=paths.task_root_rel,
        )

        task = await queue.enqueue_model_import(db, task)

        return ModelImportStartResult(
            task_id=task.id,
            model_id=model.id,
        )
    except Exception:
        await db.rollback()
        storage.unlink_path(paths.source_pt)
        storage.remove_tree(paths.task_root)
        raise


def _to_mapping_drafts(mappings) -> list[ImportMappingDraft]:
    return [
        ImportMappingDraft(
            mode=item.mode,
            native_key=item.native_key,
            segment_class_id=getattr(item, "segment_class_id", None),
            new_class_name=getattr(item, "new_class_name", None),
            new_class_hue=getattr(item, "new_class_hue", None),
            new_class_group_id=getattr(item, "new_class_group_id", None),
        )
        for item in mappings
    ]
