from __future__ import annotations

from uuid import UUID

from app.exception import NotFoundError, ValidationError
from infra.storage.file_storage import resolve_storage_path
from modules.core.groups import crud as groups_crud
from modules.core.groups.models import Group
from modules.core.segments.models import SegmentAnnotation
from modules.core.standards.models import Standard, StandardImage
from modules.tasks.constants import tasks as tasks_constants
from modules.tasks.models import Task
from modules.yolo.training.domain.types import (
    TrainingAnnotation,
    TrainingData,
    TrainingImage,
    TrainingStandard,
)
from modules.yolo.training.models import MlModel
from PIL import Image
from sqlalchemy import case, desc, exists, select, update
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import selectinload


async def get_model(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> MlModel:
    model = await db.get(MlModel, model_id)
    if model is None:
        raise NotFoundError("Модель", model_id)
    return model


async def list_models(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> list[MlModel]:
    non_terminal_training_statuses = [
        *tasks_constants.statuses.active,
        tasks_constants.statuses.paused,
    ]

    has_active_training_task = exists(
        select(Task.id).where(
            Task.entity_type == "ml_model",
            Task.entity_id == MlModel.id,
            Task.type == tasks_constants.types.training,
            Task.status.in_(non_terminal_training_statuses),
        )
    )

    result = await db.execute(
        select(MlModel)
        .where(MlModel.group_id == group_id)
        .order_by(
            case((has_active_training_task, 0), else_=1),
            MlModel.version.desc().nulls_last(),
            MlModel.created_at.desc(),
        )
    )
    return list(result.scalars().all())


async def get_active_model(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> MlModel:
    result = await db.execute(
        select(MlModel)
        .where(MlModel.group_id == group_id, MlModel.is_active.is_(True))
        .limit(1)
    )
    model = result.scalar_one_or_none()
    if model is None:
        raise ValidationError("Для группы нет активной модели")
    return model


async def get_next_model_version(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> int:
    result = await db.execute(
        select(MlModel.version)
        .where(MlModel.group_id == group_id)
        .order_by(desc(MlModel.version))
        .limit(1)
    )
    last = result.scalar_one_or_none()
    return (last or 0) + 1


async def create_model(
    db: AsyncSession,
    *,
    group_id: UUID,
    architecture: str,
    epochs: int,
    imgsz: int,
    batch_size: int,
    train_ratio: int,
    val_ratio: int,
    test_ratio: int,
) -> MlModel:
    model = MlModel(
        group_id=group_id,
        architecture=architecture,
        version=None,
        epochs=epochs,
        imgsz=imgsz,
        batch_size=batch_size,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )
    db.add(model)
    await db.flush()
    return model


async def set_active_model(
    db: AsyncSession,
    *,
    model: MlModel,
) -> MlModel:
    await db.execute(
        update(MlModel)
        .where(MlModel.group_id == model.group_id, MlModel.id != model.id)
        .values(is_active=False)
    )
    model.is_active = True
    await db.flush()
    return model


async def delete_model_record(
    db: AsyncSession,
    *,
    model: MlModel,
) -> None:
    await db.delete(model)
    await db.flush()


async def has_blocking_training_task(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> bool:
    blocking = frozenset(tasks_constants.statuses.active) | {
        tasks_constants.statuses.paused,
    }
    result = await db.execute(
        select(Task.id)
        .where(
            Task.entity_type == "ml_model",
            Task.entity_id == model_id,
            Task.type == tasks_constants.types.training,
            Task.status.in_(blocking),
        )
        .limit(1)
    )
    return result.scalar_one_or_none() is not None


async def load_training_data(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> TrainingData:
    group = await db.get(
        Group,
        group_id,
        options=[selectinload(Group.segment_classes)],
    )
    if group is None:
        raise NotFoundError("Группа", group_id)

    stmt = (
        select(Standard)
        .options(
            selectinload(Standard.images)
            .selectinload(StandardImage.annotations)
            .selectinload(SegmentAnnotation.segment_class),
        )
        .where(Standard.group_id == group_id)
        .order_by(Standard.created_at.asc())
    )
    standards = list((await db.execute(stmt)).scalars().all())
    return _build_training_data(group, standards)


async def ensure_group_exists(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> None:
    await groups_crud.get_group(db, group_id=group_id)


def _build_training_data(
    group: Group,
    standards: list[Standard],
) -> TrainingData:
    sorted_classes = sorted(group.segment_classes, key=lambda item: item.name.lower())

    class_index_by_id: dict[UUID, int] = {}
    class_meta: list[dict] = []

    for index, segment_class in enumerate(sorted_classes):
        class_index_by_id[segment_class.id] = index
        class_meta.append(
            {
                "id": str(segment_class.id),
                "key": str(segment_class.id),
                "name": segment_class.name,
                "index": index,
                "class_group_id": str(segment_class.class_group_id)
                if segment_class.class_group_id
                else None,
                "hue": segment_class.hue,
            }
        )

    standards_data: list[TrainingStandard] = []
    for standard in standards:
        images_data: list[TrainingImage] = []

        for image in standard.images:
            width, height = _image_size(image.image_path)
            annotations: list[TrainingAnnotation] = []

            for annotation in image.annotations:
                if not annotation.points or annotation.segment_class is None:
                    continue

                class_index = class_index_by_id.get(annotation.segment_class_id)
                if class_index is None:
                    continue

                annotations.append(
                    TrainingAnnotation(
                        segment_class_id=annotation.segment_class_id,
                        class_key=str(annotation.segment_class.id),
                        class_index=class_index,
                        points=annotation.points,
                    )
                )

            images_data.append(
                TrainingImage(
                    image_id=image.id,
                    image_path=image.image_path,
                    width=width,
                    height=height,
                    is_annotated=bool(annotations),
                    annotations=annotations,
                )
            )

        standards_data.append(
            TrainingStandard(
                standard_id=standard.id,
                standard_name=standard.name,
                angle=standard.angle,
                images=images_data,
            )
        )

    return TrainingData(
        group_id=group.id,
        group_name=group.name,
        class_keys=[item["key"] for item in class_meta],
        class_meta=class_meta,
        standards=standards_data,
    )


def _image_size(image_path: str) -> tuple[int, int]:
    with Image.open(resolve_storage_path(image_path)) as image:
        return image.size
