from __future__ import annotations

from uuid import UUID

from modules.core.groups import crud as groups_crud
from modules.core.segments import crud as segments_crud
from modules.core.segments.models import SegmentClass
from modules.yolo.interop.domain.types import SegmentClassGroupRef, SegmentClassRef
from modules.yolo.training.adapters import repository as training_repository
from modules.yolo.training.adapters import storage as training_storage
from modules.yolo.training.models import MlModel
from sqlalchemy.ext.asyncio import AsyncSession


async def ensure_group_exists(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> None:
    await groups_crud.get_group(db, group_id=group_id)


async def load_group_segment_tree(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> tuple[list[SegmentClassGroupRef], list[SegmentClassRef]]:
    categories, ungrouped_classes = await segments_crud.get_group_tree(
        db,
        group_id=group_id,
    )

    return (
        [
            SegmentClassGroupRef(
                id=category.id,
                segment_classes=[
                    SegmentClassRef(
                        id=segment_class.id,
                        name=segment_class.name,
                        hue=segment_class.hue,
                        class_group_id=segment_class.class_group_id,
                    )
                    for segment_class in category.segment_classes
                ],
            )
            for category in categories
        ],
        [
            SegmentClassRef(
                id=segment_class.id,
                name=segment_class.name,
                hue=segment_class.hue,
                class_group_id=segment_class.class_group_id,
            )
            for segment_class in ungrouped_classes
        ],
    )


async def get_next_model_version(
    db: AsyncSession,
    *,
    group_id: UUID,
) -> int:
    return await training_repository.get_next_model_version(db, group_id=group_id)


async def create_import_model(
    db: AsyncSession,
    *,
    group_id: UUID,
    architecture: str,
    imgsz: int,
    batch_size: int,
) -> MlModel:
    model = MlModel(
        group_id=group_id,
        architecture=architecture,
        imgsz=imgsz,
        batch_size=batch_size,
    )
    db.add(model)
    await db.flush()
    return model


async def get_model(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> MlModel:
    return await training_repository.get_model(db, model_id=model_id)


async def create_segment_class(
    db: AsyncSession,
    *,
    group_id: UUID,
    name: str,
    hue: int,
    class_group_id: UUID | None,
) -> SegmentClassRef:
    await segments_crud.ensure_segment_class_name_unique(
        db,
        group_id=group_id,
        name=name,
    )

    segment_class = SegmentClass(
        group_id=group_id,
        class_group_id=class_group_id,
        name=name,
        hue=hue,
    )
    db.add(segment_class)
    await db.flush()

    return SegmentClassRef(
        id=segment_class.id,
        name=segment_class.name,
        hue=segment_class.hue,
        class_group_id=segment_class.class_group_id,
    )


def apply_import_to_model(
    model: MlModel,
    *,
    version: int,
    architecture: str,
    imgsz: int,
    batch_size: int,
    weights_path: str,
    class_keys: list[str],
    class_meta: list[dict],
) -> None:
    model.version = version
    model.weights_path = weights_path
    model.architecture = architecture
    model.imgsz = imgsz
    model.batch_size = batch_size
    model.epochs = None
    model.num_classes = len(class_keys)
    model.class_keys = class_keys
    model.class_meta = class_meta
    model.metrics = None
    model.train_ratio = None
    model.val_ratio = None
    model.test_ratio = None
    model.total_images = None
    model.train_count = None
    model.val_count = None
    model.test_count = None
    model.is_active = False


async def activate_model(
    db: AsyncSession,
    *,
    model: MlModel,
) -> MlModel:
    training_storage.ensure_model_weights_ready(model)
    model = await training_repository.set_active_model(db, model=model)
    await db.commit()
    await db.refresh(model)
    return model


async def reset_failed_model(
    db: AsyncSession,
    *,
    model_id: UUID,
) -> None:
    model = await db.get(MlModel, model_id)
    if model is None:
        return

    model.version = None
    model.weights_path = None
    model.num_classes = None
    model.class_keys = None
    model.class_meta = None
    model.metrics = None
    model.trained_at = None
    model.is_active = False
    await db.commit()
