from uuid import UUID

from app.dependencies import DbSession
from fastapi import APIRouter

from .schemas import (
    AnnotationSave,
    SaveSegmentClassesRequest,
    SaveSegmentClassesResponse,
    SegmentClassWithPointsResponse,
)
from .service import (
    list_segment_classes,
    save_annotation,
    save_segment_classes,
)

router = APIRouter(prefix="/segment-classes", tags=["segment-classes"])


@router.get("/group/{group_id}", response_model=SaveSegmentClassesResponse)
async def list_segment_classes_route(
    db: DbSession,
    group_id: UUID,
) -> SaveSegmentClassesResponse:
    return await list_segment_classes(db, group_id)


@router.put("/group/{group_id}", response_model=SaveSegmentClassesResponse)
async def save_segment_classes_route(
    db: DbSession,
    group_id: UUID,
    data: SaveSegmentClassesRequest,
) -> SaveSegmentClassesResponse:
    return await save_segment_classes(db, group_id, data)


@router.put(
    "/{segment_class_id}/annotations/{image_id}",
    response_model=SegmentClassWithPointsResponse,
)
async def save_annotation_route(
    db: DbSession,
    segment_class_id: UUID,
    image_id: UUID,
    data: AnnotationSave,
) -> SegmentClassWithPointsResponse:
    return await save_annotation(db, segment_class_id, image_id, data)
