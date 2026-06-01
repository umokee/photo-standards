from uuid import UUID

from app.dependencies import DbSession
from fastapi import APIRouter, File, UploadFile

from .schemas import (
    StandardCreate,
    StandardDetailResponse,
    StandardImageDetailResponse,
    StandardImageResponse,
    StandardMutationResponse,
    StandardUpdate,
)
from .service import (
    create_standard,
    delete_image,
    delete_standard,
    get_image,
    get_standard,
    set_reference,
    update_standard,
    upload_images,
)

router = APIRouter(prefix="/standards", tags=["standards"])


@router.get("/{standard_id}", response_model=StandardDetailResponse)
async def get_standard_route(
    db: DbSession,
    standard_id: UUID,
) -> StandardDetailResponse:
    return await get_standard(db, standard_id)


@router.post("", response_model=StandardMutationResponse, status_code=201)
async def create_standard_route(
    db: DbSession,
    data: StandardCreate,
) -> StandardMutationResponse:
    return await create_standard(db, data)


@router.put("/{standard_id}", response_model=StandardMutationResponse)
async def update_standard_route(
    db: DbSession,
    standard_id: UUID,
    data: StandardUpdate,
) -> StandardMutationResponse:
    return await update_standard(db, standard_id, data)


@router.delete("/{standard_id}", status_code=204)
async def delete_standard_route(
    db: DbSession,
    standard_id: UUID,
) -> None:
    await delete_standard(db, standard_id)


@router.post(
    "/{standard_id}/images",
    response_model=list[StandardImageResponse],
    status_code=201,
)
async def upload_images_route(
    db: DbSession,
    standard_id: UUID,
    images: list[UploadFile] = File(...),  # noqa: B008
) -> list[StandardImageResponse]:
    return await upload_images(db, standard_id, images)


@router.patch("/images/{image_id}/reference", response_model=StandardImageResponse)
async def set_reference_route(
    db: DbSession,
    image_id: UUID,
) -> StandardImageResponse:
    return await set_reference(db, image_id)


@router.get("/images/{image_id}", response_model=StandardImageDetailResponse)
async def get_image_route(
    db: DbSession,
    image_id: UUID,
) -> StandardImageDetailResponse:
    return await get_image(db, image_id)


@router.delete("/images/{image_id}", status_code=204)
async def delete_image_route(
    db: DbSession,
    image_id: UUID,
) -> None:
    await delete_image(db, image_id)
