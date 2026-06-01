from uuid import UUID

from app.dependencies import DbSession
from fastapi import APIRouter

from . import service
from .schemas import SamClickRequest, SamClickResponse

router = APIRouter(prefix="/sam", tags=["sam"])


@router.post("/predict", response_model=SamClickResponse)
async def predict(
    db: DbSession,
    data: SamClickRequest,
) -> SamClickResponse:
    return await service.predict_from_clicks(db, data)


@router.post("/warmup/{image_id}", status_code=204)
async def sam_warmup(
    db: DbSession,
    image_id: UUID,
) -> None:
    await service.warmup_image(db, image_id)
