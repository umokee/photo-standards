from __future__ import annotations

from constants_base import ConstModel, ValuesCollection


class UploadAllowedTypes(ValuesCollection[str]):
    pass


class UploadsConstants(ConstModel):
    allowed_types: UploadAllowedTypes
    max_size_bytes: int


uploads = UploadsConstants(
    allowed_types=UploadAllowedTypes(
        values=("image/jpeg", "image/png", "image/jpg"),
    ),
    max_size_bytes=20971520,
)
