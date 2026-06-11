import mimetypes
import shutil
from pathlib import Path

from app.config import settings
from app.exception import ValidationError
from fastapi import UploadFile
from infra.storage.constants import uploads


def resolve_storage_path(
    relative_path: str | Path,
) -> Path:
    path = Path(relative_path)
    if path.is_absolute():
        return path
    return settings.STORAGE_ROOT / path


def ensure_parent_dir(
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)


def prune_empty_dirs(
    path: str | Path,
    *,
    stop_at: str | Path | None = None,
) -> None:
    current = resolve_storage_path(path)
    if current.is_file():
        current = current.parent

    boundary = resolve_storage_path(stop_at) if stop_at is not None else settings.STORAGE_ROOT

    try:
        current.relative_to(boundary)
    except ValueError:
        return

    while current != boundary and current != current.parent:
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def delete_storage_file(
    relative_path: str | Path | None,
    *,
    prune: bool = True,
    stop_at: str | Path | None = None,
) -> None:
    if not relative_path:
        return

    absolute_path = resolve_storage_path(relative_path)
    absolute_path.unlink(missing_ok=True)

    if prune:
        prune_empty_dirs(absolute_path.parent, stop_at=stop_at)


def delete_storage_tree(
    relative_path: str | Path | None,
    *,
    prune: bool = True,
    stop_at: str | Path | None = None,
) -> None:
    if not relative_path:
        return

    absolute_path = resolve_storage_path(relative_path)
    shutil.rmtree(absolute_path, ignore_errors=True)

    if prune:
        prune_empty_dirs(absolute_path.parent, stop_at=stop_at)


def _guess_suffix(
    filename: str | None,
    content_type: str | None,
) -> str:
    if filename:
        suffix = Path(filename).suffix.lower()
        if suffix:
            return suffix

    if content_type:
        guessed = mimetypes.guess_extension(content_type)
        if guessed:
            return guessed

    return ".jpg"


def _validate_upload(
    content_type: str | None,
    size: int,
) -> None:
    if content_type and content_type not in uploads.allowed_types:
        raise ValidationError(
            "Неподдерживаемый тип файла",
            details={"content_type": content_type},
        )

    if size > uploads.max_size_bytes:
        raise ValidationError(
            "Файл превышает допустимый размер",
            details={
                "size": size,
                "max_size": uploads.max_size_bytes,
            },
        )


async def save_upload(
    upload: UploadFile,
    directory: str,
    filename_stem: str,
) -> str:
    data = await upload.read()
    _validate_upload(upload.content_type, len(data))

    suffix = _guess_suffix(upload.filename, upload.content_type)
    relative_path = Path(directory) / f"{filename_stem}{suffix}"
    absolute_path = resolve_storage_path(relative_path)

    ensure_parent_dir(absolute_path)
    absolute_path.write_bytes(data)

    return relative_path.as_posix()


async def delete_file(
    relative_path: str | None,
) -> None:
    delete_storage_file(relative_path)
