from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class AppError(Exception):
    code: str
    message: str
    status_code: int
    details: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.message


class ValidationError(AppError):
    def __init__(
        self,
        message: str = "Проверьте введенные данные",
        details: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(
            code="VALIDATION_ERROR",
            message=message,
            status_code=422,
            details=details or {},
        )


class NotFoundError(AppError):
    def __init__(
        self,
        entity: str,
        entity_id: Any,
        verb: str = "не найден",
    ):
        super().__init__(
            code="NOT_FOUND",
            message=f"{entity} ({entity_id}) {verb}",
            status_code=404,
        )


class ConflictError(AppError):
    def __init__(
        self,
        message: str = "Возник конфликт данных",
        details: dict[str, Any] | None = None,
    ):
        super().__init__(
            code="CONFLICT",
            message=message,
            status_code=409,
            details=details or {},
        )


class InternalServerError(AppError):
    def __init__(
        self,
        message: str = "Внутренняя ошибка сервера",
    ):
        super().__init__(
            code="INTERNAL_ERROR",
            message=message,
            status_code=500,
        )
