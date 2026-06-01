"""fix old yolo architecture values

Revision ID: a83b9f036c32
Revises: c5f69eb3c55c
Create Date: 2026-05-04 19:05:01.982587

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'a83b9f036c32'
down_revision: Union[str, Sequence[str], None] = 'c5f69eb3c55c'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    bind = op.get_bind()

    with op.get_context().autocommit_block():
        for value in (
            "yolo26n-seg",
            "yolo26s-seg",
            "yolo26m-seg",
            "yolo26l-seg",
            "yolo26x-seg",
        ):
            bind.exec_driver_sql(
                f"ALTER TYPE architecture_enum ADD VALUE IF NOT EXISTS '{value}'"
            )

    op.execute(
        """
        UPDATE ml_models
        SET architecture = REPLACE(architecture::text, 'yolov26', 'yolo26')::architecture_enum
        WHERE architecture::text LIKE 'yolov26%'
        """
    )


def downgrade() -> None:
    """Downgrade schema."""
    pass
