"""allow multiple reference images

Revision ID: d4e5f6a7b9c0
Revises: c3d4e5f6a7b8
Create Date: 2026-06-11 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "d4e5f6a7b9c0"
down_revision: str | Sequence[str] | None = "c3d4e5f6a7b8"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_standard_reference")


def downgrade() -> None:
    op.execute(
        """
        WITH ranked AS (
            SELECT
                id,
                ROW_NUMBER() OVER (
                    PARTITION BY standard_id
                    ORDER BY created_at ASC, id ASC
                ) AS rn
            FROM standard_images
            WHERE is_reference = true
        )
        UPDATE standard_images
        SET is_reference = false
        WHERE id IN (SELECT id FROM ranked WHERE rn > 1)
        """
    )
    op.execute(
        """
        CREATE UNIQUE INDEX IF NOT EXISTS uq_standard_reference
        ON standard_images (standard_id)
        WHERE is_reference = true
        """
    )
