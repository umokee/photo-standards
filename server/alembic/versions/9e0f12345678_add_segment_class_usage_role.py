"""add segment class usage role

Revision ID: 9e0f12345678
Revises: 9a0b1c2d3e4f
Create Date: 2026-06-03 15:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "9e0f12345678"
down_revision: str | Sequence[str] | None = "9a0b1c2d3e4f"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        ALTER TABLE segment_classes
        ADD COLUMN IF NOT EXISTS usage_role VARCHAR(20) NOT NULL DEFAULT 'checked'
        """
    )
    op.execute(
        """
        ALTER TABLE segment_classes
        DROP CONSTRAINT IF EXISTS ck_segment_classes_usage_role
        """
    )
    op.execute(
        """
        ALTER TABLE segment_classes
        ADD CONSTRAINT ck_segment_classes_usage_role
        CHECK (usage_role IN ('checked', 'anchor', 'ignored'))
        """
    )


def downgrade() -> None:
    op.execute(
        """
        ALTER TABLE segment_classes
        DROP CONSTRAINT IF EXISTS ck_segment_classes_usage_role
        """
    )
    op.execute(
        """
        ALTER TABLE segment_classes
        DROP COLUMN IF EXISTS usage_role
        """
    )
