"""remove alignment anchor experiment

Revision ID: c3d4e5f6a7b8
Revises: b7a4f23d9c10
Create Date: 2026-06-03 16:05:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "c3d4e5f6a7b8"
down_revision: str | Sequence[str] | None = "b7a4f23d9c10"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute("DROP TABLE IF EXISTS alignment_anchors")
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


def downgrade() -> None:
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
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS alignment_anchors (
            id UUID PRIMARY KEY,
            image_id UUID NOT NULL,
            name VARCHAR(255) NOT NULL,
            anchor_type VARCHAR(20) NOT NULL,
            points JSONB NOT NULL,
            created_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL,
            updated_at TIMESTAMP WITH TIME ZONE DEFAULT now() NOT NULL
        )
        """
    )
