"""repair alignment anchors table

Revision ID: b7a4f23d9c10
Revises: 9e0f12345678
Create Date: 2026-06-03 15:25:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "b7a4f23d9c10"
down_revision: str | Sequence[str] | None = "9e0f12345678"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.execute(
        """
        CREATE TABLE IF NOT EXISTS alignment_anchors (
            id UUID NOT NULL,
            image_id UUID NOT NULL,
            name VARCHAR(255) NOT NULL,
            kind VARCHAR(20) NOT NULL DEFAULT 'area',
            points JSON NOT NULL DEFAULT '[]',
            search_radius INTEGER NOT NULL DEFAULT 80,
            created_at TIMESTAMP WITHOUT TIME ZONE NOT NULL DEFAULT now(),
            PRIMARY KEY (id),
            CONSTRAINT fk_alignment_anchors_image_id_standard_images
                FOREIGN KEY(image_id) REFERENCES standard_images (id) ON DELETE CASCADE
        )
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_alignment_anchors_id
        ON alignment_anchors (id)
        """
    )
    op.execute(
        """
        CREATE INDEX IF NOT EXISTS ix_alignment_anchors_image_id
        ON alignment_anchors (image_id)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_alignment_anchors_image_id")
    op.execute("DROP INDEX IF EXISTS ix_alignment_anchors_id")
    op.execute("DROP TABLE IF EXISTS alignment_anchors")
