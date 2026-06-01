"""1

Revision ID: de3b4cc13c02
Revises:
Create Date: 2026-04-22 23:26:35.831696

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "de3b4cc13c02"
down_revision: str | Sequence[str] | None = None
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column(
        "inspection_results",
        sa.Column(
            "debug_payload", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
    )
    op.add_column(
        "inspection_segment_results", sa.Column("hue", sa.Integer(), nullable=False)
    )
    op.add_column(
        "inspection_segment_results",
        sa.Column("detections", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("inspection_segment_results", "detections")
    op.drop_column("inspection_segment_results", "hue")
    op.drop_column("inspection_results", "debug_payload")
