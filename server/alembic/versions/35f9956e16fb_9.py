"""9

Revision ID: 35f9956e16fb
Revises: c667f23358b2
Create Date: 2026-04-30 04:10:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "35f9956e16fb"
down_revision: str | Sequence[str] | None = "c667f23358b2"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.alter_column(
        "cameras",
        "host",
        existing_type=sa.String(length=255),
        nullable=True,
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.alter_column(
        "cameras",
        "host",
        existing_type=sa.String(length=255),
        nullable=False,
    )
