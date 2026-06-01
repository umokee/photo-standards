"""drop inspection serial number

Revision ID: 2f1d9a1b7c6e
Revises: c5f69eb3c55c
Create Date: 2026-05-07 00:00:00.000000

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "2f1d9a1b7c6e"
down_revision = "a83b9f036c32"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.drop_index("ix_inspection_results_serial_number", table_name="inspection_results")
    op.drop_column("inspection_results", "serial_number")


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column(
        "inspection_results",
        sa.Column("serial_number", sa.String(length=100), nullable=True),
    )
    op.create_index(
        "ix_inspection_results_serial_number",
        "inspection_results",
        ["serial_number"],
    )
