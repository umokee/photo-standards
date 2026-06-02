"""add unmatched segment status

Revision ID: 6b7c8d9e0f13
Revises: 2f1d9a1b7c6e
Create Date: 2026-06-02 00:00:00.000000

"""

from collections.abc import Sequence

from alembic import op

revision: str = "6b7c8d9e0f13"
down_revision: str | Sequence[str] | None = "2f1d9a1b7c6e"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()

    with op.get_context().autocommit_block():
        bind.exec_driver_sql(
            "ALTER TYPE segment_result_status_enum "
            "ADD VALUE IF NOT EXISTS 'unmatched'"
        )


def downgrade() -> None:
    """Downgrade schema."""
    pass
