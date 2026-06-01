"""4

Revision ID: f59e7a83a5c9
Revises: 97bdf53c24ca
Create Date: 2026-04-24 08:28:23.791836

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'f59e7a83a5c9'
down_revision: Union[str, Sequence[str], None] = '97bdf53c24ca'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.execute("ALTER TYPE camera_status_enum ADD VALUE IF NOT EXISTS 'unknown'")
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
