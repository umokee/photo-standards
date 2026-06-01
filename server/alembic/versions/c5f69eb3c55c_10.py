"""10

Revision ID: c5f69eb3c55c
Revises: 35f9956e16fb
Create Date: 2026-05-04 18:58:27.154415

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c5f69eb3c55c'
down_revision: Union[str, Sequence[str], None] = '35f9956e16fb'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    pass


def downgrade() -> None:
    """Downgrade schema."""
    pass
