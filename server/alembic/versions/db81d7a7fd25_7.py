"""7

Revision ID: db81d7a7fd25
Revises: 3c5cc8eab9ae
Create Date: 2026-04-25 23:58:05.354904

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'db81d7a7fd25'
down_revision: Union[str, Sequence[str], None] = '3c5cc8eab9ae'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('cameras', sa.Column('device_path', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('cameras', 'device_path')
