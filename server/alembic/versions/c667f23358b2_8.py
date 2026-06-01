"""8

Revision ID: c667f23358b2
Revises: db81d7a7fd25
Create Date: 2026-04-26 08:42:14.061260

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = 'c667f23358b2'
down_revision: Union[str, Sequence[str], None] = 'db81d7a7fd25'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('cameras', sa.Column('stream_path', sa.String(length=255), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('cameras', 'stream_path')
