"""6

Revision ID: 4e62b3abc3cd
Revises: 8dfecd092332
Create Date: 2026-04-25 04:38:18.978707

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '4e62b3abc3cd'
down_revision: Union[str, Sequence[str], None] = '8dfecd092332'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('standard_images', sa.Column('features_path', sa.String(length=500), nullable=True))
    op.drop_column('standard_images', 'featires_path')


def downgrade() -> None:
    """Downgrade schema."""
    op.add_column('standard_images', sa.Column('featires_path', sa.VARCHAR(length=500), autoincrement=False, nullable=True))
    op.drop_column('standard_images', 'features_path')
