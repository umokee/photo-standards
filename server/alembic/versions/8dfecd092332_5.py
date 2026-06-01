"""5

Revision ID: 8dfecd092332
Revises: f59e7a83a5c9
Create Date: 2026-04-25 04:36:11.283091

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = '8dfecd092332'
down_revision: Union[str, Sequence[str], None] = 'f59e7a83a5c9'
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column('standard_images', sa.Column('featires_path', sa.String(length=500), nullable=True))
    op.add_column('standard_images', sa.Column('features_keypoint_count', sa.Integer(), nullable=True))
    op.add_column('standard_images', sa.Column('features_computed_at', sa.DateTime(), nullable=True))


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column('standard_images', 'features_computed_at')
    op.drop_column('standard_images', 'features_keypoint_count')
    op.drop_column('standard_images', 'featires_path')
