"""add alignment anchors

Revision ID: 9a0b1c2d3e4f
Revises: 6b7c8d9e0f13
Create Date: 2026-06-03 14:40:00.000000

"""

from collections.abc import Sequence

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision: str = "9a0b1c2d3e4f"
down_revision: str | Sequence[str] | None = "6b7c8d9e0f13"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "alignment_anchors",
        sa.Column("id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("image_id", postgresql.UUID(as_uuid=True), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("kind", sa.String(length=20), nullable=False, server_default="area"),
        sa.Column("points", sa.JSON(), nullable=False, server_default="[]"),
        sa.Column("search_radius", sa.Integer(), nullable=False, server_default="80"),
        sa.Column("created_at", sa.DateTime(), server_default=sa.func.now(), nullable=False),
        sa.ForeignKeyConstraint(["image_id"], ["standard_images.id"], ondelete="CASCADE"),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(op.f("ix_alignment_anchors_id"), "alignment_anchors", ["id"], unique=False)
    op.create_index(
        op.f("ix_alignment_anchors_image_id"),
        "alignment_anchors",
        ["image_id"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(op.f("ix_alignment_anchors_image_id"), table_name="alignment_anchors")
    op.drop_index(op.f("ix_alignment_anchors_id"), table_name="alignment_anchors")
    op.drop_table("alignment_anchors")
