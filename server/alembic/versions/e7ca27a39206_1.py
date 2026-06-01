"""1

Revision ID: e7ca27a39206
Revises: de3b4cc13c02
Create Date: 2026-04-22 23:36:44.769175

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "e7ca27a39206"
down_revision: str | Sequence[str] | None = "de3b4cc13c02"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    """Upgrade schema."""
    op.create_table(
        "inspection_items",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("inspection_id", sa.Uuid(), nullable=False),
        sa.Column("segment_class_id", sa.Uuid(), nullable=True),
        sa.Column("segment_class_group_id", sa.Uuid(), nullable=True),
        sa.Column("class_key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("hue", sa.Integer(), nullable=False),
        sa.Column("status", sa.String(length=20), nullable=True),
        sa.Column("is_found", sa.Boolean(), nullable=False),
        sa.Column("expected_count", sa.Integer(), nullable=True),
        sa.Column("detected_count", sa.Integer(), nullable=True),
        sa.Column("delta", sa.Integer(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column("detections", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
        sa.CheckConstraint(
            "hue BETWEEN 0 AND 359",
            name=op.f("ck_inspection_items_inspection_item_hue_range"),
        ),
        sa.ForeignKeyConstraint(
            ["inspection_id"],
            ["inspection_results.id"],
            name=op.f("fk_inspection_items_inspection_id_inspection_results"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["segment_class_group_id"],
            ["segment_class_groups.id"],
            name=op.f(
                "fk_inspection_items_segment_class_group_id_segment_class_groups"
            ),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["segment_class_id"],
            ["segment_classes.id"],
            name=op.f("fk_inspection_items_segment_class_id_segment_classes"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inspection_items")),
    )
    op.create_index(
        op.f("ix_inspection_items_class_key"),
        "inspection_items",
        ["class_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_items_id"), "inspection_items", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_inspection_items_inspection_id"),
        "inspection_items",
        ["inspection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_items_segment_class_group_id"),
        "inspection_items",
        ["segment_class_group_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_items_segment_class_id"),
        "inspection_items",
        ["segment_class_id"],
        unique=False,
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_class_key"),
        table_name="inspection_segment_results",
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_id"),
        table_name="inspection_segment_results",
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_inspection_id"),
        table_name="inspection_segment_results",
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_segment_class_group_id"),
        table_name="inspection_segment_results",
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_segment_class_id"),
        table_name="inspection_segment_results",
    )
    op.drop_table("inspection_segment_results")


def downgrade() -> None:
    """Downgrade schema."""
    op.create_table(
        "inspection_segment_results",
        sa.Column("id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column("inspection_id", sa.UUID(), autoincrement=False, nullable=False),
        sa.Column("segment_class_id", sa.UUID(), autoincrement=False, nullable=True),
        sa.Column(
            "segment_class_group_id", sa.UUID(), autoincrement=False, nullable=True
        ),
        sa.Column(
            "class_key", sa.VARCHAR(length=255), autoincrement=False, nullable=False
        ),
        sa.Column("name", sa.VARCHAR(length=255), autoincrement=False, nullable=False),
        sa.Column("is_found", sa.BOOLEAN(), autoincrement=False, nullable=False),
        sa.Column(
            "confidence",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=True,
        ),
        sa.Column("expected_count", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("detected_count", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("delta", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("status", sa.VARCHAR(length=20), autoincrement=False, nullable=True),
        sa.Column("hue", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column(
            "detections",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=True,
        ),
        sa.ForeignKeyConstraint(
            ["inspection_id"],
            ["inspection_results.id"],
            name=op.f("fk_inspection_segment_results_inspection_id_inspection_results"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["segment_class_group_id"],
            ["segment_class_groups.id"],
            name=op.f("fk_inspection_segment_results_segment_class_group_id_se_d64d"),
            ondelete="SET NULL",
        ),
        sa.ForeignKeyConstraint(
            ["segment_class_id"],
            ["segment_classes.id"],
            name=op.f("fk_inspection_segment_results_segment_class_id_segment_classes"),
            ondelete="SET NULL",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_inspection_segment_results")),
    )
    op.create_index(
        op.f("ix_inspection_segment_results_segment_class_id"),
        "inspection_segment_results",
        ["segment_class_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_segment_results_segment_class_group_id"),
        "inspection_segment_results",
        ["segment_class_group_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_segment_results_inspection_id"),
        "inspection_segment_results",
        ["inspection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_segment_results_id"),
        "inspection_segment_results",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_segment_results_class_key"),
        "inspection_segment_results",
        ["class_key"],
        unique=False,
    )
    op.drop_index(
        op.f("ix_inspection_items_segment_class_id"), table_name="inspection_items"
    )
    op.drop_index(
        op.f("ix_inspection_items_segment_class_group_id"),
        table_name="inspection_items",
    )
    op.drop_index(
        op.f("ix_inspection_items_inspection_id"), table_name="inspection_items"
    )
    op.drop_index(op.f("ix_inspection_items_id"), table_name="inspection_items")
    op.drop_index(op.f("ix_inspection_items_class_key"), table_name="inspection_items")
    op.drop_table("inspection_items")
