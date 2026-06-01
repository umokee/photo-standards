"""6

Revision ID: 3c5cc8eab9ae
Revises: 4e62b3abc3cd
Create Date: 2026-04-25 08:22:20.490133

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "3c5cc8eab9ae"
down_revision: str | Sequence[str] | None = "4e62b3abc3cd"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

alignment_status_enum = postgresql.ENUM(
    "success",
    "insufficient_matches",
    "insufficient_inliers",
    "homography_failed",
    name="alignment_status_enum",
    create_type=False,
)

segment_result_status_enum = postgresql.ENUM(
    "ok",
    "missing",
    "extra",
    name="segment_result_status_enum",
    create_type=False,
)


def upgrade() -> None:
    """Upgrade schema."""
    bind = op.get_bind()
    alignment_status_enum.create(bind, checkfirst=True)
    segment_result_status_enum.create(bind, checkfirst=True)
    op.create_table(
        "inspection_segment_results",
        sa.Column("id", sa.Uuid(), nullable=False),
        sa.Column("inspection_id", sa.Uuid(), nullable=False),
        sa.Column("segment_annotation_id", sa.Uuid(), nullable=True),
        sa.Column("segment_class_id", sa.Uuid(), nullable=True),
        sa.Column("class_key", sa.String(length=255), nullable=False),
        sa.Column("name", sa.String(length=255), nullable=False),
        sa.Column("hue", sa.Integer(), nullable=False),
        sa.Column(
            "status",
            segment_result_status_enum,
            nullable=False,
        ),
        sa.Column("iou", sa.Float(), nullable=True),
        sa.Column("confidence", sa.Float(), nullable=True),
        sa.Column(
            "expected_polygon", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "detected_polygon", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.Column(
            "detected_bbox", postgresql.JSONB(astext_type=sa.Text()), nullable=True
        ),
        sa.CheckConstraint(
            "hue IS NULL OR hue BETWEEN 0 AND 359",
            name=op.f("ck_inspection_segment_results_hue_range"),
        ),
        sa.CheckConstraint(
            "iou IS NULL OR (iou >= 0 AND iou <= 1)",
            name=op.f("ck_inspection_segment_results_iou_range"),
        ),
        sa.ForeignKeyConstraint(
            ["inspection_id"],
            ["inspection_results.id"],
            name=op.f("fk_inspection_segment_results_inspection_id_inspection_results"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["segment_annotation_id"],
            ["segment_annotations.id"],
            name=op.f(
                "fk_inspection_segment_results_segment_annotation_id_segment_annotations"
            ),
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
        op.f("ix_inspection_segment_results_class_key"),
        "inspection_segment_results",
        ["class_key"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_segment_results_id"),
        "inspection_segment_results",
        ["id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_segment_results_inspection_id"),
        "inspection_segment_results",
        ["inspection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_segment_results_segment_annotation_id"),
        "inspection_segment_results",
        ["segment_annotation_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_segment_results_segment_class_id"),
        "inspection_segment_results",
        ["segment_class_id"],
        unique=False,
    )
    op.drop_index(op.f("ix_inspection_items_class_key"), table_name="inspection_items")
    op.drop_index(op.f("ix_inspection_items_id"), table_name="inspection_items")
    op.drop_index(
        op.f("ix_inspection_items_inspection_id"), table_name="inspection_items"
    )
    op.drop_index(
        op.f("ix_inspection_items_segment_class_group_id"),
        table_name="inspection_items",
    )
    op.drop_index(
        op.f("ix_inspection_items_segment_class_id"), table_name="inspection_items"
    )
    op.drop_table("inspection_items")
    op.add_column(
        "inspection_results",
        sa.Column(
            "alignment_status",
            alignment_status_enum,
            nullable=True,
        ),
    )
    op.add_column(
        "inspection_results",
        sa.Column("alignment_inlier_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "inspection_results",
        sa.Column("alignment_raw_match_count", sa.Integer(), nullable=True),
    )
    op.add_column(
        "inspection_results",
        sa.Column("homography", postgresql.JSONB(astext_type=sa.Text()), nullable=True),
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_column("inspection_results", "homography")
    op.drop_column("inspection_results", "alignment_raw_match_count")
    op.drop_column("inspection_results", "alignment_inlier_count")
    op.drop_column("inspection_results", "alignment_status")
    op.create_table(
        "inspection_items",
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
        sa.Column("hue", sa.INTEGER(), autoincrement=False, nullable=False),
        sa.Column("status", sa.VARCHAR(length=20), autoincrement=False, nullable=True),
        sa.Column("is_found", sa.BOOLEAN(), autoincrement=False, nullable=False),
        sa.Column("expected_count", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("detected_count", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column("delta", sa.INTEGER(), autoincrement=False, nullable=True),
        sa.Column(
            "confidence",
            sa.DOUBLE_PRECISION(precision=53),
            autoincrement=False,
            nullable=True,
        ),
        sa.Column(
            "detections",
            postgresql.JSONB(astext_type=sa.Text()),
            autoincrement=False,
            nullable=True,
        ),
        sa.CheckConstraint(
            "hue >= 0 AND hue <= 359",
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
        sa.UniqueConstraint(
            "inspection_id",
            "segment_class_id",
            name=op.f("uq_inspection_segment_class"),
            postgresql_include=[],
            postgresql_nulls_not_distinct=False,
        ),
    )
    op.create_index(
        op.f("ix_inspection_items_segment_class_id"),
        "inspection_items",
        ["segment_class_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_items_segment_class_group_id"),
        "inspection_items",
        ["segment_class_group_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_items_inspection_id"),
        "inspection_items",
        ["inspection_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_inspection_items_id"), "inspection_items", ["id"], unique=False
    )
    op.create_index(
        op.f("ix_inspection_items_class_key"),
        "inspection_items",
        ["class_key"],
        unique=False,
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_segment_class_id"),
        table_name="inspection_segment_results",
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_segment_annotation_id"),
        table_name="inspection_segment_results",
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_inspection_id"),
        table_name="inspection_segment_results",
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_id"),
        table_name="inspection_segment_results",
    )
    op.drop_index(
        op.f("ix_inspection_segment_results_class_key"),
        table_name="inspection_segment_results",
    )
    op.drop_table("inspection_segment_results")
