"""3

Revision ID: 97bdf53c24ca
Revises: e7ca27a39206
Create Date: 2026-04-24 04:31:15.128658

"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "97bdf53c24ca"
down_revision: str | Sequence[str] | None = "e7ca27a39206"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

camera_protocol_enum = postgresql.ENUM(
    "rtsp",
    "usb",
    "http",
    name="camera_protocol_enum",
    create_type=False,
)

camera_status_enum = postgresql.ENUM(
    "online",
    "offline",
    "error",
    name="camera_status_enum",
    create_type=False,
)


def upgrade() -> None:
    """Upgrade schema."""
    op.add_column("cameras", sa.Column("description", sa.Text(), nullable=True))

    bind = op.get_bind()
    camera_protocol_enum.create(bind, checkfirst=True)
    op.add_column(
        "cameras",
        sa.Column(
            "protocol",
            camera_protocol_enum,
            nullable=True,
        ),
    )
    op.execute("UPDATE cameras SET protocol = 'rtsp' WHERE protocol IS NULL")
    op.alter_column("cameras", "protocol", nullable=False)

    op.add_column("cameras", sa.Column("host", sa.String(length=255), nullable=False))
    op.add_column("cameras", sa.Column("port", sa.Integer(), nullable=True))
    op.add_column("cameras", sa.Column("path", sa.String(length=255), nullable=True))
    op.add_column(
        "cameras", sa.Column("username", sa.String(length=255), nullable=True)
    )
    op.add_column(
        "cameras", sa.Column("password", sa.String(length=255), nullable=True)
    )
    op.add_column("cameras", sa.Column("timeout_sec", sa.Integer(), nullable=False))

    camera_status_enum.create(bind, checkfirst=True)
    op.add_column(
        "cameras",
        sa.Column("last_status", camera_status_enum, nullable=True),
    )

    op.add_column("cameras", sa.Column("last_checked_at", sa.DateTime(), nullable=True))
    op.add_column(
        "cameras", sa.Column("last_error", sa.String(length=500), nullable=True)
    )
    op.drop_column("cameras", "last_seen_at")
    op.drop_column("cameras", "resolution")
    op.drop_column("cameras", "rtsp_url")
    op.create_unique_constraint(
        "uq_inspection_segment_class",
        "inspection_items",
        ["inspection_id", "segment_class_id"],
    )


def downgrade() -> None:
    """Downgrade schema."""
    op.drop_constraint(
        "uq_inspection_segment_class", "inspection_items", type_="unique"
    )
    op.add_column(
        "cameras",
        sa.Column(
            "rtsp_url", sa.VARCHAR(length=500), autoincrement=False, nullable=False
        ),
    )
    op.add_column(
        "cameras",
        sa.Column(
            "resolution", sa.VARCHAR(length=20), autoincrement=False, nullable=True
        ),
    )
    op.add_column(
        "cameras",
        sa.Column(
            "last_seen_at", postgresql.TIMESTAMP(), autoincrement=False, nullable=True
        ),
    )
    op.drop_column("cameras", "last_error")
    op.drop_column("cameras", "last_checked_at")
    op.drop_column("cameras", "last_status")
    op.drop_column("cameras", "timeout_sec")
    op.drop_column("cameras", "password")
    op.drop_column("cameras", "username")
    op.drop_column("cameras", "path")
    op.drop_column("cameras", "port")
    op.drop_column("cameras", "host")
    op.drop_column("cameras", "protocol")
    op.drop_column("cameras", "description")
