from __future__ import annotations

from pathlib import Path

import structlog
from alembic import command
from alembic.config import Config
from app.config import settings
from app.db import Base
from app.import_models import import_models
from app.logging import configure_logging
from app.observability import log_event
from sqlalchemy import create_engine, inspect

logger = structlog.get_logger(__name__)


def build_alembic_config() -> Config:
    project_root = Path(__file__).resolve().parents[1]
    cfg = Config(str(project_root / "alembic.ini"))
    cfg.set_main_option("script_location", str(project_root / "alembic"))
    cfg.set_main_option("sqlalchemy.url", settings.database_url_sync.replace("%", "%%"))
    return cfg


def has_alembic_revisions() -> bool:
    versions_dir = Path(__file__).resolve().parents[1] / "alembic" / "versions"
    return any(versions_dir.glob("*.py"))


def bootstrap_database() -> None:
    import_models()
    engine = create_engine(
        settings.database_url_sync,
        pool_pre_ping=True,
        future=True,
    )
    alembic_cfg = build_alembic_config()
    revisions_exist = has_alembic_revisions()

    try:
        inspector = inspect(engine)
        existing_tables = set(inspector.get_table_names())
        model_tables = {table.name for table in Base.metadata.sorted_tables}

        has_any_model_tables = bool(existing_tables & model_tables)
        has_alembic_version = "alembic_version" in existing_tables

        if not revisions_exist:
            Base.metadata.create_all(bind=engine, checkfirst=True)
            logger.info(
                "db.bootstrap_completed",
                action="create_all",
                reason="missing_alembic_revisions",
            )
        elif not has_any_model_tables:
            Base.metadata.create_all(bind=engine, checkfirst=True)
            command.stamp(alembic_cfg, "head")
            logger.info(
                "db.bootstrap_completed",
                action="create_all_and_stamp_head",
                reason="empty_model_schema",
            )
        elif has_any_model_tables and not has_alembic_version:
            command.stamp(alembic_cfg, "head")
            logger.info(
                "db.bootstrap_completed",
                action="stamp_head",
                reason="existing_schema_without_alembic_version",
            )
        else:
            command.upgrade(alembic_cfg, "head")
            logger.info(
                "db.bootstrap_completed",
                action="upgrade_head",
                reason="schema_already_initialized",
            )

    finally:
        engine.dispose()


if __name__ == "__main__":
    configure_logging(debug=settings.DEBUG)
    try:
        bootstrap_database()
    except Exception as exc:
        log_event(
            logger,
            "error",
            "db.bootstrap_failed",
            error_type=type(exc).__name__,
            exception=exc,
        )
        raise
