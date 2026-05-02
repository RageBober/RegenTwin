"""Alembic environment configuration.

URL берётся из `src.api.config.settings.database_url`, что позволяет
переопределить его через переменную окружения `REGENTWIN_DATABASE_URL`.
"""

from logging.config import fileConfig

from alembic import context
from alembic.ddl.impl import DefaultImpl
from sqlalchemy import engine_from_config, pool

from src.api.config import settings
from src.db.models import Base


# Зарегистрировать Alembic DDL impl для DuckDB.
# duckdb-engine не предоставляет свой Alembic impl, а DuckDB поддерживает
# базовый DDL (CREATE TABLE / INDEX), поэтому DefaultImpl достаточен.
class DuckDBImpl(DefaultImpl):
    __dialect__ = "duckdb"


config = context.config
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# Inject URL from settings (overrides empty alembic.ini key).
config.set_main_option("sqlalchemy.url", settings.database_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode."""
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(connection=connection, target_metadata=target_metadata)
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
