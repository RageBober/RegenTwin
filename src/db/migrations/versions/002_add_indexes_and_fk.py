"""Add indexes on status/created_at and FK on analyses.simulation_id.

Revision ID: 002
Revises: 001
Create Date: 2026-03-02
"""
from typing import Sequence, Union

from alembic import op

revision: str = "002"
down_revision: Union[str, None] = "001"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Indexes for simulations
    op.create_index("ix_simulations_status", "simulations", ["status"])
    op.create_index("ix_simulations_created_at", "simulations", ["created_at"])

    # Indexes for uploads
    op.create_index("ix_uploads_status", "uploads", ["status"])
    op.create_index("ix_uploads_created_at", "uploads", ["created_at"])

    # Indexes for analyses
    op.create_index("ix_analyses_status", "analyses", ["status"])
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])
    op.create_index("ix_analyses_simulation_id", "analyses", ["simulation_id"])

    # SQLite не поддерживает ALTER TABLE ADD FOREIGN KEY напрямую,
    # но индекс на simulation_id обеспечит быстрые JOIN-ы.
    # FK constraint добавлен в ORM модели (src/db/models.py).


def downgrade() -> None:
    op.drop_index("ix_analyses_simulation_id", "analyses")
    op.drop_index("ix_analyses_created_at", "analyses")
    op.drop_index("ix_analyses_status", "analyses")
    op.drop_index("ix_uploads_created_at", "uploads")
    op.drop_index("ix_uploads_status", "uploads")
    op.drop_index("ix_simulations_created_at", "simulations")
    op.drop_index("ix_simulations_status", "simulations")
