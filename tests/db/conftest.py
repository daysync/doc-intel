"""A live Postgres from docker-compose (or CI's service). Tests skip when it is unreachable."""

import os
from collections.abc import AsyncIterator

import psycopg
import pytest

from doc_intel.db.connection import apply_schema, make_pool

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://doc_intel:doc_intel@localhost:5433/doc_intel"
)


def _reachable() -> bool:
    try:
        with psycopg.connect(
            DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"), connect_timeout=2
        ):
            return True
    except psycopg.OperationalError:
        return False


requires_postgres = pytest.mark.skipif(
    not _reachable(), reason="Postgres not reachable; docker compose up -d postgres"
)


@pytest.fixture
async def pool() -> AsyncIterator[object]:
    pool = make_pool(DATABASE_URL, min_size=1, max_size=2)
    await pool.open()
    await apply_schema(pool)
    async with pool.connection() as connection:
        await connection.execute("DELETE FROM documents")
        await connection.commit()
    yield pool
    await pool.close()
