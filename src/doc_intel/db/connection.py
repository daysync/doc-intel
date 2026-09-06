"""One async connection pool per process, pgvector codecs registered on every connection."""

from pathlib import Path

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

SCHEMA = Path(__file__).with_name("schema.sql")


async def _configure(connection: AsyncConnection) -> None:
    """Runs for every new pool connection.

    The vector codec can only be registered once the extension exists, and on a fresh
    database nothing has created it yet, so this creates it first. Without that, every
    connection setup fails and the pool silently waits out its timeouts (found by CI, which
    always starts from an empty database).
    """
    await connection.execute("CREATE EXTENSION IF NOT EXISTS vector")
    await connection.commit()
    await register_vector_async(connection)


def make_pool(database_url: str, min_size: int = 1, max_size: int = 8) -> AsyncConnectionPool:
    """psycopg wants ``postgresql://``; the SQLAlchemy-style ``+psycopg`` suffix is tolerated."""
    url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return AsyncConnectionPool(
        url, min_size=min_size, max_size=max_size, open=False, configure=_configure, timeout=15
    )


async def apply_schema(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as connection:
        await connection.execute(SCHEMA.read_text())
        await connection.commit()
