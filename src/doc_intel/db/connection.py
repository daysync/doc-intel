"""One async connection pool per process, pgvector codecs registered on every connection."""

from pathlib import Path

from pgvector.psycopg import register_vector_async
from psycopg import AsyncConnection
from psycopg_pool import AsyncConnectionPool

SCHEMA = Path(__file__).with_name("schema.sql")


async def _configure(connection: AsyncConnection) -> None:
    await register_vector_async(connection)


def make_pool(database_url: str, min_size: int = 1, max_size: int = 8) -> AsyncConnectionPool:
    """psycopg wants ``postgresql://``; the SQLAlchemy-style ``+psycopg`` suffix is tolerated."""
    url = database_url.replace("postgresql+psycopg://", "postgresql://", 1)
    return AsyncConnectionPool(
        url, min_size=min_size, max_size=max_size, open=False, configure=_configure
    )


async def apply_schema(pool: AsyncConnectionPool) -> None:
    async with pool.connection() as connection:
        await connection.execute(SCHEMA.read_text())
        await connection.commit()
        # the vector type may not have existed when the pool configured this connection
        await register_vector_async(connection)
