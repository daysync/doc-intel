from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from doc_intel.api.app import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    from tests.api.fakes import FakeIndexer, FakeProcessor

    app = create_app(FakeProcessor(), indexer=FakeIndexer())
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="session")
def en_receipt() -> tuple[object, object]:
    """Re-exported for tests outside tests/extract; see tests/extract/conftest.py."""
    import random

    from doc_intel.dataset.content import generate_content
    from doc_intel.dataset.layouts import render_pdf
    from doc_intel.dataset.render import pdf_to_image

    rng = random.Random(21)
    content = generate_content(rng, "en")
    return content, pdf_to_image(render_pdf(content, "receipt", rng), dpi=110)


# --- live Postgres (docker compose locally, a service in CI); tests skip when unreachable

import os  # noqa: E402

import psycopg  # noqa: E402

from doc_intel.db.connection import apply_schema, make_pool  # noqa: E402

DATABASE_URL = os.environ.get(
    "DATABASE_URL", "postgresql://doc_intel:doc_intel@localhost:5433/doc_intel"
)


def _postgres_reachable() -> bool:
    try:
        with psycopg.connect(
            DATABASE_URL.replace("postgresql+psycopg://", "postgresql://"), connect_timeout=2
        ):
            return True
    except psycopg.OperationalError:
        return False


requires_postgres = pytest.mark.skipif(
    not _postgres_reachable(), reason="Postgres not reachable; docker compose up -d postgres"
)


@pytest.fixture
async def pool():  # type: ignore[no-untyped-def]  # AsyncConnectionPool, opened, schema applied, tables emptied
    pool = make_pool(DATABASE_URL, min_size=1, max_size=2)
    await pool.open()
    await apply_schema(pool)
    async with pool.connection() as connection:
        await connection.execute("DELETE FROM documents")
        await connection.commit()
    yield pool
    await pool.close()
