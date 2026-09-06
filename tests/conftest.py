from collections.abc import Iterator

import pytest
from fastapi.testclient import TestClient

from doc_intel.api.app import create_app


@pytest.fixture
def client() -> Iterator[TestClient]:
    from tests.api.fakes import FakeIndexer, FakeProcessor, FakeQa

    app = create_app(FakeProcessor(), indexer=FakeIndexer(), qa=FakeQa())
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


# --- live Postgres (docker compose locally, a service in CI); tests skip when unreachable.
# Tests use their OWN database (doc_intel_test), created on demand: the pool fixture truncates
# tables, and sharing the dev database once wiped an eval's index mid-run.

import os  # noqa: E402

import psycopg  # noqa: E402

from doc_intel.db.connection import apply_schema, make_pool  # noqa: E402

_SERVER_URL = os.environ.get(
    "DATABASE_URL", "postgresql://doc_intel:doc_intel@localhost:5433/doc_intel"
)
_SERVER_URL = _SERVER_URL.replace("postgresql+psycopg://", "postgresql://", 1)
TEST_DB = os.environ.get("TEST_DATABASE_NAME", "doc_intel_test")
TEST_DATABASE_URL = _SERVER_URL.rsplit("/", 1)[0] + f"/{TEST_DB}"


def _ensure_test_database() -> bool:
    try:
        with psycopg.connect(_SERVER_URL, connect_timeout=2, autocommit=True) as connection:
            exists = connection.execute(
                "SELECT 1 FROM pg_database WHERE datname = %s", (TEST_DB,)
            ).fetchone()
            if not exists:
                connection.execute(f'CREATE DATABASE "{TEST_DB}"')
        return True
    except psycopg.OperationalError:
        return False


requires_postgres = pytest.mark.skipif(
    not _ensure_test_database(), reason="Postgres not reachable; docker compose up -d postgres"
)


@pytest.fixture
async def pool():  # type: ignore[no-untyped-def]  # AsyncConnectionPool on the test DB, schema applied, tables emptied
    pool = make_pool(TEST_DATABASE_URL, min_size=1, max_size=2)
    await pool.open()
    await apply_schema(pool)
    async with pool.connection() as connection:
        await connection.execute("DELETE FROM documents")
        await connection.commit()
    yield pool
    await pool.close()
