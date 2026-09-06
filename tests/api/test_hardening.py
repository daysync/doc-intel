import logging

from fastapi.testclient import TestClient

from doc_intel.api.app import create_app
from doc_intel.api.middleware import JsonFormatter, request_id_var
from doc_intel.api.settings import Settings
from tests.api.fakes import FakeIndexer, FakeProcessor, FakeQa


def _client(**overrides: object) -> TestClient:
    settings = Settings(_env_file=None, **overrides)  # type: ignore[arg-type]
    return TestClient(
        create_app(FakeProcessor(), indexer=FakeIndexer(), qa=FakeQa(), settings=settings)
    )


def test_open_api_when_no_keys_configured() -> None:
    client = _client()
    assert client.get("/documents").status_code == 200
    assert client.get("/ready").json()["auth"] == "open"


def test_api_key_required_when_configured() -> None:
    client = _client(api_keys="secret-1, secret-2")
    assert client.get("/documents").status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "wrong"}).status_code == 401
    assert client.get("/documents", headers={"X-API-Key": "secret-2"}).status_code == 200
    assert client.post("/ask", json={"question": "x"}).status_code == 401
    # probes stay open
    assert client.get("/health").status_code == 200
    body = client.get("/ready").json()
    assert body["auth"] == "api-key" and body["checks"]["postgres"] == "in-memory store"


def test_upload_size_limit() -> None:
    client = _client(max_upload_mb=1)
    too_big = b"x" * (1024 * 1024 + 1)
    assert (
        client.post("/ingest", files={"file": ("big.pdf", too_big, "application/pdf")}).status_code
        == 413
    )
    assert (
        client.post("/ingest", files={"file": ("ok.pdf", b"%PDF", "application/pdf")}).status_code
        == 202
    )


def test_request_id_is_echoed_and_honoured() -> None:
    client = _client()
    fresh = client.get("/health")
    assert len(fresh.headers["X-Request-Id"]) == 32
    given = client.get("/health", headers={"X-Request-Id": "trace-abc"})
    assert given.headers["X-Request-Id"] == "trace-abc"


def test_json_formatter_includes_request_id() -> None:
    record = logging.LogRecord("doc_intel.api", logging.INFO, "f", 1, "hello %s", ("world",), None)
    token = request_id_var.set("req-1")
    try:
        line = JsonFormatter().format(record)
    finally:
        request_id_var.reset(token)
    assert '"message": "hello world"' in line and '"request_id": "req-1"' in line
