from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.0.1"}


def test_ingest_queues_a_job_that_documents_lists(client: TestClient) -> None:
    response = client.post(
        "/ingest", files={"file": ("inv.jpg", b"not-really-a-jpeg", "image/jpeg")}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"

    documents = client.get("/documents").json()["documents"]
    assert [d["id"] for d in documents] == [body["job_id"]]
    assert documents[0]["filename"] == "inv.jpg"
    assert documents[0]["invoice"] is None
    assert documents[0]["issues"] == []


def test_ingest_rejects_unknown_mime(client: TestClient) -> None:
    response = client.post("/ingest", files={"file": ("x.gif", b"GIF89a", "image/gif")})
    assert response.status_code == 415


def test_issues_is_empty_before_processing(client: TestClient) -> None:
    client.post("/ingest", files={"file": ("inv.pdf", b"%PDF-1.4", "application/pdf")})
    assert client.get("/issues").json() == {"issues": []}


def test_ask_answers_not_in_documents(client: TestClient) -> None:
    response = client.post("/ask", json={"question": "What is the total of invoice INV-1042?"})
    assert response.status_code == 200
    assert response.json() == {"answer": "Not in the documents.", "citations": [], "cost_usd": "0"}


def test_ask_requires_a_question(client: TestClient) -> None:
    assert client.post("/ask", json={"question": ""}).status_code == 422


def test_metrics_shape_matches_readme_table(client: TestClient) -> None:
    body = client.get("/metrics").json()
    assert set(body) == {
        "run_id",
        "field_accuracy",
        "retrieval_recall_at_5",
        "faithfulness",
        "cost_per_doc_usd",
        "p95_latency_ms",
    }
    assert all(value is None for value in body.values())


def test_each_test_gets_a_fresh_store(client: TestClient) -> None:
    assert client.get("/documents").json() == {"documents": []}
