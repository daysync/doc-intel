from fastapi.testclient import TestClient


def test_health(client: TestClient) -> None:
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "version": "0.0.1"}


def test_ingest_queues_then_processes_in_the_background(client: TestClient) -> None:
    response = client.post(
        "/ingest", files={"file": ("inv.jpg", b"not-really-a-jpeg", "image/jpeg")}
    )
    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "queued"

    # TestClient runs background tasks before returning, so the job is already done here
    document = client.get(f"/documents/{body['job_id']}").json()
    assert document["status"] == "done"
    assert document["filename"] == "inv.jpg"
    assert document["invoice"]["number"]["value"] == "INV-1042"
    assert document["cost_usd"] == "0.0042"
    assert document["confidence"]["number"] == 0.9
    assert document["issues"] == []


def test_failed_processing_is_recorded_on_the_job(client: TestClient) -> None:
    job_id = client.post("/ingest", files={"file": ("bad.pdf", b"BOOM", "application/pdf")}).json()[
        "job_id"
    ]
    document = client.get(f"/documents/{job_id}").json()
    assert document["status"] == "failed"
    assert "unreadable document" in document["error"]
    assert document["invoice"] is None


def test_unknown_document_is_404(client: TestClient) -> None:
    assert client.get("/documents/nope").status_code == 404


def test_ingest_rejects_unknown_mime(client: TestClient) -> None:
    response = client.post("/ingest", files={"file": ("x.gif", b"GIF89a", "image/gif")})
    assert response.status_code == 415


def test_issues_include_cross_document_duplicates(client: TestClient) -> None:
    assert client.get("/issues").json() == {"issues": []}
    a = client.post("/ingest", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}).json()[
        "job_id"
    ]
    b = client.post("/ingest", files={"file": ("b.pdf", b"%PDF-1.4", "application/pdf")}).json()[
        "job_id"
    ]
    issues = client.get("/issues").json()["issues"]
    assert {(i["document_id"], i["issue"]["code"]) for i in issues} == {
        (a, "duplicate_number"),
        (b, "duplicate_number"),
    }
    assert issues[0]["issue"]["related_document_ids"] == [b]


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


def test_processed_documents_are_indexed(client: TestClient) -> None:
    job_id = client.post(
        "/ingest", files={"file": ("a.pdf", b"%PDF-1.4", "application/pdf")}
    ).json()["job_id"]
    client.post("/ingest", files={"file": ("bad.pdf", b"BOOM", "application/pdf")})
    assert client.app.state.indexer.indexed == [job_id]  # type: ignore[attr-defined]
