from pathlib import Path

import pytest
from psycopg_pool import AsyncConnectionPool

from doc_intel.dataset.generate import generate_dataset
from doc_intel.eval.retrieval import Question, QuestionResult, RetrievalReport, questions_for, run
from tests.conftest import requires_postgres
from tests.models.factories import invoice
from tests.rag.fakes import HashEmbedder


def test_questions_target_known_chunks() -> None:
    questions = questions_for("d1", invoice())
    assert [q.kind for q in questions] == ["totals", "header", "header", "line_item"]
    assert "INV-1042" in questions[0].text and "Shampoo 1L" in questions[3].text


def test_report_metrics() -> None:
    q = Question(text="?", document_id="d", kind="totals", language="en")
    report = RetrievalReport(
        config="c",
        embeddings_model="m",
        k=5,
        results=[
            QuestionResult(question=q, rank=1, top_document="d"),
            QuestionResult(question=q, rank=3, top_document="x"),
            QuestionResult(question=q, rank=None, top_document="x"),
            QuestionResult(question=q, rank=7, top_document="x"),
        ],
    )
    assert report.recall_at_k() == 0.5
    assert abs(report.mrr() - (1 + 1 / 3 + 0 + 1 / 7) / 4) < 1e-9
    assert report.by_kind() == {"totals": 0.5}


@requires_postgres
async def test_end_to_end_on_a_generated_dataset(
    tmp_path: Path, pool: AsyncConnectionPool, monkeypatch: pytest.MonkeyPatch
) -> None:
    from doc_intel.api.settings import get_settings
    from tests.conftest import TEST_DATABASE_URL

    monkeypatch.setenv("DATABASE_URL", TEST_DATABASE_URL)
    get_settings.cache_clear()
    generate_dataset(tmp_path, n=8, seed=5)
    report = await run(tmp_path, Path("configs/default.yaml"), k=5, embedder=HashEmbedder())
    assert len(report.results) == 32
    assert report.recall_at_k() > 0.5  # bag-of-words vectors plus full-text on invoice numbers
