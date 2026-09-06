import json

import pytest
from psycopg_pool import AsyncConnectionPool

from doc_intel.rag.index import Hit, Retriever, normalise_number
from doc_intel.rag.rerank import Reranker
from doc_intel.rag.scope import invoice_numbers_in, resolve_scope
from tests.conftest import requires_postgres
from tests.llm.fakes import FakeLLM
from tests.rag.fakes import HashEmbedder
from tests.rag.test_index import _stored


@pytest.mark.parametrize(
    ("question", "expected"),
    [
        ("What is the grand total of invoice INV-2026-0917?", ["INV-2026-0917"]),
        ("Who supplied F-2026-3165?", ["F-2026-3165"]),
        ("How much was invoice 4587?", ["4587"]),
        ("Сума рахунку SF-2026-4609?", ["SF-2026-4609"]),
        ("grand total of INV-7001", ["INV-7001"]),
        ("What did we buy last month?", []),
    ],
)
def test_invoice_numbers_are_recognised(question: str, expected: list[str]) -> None:
    assert invoice_numbers_in(question) == expected


def test_normalise_number_ignores_separators_and_case() -> None:
    assert normalise_number("inv 2026/0917") == normalise_number("INV-2026-0917") == "INV20260917"


@requires_postgres
async def test_scope_restricts_search_to_the_named_document(pool: AsyncConnectionPool) -> None:
    embedder = HashEmbedder()
    a = await _stored(pool, "INV-7001", "Beauty Supplies Ltd")
    b = await _stored(pool, "INV-7002", "Glow Wholesale")
    from doc_intel.rag.index import Indexer

    await Indexer(pool, embedder).index(a)
    await Indexer(pool, embedder).index(b)
    retriever = Retriever(pool, embedder)

    scope = await resolve_scope("Who is the supplier on invoice INV-7002?", retriever)
    assert scope == [b.document_id]
    hits = await retriever.search("Who is the supplier?", k=5, document_ids=scope)
    assert hits and all(h.document_id == b.document_id for h in hits)
    assert await resolve_scope("Who is the supplier on invoice INV-9999?", retriever) is None
    assert await resolve_scope("anything at all", retriever) is None


async def test_reranker_orders_by_model_relevance_then_fusion_score() -> None:
    hits = [
        Hit(1, "d", 1, "ocr", None, "a", 0.5),
        Hit(2, "d", 1, "totals", None, "b", 0.4),
        Hit(3, "d", 1, "header", None, "c", 0.3),
    ]
    llm = FakeLLM(
        json.dumps(
            {"scores": [{"chunk_id": 3, "relevance": 0.9}, {"chunk_id": 2, "relevance": 0.9}]}
        )
    )
    ordered = await Reranker(llm, "m").rerank("q", hits, k=2)
    assert [h.chunk_id for h in ordered] == [2, 3]  # tie on relevance broken by fusion score
    assert await Reranker(llm, "m").rerank("q", hits[:1], k=5) == hits[:1] and len(llm.calls) == 1
