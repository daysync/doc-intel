import json

import pytest

from doc_intel.rag.answer import (
    NOT_IN_DOCUMENTS,
    Answer,
    Answerer,
    Citation,
    render_excerpts,
    verify_citations,
)
from doc_intel.rag.index import Hit
from tests.llm.fakes import FakeLLM

HITS = [
    Hit(
        1,
        "doc-a",
        1,
        "totals",
        "totals",
        "Invoice INV-7001 totals: subtotal 65.50; tax VAT 20% = 13.10; grand total 78.60 EUR",
        0.9,
    ),
    Hit(
        2,
        "doc-a",
        1,
        "header",
        "number",
        "Invoice INV-7001. issued 30 Aug 2026. supplier Beauty Supplies Ltd",
        0.8,
    ),
]


def _reply(
    answer: str,
    citations: list[tuple[int, str]],
    not_in_documents: bool = False,
    confidence: float = 0.9,
) -> str:
    return json.dumps(
        {
            "answer": answer,
            "citations": [{"chunk_id": i, "quote": q} for i, q in citations],
            "not_in_documents": not_in_documents,
            "confidence": confidence,
        }
    )


def test_excerpts_are_numbered_by_chunk_id() -> None:
    text = render_excerpts(HITS)
    assert text.startswith("[1] document doc-a, page 1, totals:")
    assert "[2] document doc-a, page 1, header:" in text


def test_verify_keeps_only_quotes_found_in_their_chunk() -> None:
    answer = Answer(
        answer="x",
        citations=[
            Citation(chunk_id=1, quote="grand total 78.60 EUR"),
            Citation(chunk_id=1, quote="GRAND  total 78.60"),  # case and spacing tolerated
            Citation(chunk_id=2, quote="grand total 78.60 EUR"),  # right text, wrong chunk
            Citation(chunk_id=9, quote="anything"),  # unknown chunk
            Citation(chunk_id=1, quote="totally invented"),
        ],
        not_in_documents=False,
        confidence=0.9,
    )
    kept, dropped = verify_citations(answer, HITS)
    assert [c.chunk_id for c in kept] == [1, 1] and dropped == 3
    assert kept[0].document_id == "doc-a" and kept[0].snippet == "grand total 78.60 EUR"


async def test_supported_answer_carries_verified_citations() -> None:
    llm = FakeLLM(_reply("The grand total is 78.60 EUR.", [(1, "grand total 78.60 EUR")]))
    result = await Answerer(llm, "m").answer("What is the total of INV-7001?", HITS)
    assert result.answer.startswith("The grand total")
    assert result.supported and not result.not_in_documents
    assert [c.chunk_id for c in result.citations] == [1] and result.dropped_citations == 0
    assert result.confidence == 0.9 and result.cost_usd == 0
    assert "Question: What is the total" in llm.calls[0][0].messages[0].parts[0].text  # type: ignore[union-attr]


async def test_answer_with_only_bad_citations_is_unsupported_and_low_confidence() -> None:
    llm = FakeLLM(_reply("The total is 99.99 EUR.", [(1, "99.99")]))
    result = await Answerer(llm, "m").answer("total?", HITS)
    assert not result.supported and result.citations == [] and result.dropped_citations == 1
    assert result.confidence <= 0.3


async def test_not_in_documents_is_first_class() -> None:
    llm = FakeLLM(_reply(NOT_IN_DOCUMENTS, [], not_in_documents=True, confidence=0.95))
    result = await Answerer(llm, "m").answer("Who is the CEO?", HITS)
    assert result.not_in_documents and result.supported and result.answer == NOT_IN_DOCUMENTS


async def test_no_hits_means_not_in_documents_without_a_call() -> None:
    llm = FakeLLM("never called")
    result = await Answerer(llm, "m").answer("anything", [])
    assert result.not_in_documents and llm.calls == [] and result.records == []


@pytest.mark.parametrize("bad", ['{"answer": "x"}', "not json"])
async def test_invalid_model_output_raises(bad: str) -> None:
    from doc_intel.llm.errors import StructuredOutputError

    with pytest.raises(StructuredOutputError):
        await Answerer(FakeLLM(bad), "m").answer("q", HITS)
