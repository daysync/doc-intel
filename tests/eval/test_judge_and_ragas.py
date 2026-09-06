import json

import pytest

from doc_intel.eval.judge import Judge
from doc_intel.eval.ragas_style import RagasStyle
from doc_intel.rag.answer import NOT_IN_DOCUMENTS
from tests.llm.fakes import FakeLLM
from tests.llm.test_embeddings import FakeEmbedder


def test_judge_refuses_to_grade_its_own_model() -> None:
    with pytest.raises(ValueError, match="must not be the model under test"):
        Judge(FakeLLM(""), "m", model_under_test="m")
    Judge(FakeLLM(""), "judge-7b", model_under_test="m")


async def test_grade_returns_a_verdict_with_reason() -> None:
    llm = FakeLLM(json.dumps({"grade": "correct", "reason": "same amount, different format"}))
    verdict = await Judge(llm, "j").grade("total?", "78.60 EUR", "78,60 EUR")
    assert verdict.grade == "correct" and "format" in verdict.reason
    assert "Reference answer: 78.60 EUR" in llm.calls[0][0].messages[0].parts[0].text  # type: ignore[union-attr]


async def test_faithfulness_is_the_share_of_supported_claims() -> None:
    llm = FakeLLM(
        json.dumps(
            {
                "claims": [
                    {"claim": "total is 78.60", "supported": True},
                    {"claim": "paid in cash", "supported": False},
                ]
            }
        )
    )
    metrics = RagasStyle(Judge(llm, "j"), FakeEmbedder())
    faith, total, supported = await metrics.faithfulness(
        "The total is 78.60, paid in cash.", ["grand total 78.60 EUR"]
    )
    assert (faith, total, supported) == (0.5, 2, 1)
    assert await metrics.faithfulness(NOT_IN_DOCUMENTS, ["x"]) == (None, 0, 0)
    assert await metrics.faithfulness("anything", []) == (None, 0, 0)


async def test_answer_relevancy_compares_generated_questions_to_the_real_one() -> None:
    class Emb(FakeEmbedder):
        async def _embed_raw(self, texts):  # type: ignore[no-untyped-def]
            from doc_intel.llm.embeddings import RawEmbeddings

            # the real question and a matching generated question align; a stray one does not
            table = {
                "What is the total?": [1.0, 0.0],
                "total?": [1.0, 0.0],
                "who is the ceo?": [0.0, 1.0],
            }
            return RawEmbeddings(vectors=[table.get(t, [0.5, 0.5]) for t in texts], input_tokens=1)

    llm = FakeLLM(json.dumps({"questions": ["total?", "who is the ceo?"]}))
    metrics = RagasStyle(Judge(llm, "j"), Emb(dims=2))
    relevancy = await metrics.answer_relevancy("What is the total?", "78.60 EUR")
    assert relevancy == pytest.approx(0.5)
    assert await metrics.answer_relevancy("q", NOT_IN_DOCUMENTS) is None


async def test_score_bundles_both() -> None:
    llm = FakeLLM(
        [
            json.dumps({"claims": [{"claim": "c", "supported": True}]}),
            json.dumps({"questions": ["q"]}),
        ]
    )
    scores = await RagasStyle(Judge(llm, "j"), FakeEmbedder()).score("q", "answer", ["ctx"])
    assert (
        scores.faithfulness == 1.0
        and scores.claims_total == 1
        and scores.answer_relevancy is not None
    )
