from decimal import Decimal

from doc_intel.eval.answers import AnswerCase, AnswerOutcome, AnswersReport
from doc_intel.eval.compare import compare


def _report(name: str, pattern: list[bool]) -> AnswersReport:
    outcomes = [
        AnswerOutcome(
            case=AnswerCase(
                question=f"q{i}", document_id="d", kind="totals", expected="x", language="en"
            ),
            answer="a",
            correct=ok,
            supported=True,
            not_in_documents=False,
            citations=1,
            retrieval_rank=1,
            scoped=True,
            cost_usd=Decimal(0),
            latency_ms=1,
        )
        for i, ok in enumerate(pattern)
    ]
    return AnswersReport(config=name, model="m", rerank=name == "b", outcomes=outcomes)


def test_compare_reports_a_real_improvement_and_a_tie() -> None:
    base = [i % 2 == 0 for i in range(40)]
    better = [True if i % 4 == 1 else v for i, v in enumerate(base)]
    text = compare(_report("a", base), _report("b", better))
    assert "verdict: b is better" in text and "rerank=on" in text
    tie = compare(_report("a", base), _report("b", base))
    assert "no measurable difference" in tie
    assert "no shared questions" in compare(_report("a", []), _report("b", []))
