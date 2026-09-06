from decimal import Decimal

from doc_intel.eval.answers import AnswerCase, AnswerOutcome, AnswersReport, cases_for, is_correct
from tests.models.factories import invoice


def test_cases_carry_expected_values() -> None:
    cases = cases_for("d", invoice())
    assert [c.kind for c in cases] == ["totals", "header", "header", "line_item"]
    assert cases[0].expected == "78.60" and cases[1].expected == "Beauty Supplies Ltd"
    assert cases[2].expected == "2026-09-13" and cases[3].expected == "3"


def test_is_correct_normalises_amounts_and_names() -> None:
    total = AnswerCase(
        question="q", document_id="d", kind="totals", expected="78.60", language="en"
    )
    assert is_correct(total, "The grand total is 78,60 EUR.")
    assert is_correct(total, "78.6 EUR")
    assert not is_correct(total, "The total is 65.50 EUR.")
    supplier = AnswerCase(
        question="q", document_id="d", kind="header", expected="Beauty Supplies Ltd", language="en"
    )
    assert is_correct(supplier, "The supplier is BEAUTY SUPPLIES LTD.")
    assert not is_correct(
        AnswerCase(question="q", document_id=None, kind="x", language="en"), "anything"
    )


def _outcome(
    kind: str, expected: str | None, correct: bool, supported: bool, declined: bool
) -> AnswerOutcome:
    return AnswerOutcome(
        case=AnswerCase(
            question="q",
            document_id="d" if expected else None,
            kind=kind,
            expected=expected,
            language="en",
        ),
        answer="a",
        correct=correct,
        supported=supported,
        not_in_documents=declined,
        citations=1 if supported else 0,
        retrieval_rank=1 if expected else None,
        scoped=True,
        cost_usd=Decimal(0),
        latency_ms=1,
    )


def test_report_metrics() -> None:
    report = AnswersReport(
        config="c",
        model="m",
        rerank=False,
        outcomes=[
            _outcome("totals", "78.60", True, True, False),
            _outcome("totals", "78.60", True, False, False),  # right text, no verified citation
            _outcome("header", "x", False, True, False),
            _outcome("missing", None, False, False, True),
            _outcome("missing", None, False, True, False),
        ],
    )
    assert abs(report.answer_accuracy() - 1 / 3) < 1e-9
    assert report.decline_rate() == 0.5
    assert abs(report.citation_support_rate() - 2 / 3) < 1e-9
    assert report.by_kind() == {"header": 0.0, "totals": 0.5}
