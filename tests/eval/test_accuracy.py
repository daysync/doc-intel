from datetime import date
from decimal import Decimal

from doc_intel.eval.accuracy import accuracy, compare
from doc_intel.eval.run import DocumentEval, EvalReport
from doc_intel.models import Extracted
from tests.models.factories import ex, invoice


def test_identical_invoices_score_everything_true() -> None:
    scores = compare(invoice(), invoice())
    assert all(scores.values())
    assert "line_items.count" in scores and "line_items.total" in scores


def test_tolerances_and_normalisation() -> None:
    got = invoice()
    got.totals.grand_total = ex(Decimal("78.61"))  # within 0.01
    got.supplier.name = ex("beauty supplies ltd.")  # case and punctuation ignored
    got.number = ex("inv 1042")  # separators ignored
    got.issue_date = ex(date(2026, 8, 31))  # dates are exact
    scores = compare(invoice(), got)
    assert scores["totals.grand_total"] and scores["supplier.name"] and scores["number"]
    assert not scores["issue_date"]


def test_missing_line_items_and_none_values() -> None:
    got = invoice()
    got.line_items = got.line_items[:1]
    got.currency = Extracted[str](value=None, quote=None, confidence=0)
    scores = compare(invoice(), got)
    assert not scores["line_items.count"]
    assert not scores["line_items.total"]  # second line missing
    assert not scores["currency"]


def test_accuracy_is_a_mean_per_field_plus_overall() -> None:
    result = accuracy([{"a": True, "b": False}, {"a": True, "b": True}])
    assert result == {"a": 1.0, "b": 0.5, "all_fields": 0.75}
    assert accuracy([]) == {}


def test_report_breakdowns_skip_failed_documents() -> None:
    docs = [
        DocumentEval(
            id="1", language="en", layout="classic", level="clean", ok=True, scores={"a": True}
        ),
        DocumentEval(
            id="2", language="uk", layout="classic", level="hard", ok=True, scores={"a": False}
        ),
        DocumentEval(id="3", language="uk", layout="receipt", level="hard", ok=False, error="boom"),
    ]
    report = EvalReport(config="c", model="m", documents=docs)
    assert report.overall()["all_fields"] == 0.5
    assert report.by("language") == {"en": 1.0, "uk": 0.0}
    assert report.by("level") == {"clean": 1.0, "hard": 0.0}
