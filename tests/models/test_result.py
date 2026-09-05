from decimal import Decimal

from doc_intel.models import ProcessResult, Severity, Timings, ValidationIssue
from tests.models.factories import invoice


def _result(issues: list[ValidationIssue] | None = None) -> ProcessResult:
    return ProcessResult(
        document_id="doc-1",
        invoice=invoice(),
        issues=issues or [],
        cost_usd=Decimal("0.0042"),
        timings=Timings(ocr_ms=310, extract_ms=1450, total_ms=1800),
    )


def test_confidence_is_flat_path_to_score() -> None:
    confidence = _result().confidence
    assert confidence["number"] == 0.9
    assert confidence["supplier.tax_id"] == 0.0
    assert confidence["line_items[1].unit_price"] == 0.9
    assert "line_items" not in confidence


def test_has_errors_only_for_error_severity() -> None:
    warning = ValidationIssue(
        code="due_date_past", severity=Severity.WARNING, message="Due date is past."
    )
    error = ValidationIssue(
        code="totals_mismatch",
        severity=Severity.ERROR,
        message="Line items do not add up to the subtotal.",
        field="totals.subtotal",
        expected="65.50",
        actual="60.00",
    )
    assert not _result([warning]).has_errors
    assert _result([warning, error]).has_errors


def test_skipped_stages_are_none() -> None:
    timings = Timings(total_ms=5)
    assert timings.preprocess_ms is None
    assert timings.total_ms == 5
