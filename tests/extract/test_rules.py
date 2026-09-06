from datetime import date
from decimal import Decimal
from pathlib import Path

from doc_intel.dataset.generate import generate_dataset, load_truth
from doc_intel.extract.rules import cross_document_issues, error_codes, validate_invoice
from doc_intel.models import Extracted, Severity
from tests.models.factories import ex, invoice

TODAY = date(2026, 9, 5)


def test_consistent_invoice_has_no_issues() -> None:
    assert validate_invoice(invoice(), today=TODAY) == []


def test_totals_mismatch_is_an_error() -> None:
    bad = invoice()
    bad.totals.grand_total = ex(Decimal("88.60"))
    issues = validate_invoice(bad, today=TODAY)
    assert error_codes(issues) == {"totals_mismatch"}
    assert issues[0].severity is Severity.ERROR
    assert issues[0].expected == "78.60" and issues[0].actual == "88.60"


def test_line_math_and_subtotal() -> None:
    bad = invoice()
    bad.line_items[0].total = ex(Decimal("40.00"))  # 3 x 12.50 is 37.50
    codes = error_codes(validate_invoice(bad, today=TODAY))
    assert {"line_item_math", "subtotal_mismatch"} <= codes


def test_tax_rate_mismatch_accepts_percent_or_fraction() -> None:
    ok_percent = invoice()
    ok_percent.taxes[0].rate = ex(Decimal("20"))
    assert "tax_rate_mismatch" not in error_codes(validate_invoice(ok_percent, today=TODAY))
    bad = invoice()
    bad.taxes[0].amount = ex(Decimal("5.00"))
    assert {"tax_rate_mismatch", "tax_total_mismatch"} <= error_codes(
        validate_invoice(bad, today=TODAY)
    )


def test_dates() -> None:
    odd = invoice()
    odd.issue_date = Extracted[date](
        value=date(308, 2, 20), quote="Date: 03 Aug 2026", confidence=1.0
    )
    codes = error_codes(validate_invoice(odd, today=TODAY))
    assert {"date_implausible", "date_quote_mismatch"} <= codes
    quote_mismatch = next(
        i for i in validate_invoice(odd, today=TODAY) if i.code == "date_quote_mismatch"
    )
    assert quote_mismatch.expected == "2026-08-03"

    backwards = invoice()
    backwards.due_date = ex(date(2026, 8, 1))
    assert "due_before_issue" in error_codes(validate_invoice(backwards, today=TODAY))


def test_missing_fields_and_unknown_currency_and_low_confidence() -> None:
    thin = invoice()
    thin.number = Extracted[str](value=None, quote=None, confidence=0.0)
    thin.currency = ex("XXX")
    thin.supplier.name = Extracted[str](value="Beauty Supplies Ltd", quote="Beauty", confidence=0.3)
    issues = validate_invoice(thin, today=TODAY)
    codes = error_codes(issues)
    assert {"missing_field", "currency_unknown", "low_confidence"} <= codes
    assert next(i for i in issues if i.code == "low_confidence").field == "supplier.name"


def test_cross_document_duplicates_and_supplier_mismatch() -> None:
    a, b, c = invoice(), invoice(), invoice()
    b.number = ex("inv-1042")  # same number, different case and punctuation
    c.number = ex("INV-1042")
    c.supplier.name = ex("Other Supplier")
    result = cross_document_issues({"a": a, "b": b, "c": c})
    assert error_codes(result["a"]) == {"duplicate_number", "supplier_mismatch"}
    assert set(result["a"][0].related_document_ids) == {"b", "c"}
    assert cross_document_issues({"a": a})["a"] == []


def test_planted_dataset_issues_are_found(tmp_path: Path) -> None:
    manifest = generate_dataset(tmp_path, n=12, seed=3)
    truths = {d.id: load_truth(tmp_path / d.id) for d in manifest.documents}
    cross = cross_document_issues(truths)
    for doc in manifest.documents:
        found = error_codes(validate_invoice(truths[doc.id], today=TODAY)) | error_codes(
            cross[doc.id]
        )
        expected = set(doc.expected_issues)
        assert expected <= found, (doc.id, expected, found)
        if not expected:
            assert not any(
                i.severity is Severity.ERROR for i in validate_invoice(truths[doc.id], today=TODAY)
            ), doc.id
