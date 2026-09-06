"""Validation rules: what a model cannot be trusted to check about its own answer.

Field rules look at one invoice; cross-document rules look at a batch. Every finding is a
``ValidationIssue`` with a stable code, so evals can count them and hosts can route them.
Tolerances exist because printed totals are rounded per line.
"""

import re
from collections.abc import Iterable, Mapping
from datetime import date, timedelta
from decimal import Decimal

from doc_intel.models import Invoice, Severity, ValidationIssue, iter_extracted
from doc_intel.models.parsing import parse_date

KNOWN_CURRENCIES = frozenset(
    "EUR USD GBP UAH GEL RUB PLN CZK HUF RON BGN TRY CHF SEK NOK DKK AMD AZN KZT MDL BYN "
    "ILS AED SAR JPY CNY INR CAD AUD".split()
)
ABSOLUTE_TOLERANCE = Decimal("0.02")
RELATIVE_TOLERANCE = Decimal("0.005")
LOW_CONFIDENCE = 0.5
_DATE_IN_TEXT = re.compile(
    r"\d{4}-\d{2}-\d{2}|\d{1,2}[./-]\d{1,2}[./-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3,9}\.?,?\s+\d{4}|[A-Za-z]{3,9}\s+\d{1,2},?\s+\d{4}"
)


def _issue(
    code: str,
    severity: Severity,
    message: str,
    field: str | None = None,
    expected: object = None,
    actual: object = None,
) -> ValidationIssue:
    return ValidationIssue(
        code=code,
        severity=severity,
        message=message,
        field=field,
        expected=None if expected is None else str(expected),
        actual=None if actual is None else str(actual),
    )


def _close(a: Decimal, b: Decimal) -> bool:
    return abs(a - b) <= max(ABSOLUTE_TOLERANCE, abs(b) * RELATIVE_TOLERANCE)


def validate_invoice(invoice: Invoice, today: date | None = None) -> list[ValidationIssue]:
    today = today or date.today()
    issues: list[ValidationIssue] = []
    issues += _required(invoice)
    issues += _line_items(invoice)
    issues += _totals(invoice)
    issues += _taxes(invoice)
    issues += _dates(invoice, today)
    issues += _currency(invoice)
    issues += _low_confidence(invoice)
    return issues


def _required(inv: Invoice) -> list[ValidationIssue]:
    required = (
        ("number", inv.number.value, Severity.ERROR),
        ("supplier.name", inv.supplier.name.value, Severity.ERROR),
        ("totals.grand_total", inv.totals.grand_total.value, Severity.ERROR),
        ("issue_date", inv.issue_date.value, Severity.WARNING),
        ("currency", inv.currency.value, Severity.WARNING),
    )
    return [
        _issue("missing_field", severity, f"{field} was not found on the document.", field)
        for field, value, severity in required
        if value is None
    ]


def _line_items(inv: Invoice) -> list[ValidationIssue]:
    issues = []
    for index, item in enumerate(inv.line_items):
        qty, price, total = item.quantity.value, item.unit_price.value, item.total.value
        if qty is None or price is None or total is None:
            continue
        expected = qty * price
        if not _close(expected, total):
            issues.append(
                _issue(
                    "line_item_math",
                    Severity.WARNING,
                    f"Line {index + 1}: quantity times unit price does not equal the line total.",
                    f"line_items[{index}].total",
                    expected.quantize(Decimal("0.01")),
                    total,
                )
            )
    return issues


def _totals(inv: Invoice) -> list[ValidationIssue]:
    issues = []
    subtotal, tax_total, grand = (
        inv.totals.subtotal.value,
        inv.totals.tax_total.value,
        inv.totals.grand_total.value,
    )
    line_totals = [item.total.value for item in inv.line_items]
    if subtotal is not None and line_totals and all(t is not None for t in line_totals):
        line_sum = sum((t for t in line_totals if t is not None), Decimal(0))
        if not _close(line_sum, subtotal):
            issues.append(
                _issue(
                    "subtotal_mismatch",
                    Severity.ERROR,
                    "Line items do not add up to the subtotal.",
                    "totals.subtotal",
                    line_sum,
                    subtotal,
                )
            )
    if subtotal is not None and grand is not None:
        expected = subtotal + (tax_total or Decimal(0))
        if not _close(expected, grand):
            issues.append(
                _issue(
                    "totals_mismatch",
                    Severity.ERROR,
                    "Subtotal plus tax does not equal the grand total.",
                    "totals.grand_total",
                    expected,
                    grand,
                )
            )
    return issues


def _taxes(inv: Invoice) -> list[ValidationIssue]:
    issues = []
    subtotal = inv.totals.subtotal.value
    amounts = []
    for index, tax in enumerate(inv.taxes):
        rate, amount = tax.rate.value, tax.amount.value
        if amount is not None:
            amounts.append(amount)
        if rate is None or amount is None or subtotal is None:
            continue
        fraction = rate / 100 if rate > 1 else rate
        expected = (subtotal * fraction).quantize(Decimal("0.01"))
        if not _close(expected, amount):
            issues.append(
                _issue(
                    "tax_rate_mismatch",
                    Severity.WARNING,
                    f"Tax line {index + 1}: rate times subtotal does not equal the tax amount.",
                    f"taxes[{index}].amount",
                    expected,
                    amount,
                )
            )
    tax_total = inv.totals.tax_total.value
    if amounts and tax_total is not None and not _close(sum(amounts, Decimal(0)), tax_total):
        issues.append(
            _issue(
                "tax_total_mismatch",
                Severity.WARNING,
                "Tax lines do not add up to the tax total.",
                "totals.tax_total",
                sum(amounts, Decimal(0)),
                tax_total,
            )
        )
    return issues


def _dates(inv: Invoice, today: date) -> list[ValidationIssue]:
    issues = []
    for field_name, field in (("issue_date", inv.issue_date), ("due_date", inv.due_date)):
        value = field.value
        if value is None:
            continue
        if value < date(2000, 1, 1) or value > today + timedelta(days=60):
            issues.append(
                _issue(
                    "date_implausible",
                    Severity.WARNING,
                    f"{field_name} {value} is not plausible.",
                    field_name,
                    None,
                    value,
                )
            )
        quoted = _date_in(field.quote)
        if quoted is not None and quoted != value:
            issues.append(
                _issue(
                    "date_quote_mismatch",
                    Severity.WARNING,
                    f"{field_name} value disagrees with its quoted text.",
                    field_name,
                    quoted,
                    value,
                )
            )
    issue, due = inv.issue_date.value, inv.due_date.value
    if issue is not None and due is not None and due < issue:
        issues.append(
            _issue(
                "due_before_issue",
                Severity.WARNING,
                "Due date is before the issue date.",
                "due_date",
                issue,
                due,
            )
        )
    return issues


def _date_in(text: str | None) -> date | None:
    if not text:
        return None
    match = _DATE_IN_TEXT.search(text)
    if not match:
        return None
    parsed = parse_date(match.group(0))
    return parsed if isinstance(parsed, date) else None


def _currency(inv: Invoice) -> list[ValidationIssue]:
    code = inv.currency.value
    if code is None or code in KNOWN_CURRENCIES:
        return []
    return [
        _issue(
            "currency_unknown",
            Severity.WARNING,
            f"Currency {code} is not a known code.",
            "currency",
            None,
            code,
        )
    ]


def _low_confidence(inv: Invoice) -> list[ValidationIssue]:
    return [
        _issue(
            "low_confidence",
            Severity.INFO,
            f"{path} was read with low confidence ({field.confidence:.2f}); review it.",
            path,
            None,
            field.value,
        )
        for path, field in iter_extracted(inv)
        if field.value is not None and field.confidence < LOW_CONFIDENCE
    ]


# --- cross-document


def _norm(text: str | None) -> str:
    return re.sub(r"[\s\-_/.]", "", text or "").upper()


def cross_document_issues(documents: Mapping[str, Invoice]) -> dict[str, list[ValidationIssue]]:
    """Duplicate invoice numbers across a batch, and same number with a different supplier."""
    by_number: dict[str, list[str]] = {}
    for doc_id, invoice in documents.items():
        key = _norm(invoice.number.value)
        if key:
            by_number.setdefault(key, []).append(doc_id)

    result: dict[str, list[ValidationIssue]] = {doc_id: [] for doc_id in documents}
    for ids in by_number.values():
        if len(ids) < 2:
            continue
        suppliers = {_norm(documents[i].supplier.name.value) for i in ids}
        for doc_id in ids:
            others = [i for i in ids if i != doc_id]
            result[doc_id].append(
                ValidationIssue(
                    code="duplicate_number",
                    severity=Severity.WARNING,
                    message=(
                        f"Invoice number {documents[doc_id].number.value} also appears on "
                        f"{len(others)} other document(s)."
                    ),
                    field="number",
                    related_document_ids=others,
                )
            )
            if len(suppliers) > 1:
                result[doc_id].append(
                    ValidationIssue(
                        code="supplier_mismatch",
                        severity=Severity.WARNING,
                        message="Documents sharing this invoice number name different suppliers.",
                        field="supplier.name",
                        related_document_ids=others,
                    )
                )
    return result


def error_codes(issues: Iterable[ValidationIssue]) -> set[str]:
    return {issue.code for issue in issues}
