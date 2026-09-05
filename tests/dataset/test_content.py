import random
from datetime import date
from decimal import Decimal
from typing import Any

import pytest

from doc_intel.dataset.content import LANGUAGES, format_amount, format_date, generate_content
from doc_intel.models import Extracted


def val(field: Extracted[Any]) -> Any:
    """Ground truth is never missing; unwrap and prove it to the type checker."""
    assert field.value is not None
    return field.value


@pytest.mark.parametrize("language", LANGUAGES)
def test_ground_truth_is_internally_consistent(language: str) -> None:
    invoice = generate_content(random.Random(1), language).invoice
    line_sum = sum((val(item.total) for item in invoice.line_items), Decimal(0))
    assert val(invoice.totals.subtotal) == line_sum
    tax = invoice.taxes[0]
    assert val(tax.amount) == val(invoice.totals.tax_total)
    assert val(invoice.totals.grand_total) == line_sum + val(tax.amount)
    for item in invoice.line_items:
        assert val(item.quantity) * val(item.unit_price) == val(item.total)
    assert val(invoice.language) == language
    assert len(val(invoice.currency)) == 3


def test_same_seed_same_invoice() -> None:
    a = generate_content(random.Random(42), "uk").invoice
    b = generate_content(random.Random(42), "uk").invoice
    assert a == b
    assert generate_content(random.Random(43), "uk").invoice != a


def test_every_quote_is_the_rendered_string() -> None:
    invoice = generate_content(random.Random(3), "ru").invoice
    assert invoice.totals.grand_total.quote == format_amount(val(invoice.totals.grand_total), "ru")
    assert invoice.issue_date.quote == format_date(val(invoice.issue_date), "ru")
    assert invoice.number.quote == invoice.number.value


def test_amount_and_date_formats_per_language() -> None:
    assert format_amount(Decimal("1234.5"), "en") == "1,234.50"
    assert format_amount(Decimal("1234.5"), "uk") == "1 234,50"
    assert format_date(date(2026, 8, 30), "en") == "30 Aug 2026"
    assert format_date(date(2026, 8, 30), "ka") == "30.08.2026"
