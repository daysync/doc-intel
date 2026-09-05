"""Small builders so tests read as intent, not as constructor noise."""

from datetime import date
from decimal import Decimal
from typing import Any

from doc_intel.models import Extracted, Invoice, LineItem, Party, Tax, Totals


def ex(value: Any, confidence: float = 0.9) -> Extracted[Any]:
    return Extracted[Any](
        value=value, quote=None if value is None else str(value), confidence=confidence
    )


def party(name: str) -> Party:
    return Party(name=ex(name), tax_id=ex(None, 0.0), address=ex(None, 0.0))


def line(description: str, qty: str, price: str) -> LineItem:
    return LineItem(
        description=ex(description),
        quantity=ex(Decimal(qty)),
        unit=ex("pcs"),
        unit_price=ex(Decimal(price)),
        total=ex(Decimal(qty) * Decimal(price)),
    )


def invoice(**overrides: Any) -> Invoice:
    base: dict[str, Any] = {
        "supplier": party("Beauty Supplies Ltd"),
        "buyer": party("Salon Nova"),
        "number": ex("INV-1042"),
        "issue_date": ex(date(2026, 8, 30)),
        "due_date": ex(date(2026, 9, 13)),
        "currency": ex("EUR"),
        "language": ex("en"),
        "line_items": [line("Shampoo 1L", "3", "12.50"), line("Conditioner 1L", "2", "14.00")],
        "taxes": [Tax(name=ex("VAT"), rate=ex(Decimal("0.20")), amount=ex(Decimal("13.10")))],
        "totals": Totals(
            subtotal=ex(Decimal("65.50")),
            tax_total=ex(Decimal("13.10")),
            grand_total=ex(Decimal("78.60")),
        ),
    }
    base.update(overrides)
    return Invoice(**base)
