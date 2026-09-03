"""The generic invoice schema an extractor fills in.

Every leaf is an ``Extracted[...]`` so each value carries its source quote and confidence.
Money is ``Decimal``, never ``float``. Language and currency are detected from the document,
not configured, because one account can receive invoices in several languages. Nothing here
is specific to any host product; a host maps this to its own suppliers and expenses.
"""

import re
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, field_validator

from doc_intel.models.fields import Extracted

_ISO_4217 = re.compile(r"^[A-Z]{3}$")
_ISO_639_1 = re.compile(r"^[a-z]{2}$")


class Party(BaseModel):
    """Supplier or buyer as printed on the document."""

    model_config = ConfigDict(extra="forbid")

    name: Extracted[str]
    tax_id: Extracted[str]
    address: Extracted[str]


class LineItem(BaseModel):
    model_config = ConfigDict(extra="forbid")

    description: Extracted[str]
    quantity: Extracted[Decimal]
    unit: Extracted[str]
    unit_price: Extracted[Decimal]
    total: Extracted[Decimal]


class Tax(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Extracted[str]
    rate: Extracted[Decimal]
    amount: Extracted[Decimal]


class Totals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subtotal: Extracted[Decimal]
    tax_total: Extracted[Decimal]
    grand_total: Extracted[Decimal]


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier: Party
    buyer: Party
    number: Extracted[str]
    issue_date: Extracted[date]
    due_date: Extracted[date]
    currency: Extracted[str]
    language: Extracted[str]
    line_items: list[LineItem]
    taxes: list[Tax]
    totals: Totals

    @field_validator("currency")
    @classmethod
    def _currency_is_iso_4217(cls, field: Extracted[str]) -> Extracted[str]:
        if field.value is not None and not _ISO_4217.match(field.value):
            raise ValueError(f"currency must be an ISO 4217 code, got {field.value!r}")
        return field

    @field_validator("language")
    @classmethod
    def _language_is_iso_639_1(cls, field: Extracted[str]) -> Extracted[str]:
        if field.value is not None and not _ISO_639_1.match(field.value):
            raise ValueError(f"language must be an ISO 639-1 code, got {field.value!r}")
        return field
