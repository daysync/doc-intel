"""The generic invoice schema an extractor fills in.

Every leaf is an ``Extracted[...]`` so each value carries its source quote and confidence.
Money is ``Decimal``, never ``float``. ``Money`` and ``Day`` accept the formats models and OCR
actually produce (see ``parsing.py``) and validate to ``Decimal`` and ``date``.
Language and currency are detected from the document,
not configured, because one account can receive invoices in several languages. Nothing here
is specific to any host product; a host maps this to its own suppliers and expenses.
"""

import re

from pydantic import BaseModel, ConfigDict, field_validator

from doc_intel.models.fields import Extracted
from doc_intel.models.parsing import Day, Money

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
    quantity: Extracted[Money]
    unit: Extracted[str]
    unit_price: Extracted[Money]
    total: Extracted[Money]


class Tax(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Extracted[str]
    rate: Extracted[Money]
    amount: Extracted[Money]


class Totals(BaseModel):
    model_config = ConfigDict(extra="forbid")

    subtotal: Extracted[Money]
    tax_total: Extracted[Money]
    grand_total: Extracted[Money]


class Invoice(BaseModel):
    model_config = ConfigDict(extra="forbid")

    supplier: Party
    buyer: Party
    number: Extracted[str]
    issue_date: Extracted[Day]
    due_date: Extracted[Day]
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
