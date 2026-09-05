from datetime import date
from decimal import Decimal

import pytest

from doc_intel.models import Extracted
from doc_intel.models.parsing import Day, Money, parse_amount, parse_date


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("1,731.49", "1731.49"),
        ("1 234,50", "1234.50"),
        ("1.234,50", "1234.50"),
        ("2 095,10", "2095.10"),
        ("9 pcs", "9"),
        ("12", "12"),
        ("87.93", "87.93"),
        ("0,20", "0.20"),
        ("1.234.567,89", "1234567.89"),
        ("EUR 78.60", "78.60"),
        ("-5.00", "-5.00"),
    ],
)
def test_parse_amount(raw: str, expected: str) -> None:
    assert parse_amount(raw) == Decimal(expected)


def test_parse_amount_leaves_garbage_alone() -> None:
    assert parse_amount("n/a") == "n/a"
    assert parse_amount(None) is None
    assert parse_amount(Decimal("1.5")) == Decimal("1.5")


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("2026-08-30", date(2026, 8, 30)),
        ("30.08.2026", date(2026, 8, 30)),
        ("30/08/2026", date(2026, 8, 30)),
        ("03 Aug 2026", date(2026, 8, 3)),
        ("Aug 3, 2026", date(2026, 8, 3)),
        ("3 September 2026", date(2026, 9, 3)),
    ],
)
def test_parse_date(raw: str, expected: date) -> None:
    assert parse_date(raw) == expected


def test_parse_date_leaves_garbage_alone() -> None:
    assert parse_date("2026-13-45") == "2026-13-45"
    assert parse_date("yesterday") == "yesterday"
    # a syntactically valid but absurd date parses; plausibility is a validation rule's job
    assert parse_date("0308-02-20") == date(308, 2, 20)


def test_annotated_types_work_inside_extracted() -> None:
    money = Extracted[Money].model_validate(
        {"value": "1,731.49", "quote": "1,731.49", "confidence": 1}
    )
    assert money.value == Decimal("1731.49")
    day = Extracted[Day].model_validate(
        {"value": "03 Aug 2026", "quote": "03 Aug 2026", "confidence": 1}
    )
    assert day.value == date(2026, 8, 3)
    assert Extracted[Money].model_json_schema()["properties"][
        "value"
    ]  # schema unchanged by the validator
