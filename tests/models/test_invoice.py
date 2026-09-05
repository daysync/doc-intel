from typing import Any

import pytest
from pydantic import ValidationError

from doc_intel.models import Invoice
from tests.models.factories import ex, invoice


def test_round_trips_through_json() -> None:
    original = invoice()
    restored = Invoice.model_validate_json(original.model_dump_json())
    assert restored == original
    assert restored.line_items[0].total.value == 37.5


def test_rejects_lowercase_currency() -> None:
    with pytest.raises(ValidationError, match="ISO 4217"):
        invoice(currency=ex("eur"))


def test_rejects_non_iso_language() -> None:
    with pytest.raises(ValidationError, match="ISO 639-1"):
        invoice(language=ex("english"))


def test_unknown_currency_is_allowed_when_not_found() -> None:
    assert invoice(currency=ex(None, 0.0)).currency.value is None


def _object_schemas(schema: dict[str, Any]) -> list[dict[str, Any]]:
    """Every object-typed schema: the root plus everything under $defs."""
    found = [schema] if schema.get("type") == "object" else []
    found += [d for d in schema.get("$defs", {}).values() if d.get("type") == "object"]
    return found


def test_json_schema_is_strict_everywhere() -> None:
    schema = Invoice.model_json_schema()
    objects = _object_schemas(schema)
    assert len(objects) > 5
    for obj in objects:
        assert obj["additionalProperties"] is False, obj.get("title")
        assert set(obj["required"]) == set(obj["properties"]), obj.get("title")
