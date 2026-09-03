import pytest
from pydantic import BaseModel, ValidationError

from doc_intel.models import Extracted, iter_extracted


def test_holds_value_quote_and_confidence() -> None:
    field = Extracted[str](value="INV-42", quote="Invoice No. INV-42", confidence=0.93)
    assert field.value == "INV-42"
    assert field.quote == "Invoice No. INV-42"
    assert field.confidence == 0.93


def test_missing_field_is_none_with_low_confidence() -> None:
    field = Extracted[int](value=None, quote=None, confidence=0.0)
    assert field.value is None


def test_confidence_is_bounded() -> None:
    with pytest.raises(ValidationError):
        Extracted[str](value="x", quote="x", confidence=1.5)


def test_extra_keys_are_rejected() -> None:
    with pytest.raises(ValidationError):
        Extracted[str].model_validate({"value": "x", "quote": "x", "confidence": 1, "note": "?"})


def test_json_schema_is_strict() -> None:
    schema = Extracted[str].model_json_schema()
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"value", "quote", "confidence"}


class _Inner(BaseModel):
    price: Extracted[float]


class _Outer(BaseModel):
    name: Extracted[str]
    items: list[_Inner]
    plain: int


def test_iter_extracted_walks_nested_models_and_lists() -> None:
    doc = _Outer(
        name=Extracted[str](value="a", quote="a", confidence=0.9),
        items=[
            _Inner(price=Extracted[float](value=1.0, quote="1.00", confidence=0.8)),
            _Inner(price=Extracted[float](value=None, quote=None, confidence=0.1)),
        ],
        plain=7,
    )
    paths = {path: field.confidence for path, field in iter_extracted(doc)}
    assert paths == {"name": 0.9, "items[0].price": 0.8, "items[1].price": 0.1}
