from decimal import Decimal

import pytest

from doc_intel.llm.errors import UnknownModelPriceError
from doc_intel.llm.pricing import ModelPrice, Pricing
from doc_intel.llm.types import RawCompletion


def test_cost_splits_cached_and_uncached_input() -> None:
    pricing = Pricing(
        table={
            ("acme", "m1"): ModelPrice(
                input_per_mtok=Decimal(10),
                output_per_mtok=Decimal(30),
                cached_input_per_mtok=Decimal(1),
            )
        },
        free_providers=frozenset(),
    )
    raw = RawCompletion(
        text="", input_tokens=1_000_000, output_tokens=100_000, cached_input_tokens=400_000
    )
    # 600k uncached * $10 + 400k cached * $1 + 100k out * $30, all per million
    assert pricing.cost("acme", "m1", raw) == Decimal("6") + Decimal("0.4") + Decimal("3")


def test_unknown_paid_model_raises() -> None:
    raw = RawCompletion(text="", input_tokens=1, output_tokens=1)
    with pytest.raises(UnknownModelPriceError):
        Pricing().cost("anthropic", "claude-does-not-exist", raw)


def test_free_provider_costs_zero_for_any_model() -> None:
    raw = RawCompletion(text="", input_tokens=5000, output_tokens=5000)
    assert Pricing().cost("ollama", "anything:latest", raw) == 0


def test_anthropic_opus_5_reference_price() -> None:
    raw = RawCompletion(text="", input_tokens=1_000, output_tokens=1_000)
    assert Pricing().cost("anthropic", "claude-opus-5", raw) == Decimal("0.03")
