"""Price table and cost calculation.

A Python dict rather than YAML: typed, checked by mypy, zero parsing code, and a price
change is a one-line diff. Unknown paid models raise instead of silently costing $0,
because "cost per document" is a promise the README makes.

Prices are USD per million tokens, first-party API rates. Update the date when you change them.
"""

from collections.abc import Mapping
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from doc_intel.llm.errors import UnknownModelPriceError
from doc_intel.llm.types import RawCompletion

MILLION = Decimal(1_000_000)


class ModelPrice(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    input_per_mtok: Decimal
    output_per_mtok: Decimal
    cached_input_per_mtok: Decimal


def _price(input_: str, output: str, cached: str) -> ModelPrice:
    return ModelPrice(
        input_per_mtok=Decimal(input_),
        output_per_mtok=Decimal(output),
        cached_input_per_mtok=Decimal(cached),
    )


# (provider, model) -> price.
PRICES: dict[tuple[str, str], ModelPrice] = {
    # Anthropic, first-party API, 2026-06. Cache reads are 10% of input.
    ("anthropic", "claude-opus-5"): _price("5", "25", "0.50"),
    ("anthropic", "claude-sonnet-5"): _price("2", "10", "0.20"),
    ("anthropic", "claude-haiku-4-5"): _price("1", "5", "0.10"),
    # OpenAI, developers.openai.com/api/docs/pricing, 2026-09-05.
    ("openai", "gpt-5.5"): _price("5", "30", "0.50"),
    ("openai", "gpt-5.4"): _price("2.50", "15", "0.25"),
    ("openai", "gpt-5.4-mini"): _price("0.75", "4.50", "0.075"),
    ("openai", "gpt-5-mini"): _price("0.25", "2", "0.025"),
}

# Providers whose calls cost nothing per token (local inference, fixtures).
FREE_PROVIDERS: frozenset[str] = frozenset({"ollama", "fake"})


class Pricing:
    def __init__(
        self,
        table: Mapping[tuple[str, str], ModelPrice] = PRICES,
        free_providers: frozenset[str] = FREE_PROVIDERS,
    ) -> None:
        self._table = dict(table)
        self._free = free_providers

    def cost(self, provider: str, model: str, raw: RawCompletion) -> Decimal:
        if provider in self._free:
            return Decimal(0)
        price = self._table.get((provider, model))
        if price is None:
            raise UnknownModelPriceError(provider, model)
        uncached = max(raw.input_tokens - raw.cached_input_tokens, 0)
        return (
            Decimal(uncached) * price.input_per_mtok
            + Decimal(raw.cached_input_tokens) * price.cached_input_per_mtok
            + Decimal(raw.output_tokens) * price.output_per_mtok
        ) / MILLION
