"""``make llm-smoke``: the same structured prompt through every configured provider.

Prints tokens, cost and latency side by side. This is the project's first measurement and
the quickest way to see whether an adapter, a key, or a local model is broken.
Run with ``LLM_RECORD=1`` to store the responses as fixtures.
"""

import asyncio
import sys
from datetime import date
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from doc_intel.api.settings import Settings, get_settings
from doc_intel.llm.errors import LLMError
from doc_intel.llm.factory import build_llm
from doc_intel.llm.log import CallLog
from doc_intel.llm.types import LLMRequest, Message, TextPart
from doc_intel.models import Extracted

SYSTEM = (
    "You extract fields from supplier invoices. Answer only with JSON matching the schema. "
    "For every field give the verbatim quote you read it from and a confidence between 0 and 1. "
    "Use null for values not present."
)

SYNTHETIC_INVOICE = """\
BEAUTY SUPPLIES LTD          INVOICE
Invoice No: INV-2026-0917    Date: 30.08.2026
Bill to: Salon Nova, 12 Rustaveli Ave, Tbilisi

Item                     Qty   Unit price   Total
Argan shampoo 1L          3     12.50       37.50
Keratin conditioner 1L    2     14.00       28.00
                                  Subtotal   65.50
                                  VAT 20%    13.10
                                  TOTAL EUR  78.60
"""


class SmokeExtraction(BaseModel):
    """A slice of the real Invoice schema, enough to exercise Extracted[T] end to end."""

    model_config = ConfigDict(extra="forbid")

    number: Extracted[str]
    issue_date: Extracted[date]
    currency: Extracted[str]
    grand_total: Extracted[Decimal]
    line_item_count: Extracted[int]


def candidates(settings: Settings) -> list[tuple[str, str]]:
    return [
        ("anthropic", "claude-opus-5"),
        ("anthropic", "claude-sonnet-5"),
        ("anthropic", "claude-haiku-4-5"),
        ("openai", "gpt-5.5"),
        ("openai", "gpt-5.4-mini"),
        ("ollama", settings.ollama_model),
    ]


async def run_one(settings: Settings, provider: str, model: str, log: CallLog) -> str:
    request = LLMRequest(
        model=model,
        system=SYSTEM,
        messages=[Message(role="user", parts=[TextPart(text=SYNTHETIC_INVOICE)])],
        max_tokens=1024,
    )
    try:
        llm = build_llm(settings, provider, log=log)
        response = await llm.complete(request, SmokeExtraction)
    except LLMError as error:
        return f"{provider:10} {model:20} ERROR {error}"
    r = response.record
    out = response.output
    return (
        f"{provider:10} {model:20} in={r.input_tokens:5d} cached={r.cached_input_tokens:4d} "
        f"out={r.output_tokens:4d} cost=${r.cost_usd:.5f} {r.latency_ms:6d}ms  "
        f"number={out.number.value!r} total={out.grand_total.value} "
        f"conf={min(out.number.confidence, out.grand_total.confidence):.2f}"
    )


async def main() -> int:
    settings = get_settings()
    log = CallLog()
    print(f"{'provider':10} {'model':20} tokens / cost / latency / extracted")
    for provider, model in candidates(settings):
        print(await run_one(settings, provider, model, log), flush=True)
    print(f"\ntotal cost: ${log.total_cost():.5f} over {len(log.records)} successful calls")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
