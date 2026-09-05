"""Replays the recorded smoke fixtures through the real adapter path, no network, no keys.

Every fixture under tests/fixtures/llm was produced by ``LLM_RECORD=1 make llm-smoke``.
This test proves the stored answers still validate against SmokeExtraction, so a schema
or prompt change that breaks a provider shows up here before it costs money.
"""

from decimal import Decimal
from pathlib import Path

import pytest

from doc_intel.llm.recorded import Fixture, RecordedLLM
from doc_intel.llm.smoke import SYNTHETIC_INVOICE, SYSTEM, SmokeExtraction
from doc_intel.llm.types import LLMRequest, Message, TextPart

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm"
RECORDED = sorted(FIXTURES.glob("*.json"))


@pytest.mark.parametrize("path", RECORDED, ids=[p.stem for p in RECORDED])
async def test_recorded_smoke_answer_still_validates(path: Path) -> None:
    fixture = Fixture.model_validate_json(path.read_text())
    request = LLMRequest(
        model=fixture.request.model,
        system=SYSTEM,
        messages=[Message(role="user", parts=[TextPart(text=SYNTHETIC_INVOICE)])],
        max_tokens=1024,
    )
    llm = RecordedLLM(FIXTURES, inner=None, priced_as=fixture.provider)
    response = await llm.complete(request, SmokeExtraction)

    assert response.record.fixture_key == fixture.key
    assert response.output.number.value == "INV-2026-0917"
    assert response.output.grand_total.value == Decimal("78.60")
    assert response.output.currency.value == "EUR"


def test_at_least_one_fixture_is_recorded() -> None:
    assert RECORDED, "run LLM_RECORD=1 make llm-smoke to record fixtures"
