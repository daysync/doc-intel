import json
from pathlib import Path

import pytest
from pydantic import BaseModel, ConfigDict

from doc_intel.llm import FixtureMissingError
from doc_intel.llm.recorded import RecordedLLM
from tests.llm.fakes import FakeLLM, simple_request


class Out(BaseModel):
    model_config = ConfigDict(extra="forbid")
    value: int


async def test_records_on_miss_then_replays_without_inner(tmp_path: Path) -> None:
    inner = FakeLLM('{"value": 7}', input_tokens=50, output_tokens=5)
    recorder = RecordedLLM(tmp_path, inner=inner)
    first = await recorder.complete(simple_request(), Out)
    assert first.output.value == 7
    assert len(inner.calls) == 1
    assert first.record.fixture_key is not None

    files = list(tmp_path.glob("*.json"))
    assert len(files) == 1
    fixture = json.loads(files[0].read_text())
    assert fixture["provider"] == "fake"
    assert fixture["response"]["input_tokens"] == 50
    assert "hello" not in fixture["request"]["user_text_preview"]

    replay = RecordedLLM(tmp_path, inner=None, priced_as="fake")
    second = await replay.complete(simple_request(), Out)
    assert second.output.value == 7
    assert second.record.provider == "recorded"
    assert second.record.fixture_key == first.record.fixture_key
    assert len(inner.calls) == 1


async def test_miss_without_inner_raises(tmp_path: Path) -> None:
    replay = RecordedLLM(tmp_path, inner=None, priced_as="fake")
    with pytest.raises(FixtureMissingError, match="LLM_RECORD=1"):
        await replay.complete(simple_request(), Out)


async def test_changed_prompt_is_a_different_fixture(tmp_path: Path) -> None:
    inner = FakeLLM('{"value": 1}')
    recorder = RecordedLLM(tmp_path, inner=inner)
    await recorder.complete(simple_request("A"), Out)
    await recorder.complete(simple_request("B"), Out)
    await recorder.complete(simple_request("A"), Out)
    assert len(list(tmp_path.glob("*.json"))) == 2
    assert len(inner.calls) == 2


async def test_replay_prices_as_the_source_provider(tmp_path: Path) -> None:
    from decimal import Decimal

    from doc_intel.llm.pricing import ModelPrice, Pricing

    pricing = Pricing(
        table={
            ("acme", "m"): ModelPrice(
                input_per_mtok=Decimal(1_000_000),
                output_per_mtok=Decimal(0),
                cached_input_per_mtok=Decimal(0),
            )
        },
        free_providers=frozenset({"fake"}),
    )
    inner = FakeLLM('{"value": 1}', input_tokens=3, output_tokens=0)
    await RecordedLLM(tmp_path, inner=inner).complete(simple_request(model="m"), Out)
    replay = RecordedLLM(tmp_path, inner=None, priced_as="acme", pricing=pricing)
    response = await replay.complete(simple_request(model="m"), Out)
    assert response.record.cost_usd == Decimal(3)
