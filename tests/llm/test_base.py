import pytest
from pydantic import BaseModel, ConfigDict

from doc_intel.llm import StructuredOutputError
from tests.llm.fakes import FakeLLM, simple_request


class Greeting(BaseModel):
    model_config = ConfigDict(extra="forbid")
    word: str
    friendly: bool


async def test_complete_validates_into_the_output_model() -> None:
    llm = FakeLLM('{"word": "hello", "friendly": true}')
    response = await llm.complete(simple_request(), Greeting)
    assert response.output == Greeting(word="hello", friendly=True)
    assert response.raw_text.startswith("{")


async def test_adapter_receives_the_json_schema_of_the_output_model() -> None:
    llm = FakeLLM('{"word": "hi", "friendly": false}')
    await llm.complete(simple_request(), Greeting)
    _, schema = llm.calls[0]
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"word", "friendly"}


async def test_invalid_json_raises_structured_output_error_with_raw_text() -> None:
    llm = FakeLLM('{"word": 42}')
    with pytest.raises(StructuredOutputError) as info:
        await llm.complete(simple_request(), Greeting)
    assert info.value.raw_text == '{"word": 42}'


async def test_every_call_appends_a_record_to_the_log() -> None:
    llm = FakeLLM('{"word": "hey", "friendly": true}', input_tokens=321, output_tokens=12)
    response = await llm.complete(simple_request(), Greeting)
    record = response.record
    assert llm.log.records == [record]
    assert record.provider == "fake"
    assert record.model == "fake-model"
    assert (record.input_tokens, record.output_tokens) == (321, 12)
    assert record.cost_usd == 0
    assert record.latency_ms >= 0
    assert record.request_id == "fake-1"
    assert record.fixture_key is None
