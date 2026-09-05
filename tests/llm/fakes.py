"""A scripted adapter: returns canned text, records what it was asked. Used across llm tests."""

from typing import Any, ClassVar

from doc_intel.llm import LLM, LLMRequest, RawCompletion


class FakeLLM(LLM):
    provider: ClassVar[str] = "fake"

    def __init__(self, text: str, input_tokens: int = 100, output_tokens: int = 20) -> None:
        super().__init__()
        self.text = text
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.calls: list[tuple[LLMRequest, dict[str, Any]]] = []

    async def _complete_raw(self, request: LLMRequest, schema: dict[str, Any]) -> RawCompletion:
        self.calls.append((request, schema))
        return RawCompletion(
            text=self.text,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            request_id="fake-1",
        )


def simple_request(text: str = "Extract the greeting.", model: str = "fake-model") -> LLMRequest:
    return LLMRequest(
        model=model,
        system="You extract structured data.",
        messages=[{"role": "user", "parts": [{"type": "text", "text": text}]}],  # type: ignore[list-item]
    )
