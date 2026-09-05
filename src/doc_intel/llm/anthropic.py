"""Anthropic adapter via the official SDK.

Structured output uses ``output_config.format`` with our JSON schema, so the model is
constrained to the schema rather than merely asked for JSON. ``temperature`` is never sent:
current Claude models reject sampling parameters.
"""

import base64
from typing import Any, ClassVar

import anthropic
from anthropic.types import ImageBlockParam, MessageParam, TextBlockParam

from doc_intel.llm.base import LLM
from doc_intel.llm.errors import ProviderError
from doc_intel.llm.log import CallLog
from doc_intel.llm.pricing import Pricing
from doc_intel.llm.types import ImagePart, LLMRequest, Message, RawCompletion


def to_anthropic_messages(messages: list[Message]) -> list[MessageParam]:
    """Pure mapping from our neutral messages to the SDK's shape. Unit-tested without network."""
    result: list[MessageParam] = []
    for message in messages:
        blocks: list[TextBlockParam | ImageBlockParam] = []
        for part in message.parts:
            if isinstance(part, ImagePart):
                blocks.append(
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": part.mime,
                            "data": base64.standard_b64encode(part.data).decode("ascii"),
                        },
                    }
                )
            else:
                blocks.append({"type": "text", "text": part.text})
        result.append({"role": message.role, "content": blocks})
    return result


class AnthropicLLM(LLM):
    provider: ClassVar[str] = "anthropic"

    def __init__(
        self,
        api_key: str | None = None,
        client: anthropic.AsyncAnthropic | None = None,
        pricing: Pricing | None = None,
        log: CallLog | None = None,
    ) -> None:
        super().__init__(pricing, log)
        self._client = client or anthropic.AsyncAnthropic(api_key=api_key)

    async def _complete_raw(self, request: LLMRequest, schema: dict[str, Any]) -> RawCompletion:
        try:
            response = await self._client.messages.create(
                model=request.model,
                max_tokens=request.max_tokens,
                system=request.system,
                messages=to_anthropic_messages(request.messages),
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        except anthropic.APIStatusError as error:
            raise ProviderError(self.provider, error.message, error.status_code) from error
        except anthropic.APIConnectionError as error:
            raise ProviderError(self.provider, str(error)) from error

        if response.stop_reason == "refusal":
            raise ProviderError(self.provider, "request refused by safety classifiers")
        if response.stop_reason == "max_tokens":
            raise ProviderError(
                self.provider, f"output truncated at max_tokens={request.max_tokens}"
            )

        text = "".join(block.text for block in response.content if block.type == "text")
        usage = response.usage
        return RawCompletion(
            text=text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=usage.cache_read_input_tokens or 0,
            request_id=response._request_id,
        )
