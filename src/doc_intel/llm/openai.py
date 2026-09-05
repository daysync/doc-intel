"""OpenAI adapter via the official SDK, Responses API with strict JSON schema output."""

import base64
from typing import Any, ClassVar

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    ResponseInputImageParam,
    ResponseInputMessageContentListParam,
    ResponseInputParam,
    ResponseInputTextParam,
    ResponseTextConfigParam,
)

from doc_intel.llm.base import LLM
from doc_intel.llm.errors import ProviderError
from doc_intel.llm.log import CallLog
from doc_intel.llm.pricing import Pricing
from doc_intel.llm.types import ImagePart, LLMRequest, Message, RawCompletion


def to_openai_input(messages: list[Message]) -> ResponseInputParam:
    """Pure mapping from our neutral messages to Responses API input items."""
    result: ResponseInputParam = []
    for message in messages:
        content: ResponseInputMessageContentListParam = []
        for part in message.parts:
            if isinstance(part, ImagePart):
                data = base64.standard_b64encode(part.data).decode("ascii")
                image: ResponseInputImageParam = {
                    "type": "input_image",
                    "detail": "auto",
                    "image_url": f"data:{part.mime};base64,{data}",
                }
                content.append(image)
            else:
                text: ResponseInputTextParam = {"type": "input_text", "text": part.text}
                content.append(text)
        item: EasyInputMessageParam = {"type": "message", "role": message.role, "content": content}
        result.append(item)
    return result


def json_schema_format(schema: dict[str, Any]) -> ResponseTextConfigParam:
    return {"format": {"type": "json_schema", "name": "output", "schema": schema, "strict": True}}


class OpenAILLM(LLM):
    provider: ClassVar[str] = "openai"

    def __init__(
        self,
        api_key: str | None = None,
        client: openai.AsyncOpenAI | None = None,
        pricing: Pricing | None = None,
        log: CallLog | None = None,
    ) -> None:
        super().__init__(pricing, log)
        self._client = client or openai.AsyncOpenAI(api_key=api_key)

    async def _complete_raw(self, request: LLMRequest, schema: dict[str, Any]) -> RawCompletion:
        try:
            response = await self._client.responses.create(
                model=request.model,
                instructions=request.system,
                input=to_openai_input(request.messages),
                max_output_tokens=request.max_tokens,
                temperature=request.temperature,
                text=json_schema_format(schema),
                store=False,
            )
        except openai.APIStatusError as error:
            raise ProviderError(self.provider, error.message, error.status_code) from error
        except openai.APIConnectionError as error:
            raise ProviderError(self.provider, str(error)) from error

        if response.status == "incomplete":
            reason = response.incomplete_details.reason if response.incomplete_details else "?"
            raise ProviderError(self.provider, f"incomplete response: {reason}")

        usage = response.usage
        if usage is None:
            raise ProviderError(self.provider, "response carried no usage")
        cached = usage.input_tokens_details.cached_tokens if usage.input_tokens_details else 0
        return RawCompletion(
            text=response.output_text,
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cached_input_tokens=cached,
            request_id=response._request_id,
        )
