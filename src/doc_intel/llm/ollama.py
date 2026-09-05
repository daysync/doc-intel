"""Ollama adapter: local models, zero cost per token.

``format=schema`` makes Ollama constrain generation to the JSON schema. Images are passed
as raw bytes on the message.
"""

from typing import Any, ClassVar

import ollama

from doc_intel.llm.base import LLM
from doc_intel.llm.errors import ProviderError
from doc_intel.llm.log import CallLog
from doc_intel.llm.pricing import Pricing
from doc_intel.llm.types import ImagePart, LLMRequest, Message, RawCompletion


def to_ollama_messages(system: str, messages: list[Message]) -> list[ollama.Message]:
    """Pure mapping. Ollama takes the system prompt as the first message and images per message."""
    result = [ollama.Message(role="system", content=system)]
    for message in messages:
        text = "\n".join(p.text for p in message.parts if not isinstance(p, ImagePart))
        images = [ollama.Image(value=p.data) for p in message.parts if isinstance(p, ImagePart)]
        result.append(ollama.Message(role=message.role, content=text, images=images or None))
    return result


class OllamaLLM(LLM):
    provider: ClassVar[str] = "ollama"

    def __init__(
        self,
        host: str | None = None,
        client: ollama.AsyncClient | None = None,
        pricing: Pricing | None = None,
        log: CallLog | None = None,
    ) -> None:
        super().__init__(pricing, log)
        self._client = client or ollama.AsyncClient(host=host)

    async def _complete_raw(self, request: LLMRequest, schema: dict[str, Any]) -> RawCompletion:
        options: dict[str, Any] = {"num_predict": request.max_tokens}
        if request.temperature is not None:
            options["temperature"] = request.temperature
        try:
            response = await self._client.chat(
                model=request.model,
                messages=to_ollama_messages(request.system, request.messages),
                format=schema,
                options=options,
            )
        except ollama.ResponseError as error:
            raise ProviderError(self.provider, error.error, error.status_code) from error
        except ConnectionError as error:
            raise ProviderError(self.provider, f"cannot reach Ollama: {error}") from error

        return RawCompletion(
            text=response.message.content or "",
            input_tokens=response.prompt_eval_count or 0,
            output_tokens=response.eval_count or 0,
        )
