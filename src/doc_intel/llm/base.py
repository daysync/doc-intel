"""The one interface the pipeline talks to.

``LLM.complete()`` is a template method: it owns the workflow (schema, timing, validation,
pricing, logging) and calls the single abstract hook ``_complete_raw()`` that each provider
adapter implements. Adding a provider means one file with one method.
"""

import time
from abc import ABC, abstractmethod
from datetime import UTC, datetime
from typing import Any, ClassVar

from pydantic import BaseModel, ValidationError

from doc_intel.llm.errors import StructuredOutputError
from doc_intel.llm.log import CallLog
from doc_intel.llm.pricing import Pricing
from doc_intel.llm.types import CallRecord, LLMRequest, LLMResponse, RawCompletion


class LLM(ABC):
    provider: ClassVar[str]

    def __init__(self, pricing: Pricing | None = None, log: CallLog | None = None) -> None:
        self.pricing = pricing or Pricing()
        self.log = log or CallLog()

    @abstractmethod
    async def _complete_raw(self, request: LLMRequest, schema: dict[str, Any]) -> RawCompletion:
        """Send one request asking for JSON matching ``schema``; return text and token counts."""

    async def complete[T: BaseModel](self, request: LLMRequest, output: type[T]) -> LLMResponse[T]:
        schema = output.model_json_schema()
        started_at = datetime.now(UTC)
        clock = time.perf_counter()
        raw = await self._complete_raw(request, schema)
        latency_ms = int((time.perf_counter() - clock) * 1000)

        try:
            parsed = output.model_validate_json(raw.text)
        except ValidationError as error:
            raise StructuredOutputError(raw.text, error) from error

        record = CallRecord(
            provider=self.provider,
            model=request.model,
            input_tokens=raw.input_tokens,
            output_tokens=raw.output_tokens,
            cached_input_tokens=raw.cached_input_tokens,
            cost_usd=self.pricing.cost(self.provider, request.model, raw),
            latency_ms=latency_ms,
            started_at=started_at,
            request_id=raw.request_id,
            fixture_key=self._fixture_key(request, schema),
        )
        self.log.append(record)
        return LLMResponse(output=parsed, raw_text=raw.text, record=record)

    def _fixture_key(self, request: LLMRequest, schema: dict[str, Any]) -> str | None:
        """Overridden by the recorded adapter; live adapters have no fixture."""
        return None
