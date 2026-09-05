"""Replay adapter: serves stored responses so tests and CI run with no keys and no spend.

Lookup key is ``LLMRequest.fingerprint(schema)``: change the prompt, the model, the schema
or an image and the old fixture no longer matches, which is the point. With an ``inner``
adapter and a missing fixture it records one real call and saves it; without one it raises.
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, ClassVar

from pydantic import BaseModel, ConfigDict

from doc_intel.llm.base import LLM
from doc_intel.llm.errors import FixtureMissingError
from doc_intel.llm.log import CallLog
from doc_intel.llm.pricing import Pricing
from doc_intel.llm.types import ImagePart, LLMRequest, RawCompletion, TextPart


class RequestSummary(BaseModel):
    """Human-readable context stored next to the response so a reviewer can tell fixtures apart."""

    model_config = ConfigDict(extra="forbid")

    model: str
    system_preview: str
    user_text_preview: str
    image_count: int


class Fixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    recorded_at: datetime
    provider: str
    request: RequestSummary
    response: RawCompletion


class RecordedLLM(LLM):
    provider: ClassVar[str] = "recorded"

    def __init__(
        self,
        fixtures_dir: Path,
        inner: LLM | None = None,
        priced_as: str | None = None,
        pricing: Pricing | None = None,
        log: CallLog | None = None,
    ) -> None:
        super().__init__(pricing, log)
        self.fixtures_dir = fixtures_dir
        self.inner = inner
        self.priced_as = priced_as or (inner.provider if inner else "recorded")

    def path_for(self, key: str) -> Path:
        return self.fixtures_dir / f"{key[:16]}.json"

    async def _complete_raw(self, request: LLMRequest, schema: dict[str, Any]) -> RawCompletion:
        key = request.fingerprint(schema)
        path = self.path_for(key)
        if path.exists():
            return Fixture.model_validate_json(path.read_text()).response
        if self.inner is None:
            raise FixtureMissingError(key, path)

        raw = await self.inner._complete_raw(request, schema)
        fixture = Fixture(
            key=key,
            recorded_at=datetime.now(UTC),
            provider=self.inner.provider,
            request=_summarize(request),
            response=raw,
        )
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        path.write_text(fixture.model_dump_json(indent=2) + "\n")
        return raw

    def _fixture_key(self, request: LLMRequest, schema: dict[str, Any]) -> str | None:
        return request.fingerprint(schema)

    def _pricing_provider(self) -> str:
        return self.priced_as


def _summarize(request: LLMRequest) -> RequestSummary:
    texts = [p.text for m in request.messages for p in m.parts if isinstance(p, TextPart)]
    images = sum(1 for m in request.messages for p in m.parts if isinstance(p, ImagePart))
    return RequestSummary(
        model=request.model,
        system_preview=request.system[:120],
        user_text_preview=" ".join(texts)[:200],
        image_count=images,
    )
