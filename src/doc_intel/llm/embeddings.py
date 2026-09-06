"""Embeddings behind one interface, built like ``LLM``: one abstract hook per adapter.

An embedding turns text into a vector so that similar meanings land close together. The
pipeline never calls a provider directly; it calls ``Embedder.embed()`` and gets vectors plus
a ``CallRecord`` with tokens and cost. Adapters: Ollama (local, free), OpenAI, and a recorded
one for tests. All vectors from one adapter have the same fixed ``dimensions``.
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict

from doc_intel.llm.errors import FixtureMissingError, ProviderError
from doc_intel.llm.log import CallLog
from doc_intel.llm.pricing import Pricing
from doc_intel.llm.types import CallRecord, RawCompletion


@dataclass
class RawEmbeddings:
    vectors: list[list[float]]
    input_tokens: int
    request_id: str | None = None


class Embedder(ABC):
    provider: ClassVar[str]

    def __init__(
        self,
        model: str,
        dimensions: int,
        pricing: Pricing | None = None,
        log: CallLog | None = None,
    ) -> None:
        self.model = model
        self.dimensions = dimensions
        self.pricing = pricing or Pricing()
        self.log = log or CallLog()

    @abstractmethod
    async def _embed_raw(self, texts: list[str]) -> RawEmbeddings:
        """Embed a batch; return one vector per text plus the token count."""

    async def embed(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        started_at = datetime.now(UTC)
        clock = time.perf_counter()
        raw = await self._embed_raw(texts)
        latency_ms = int((time.perf_counter() - clock) * 1000)
        if len(raw.vectors) != len(texts):
            raise ProviderError(
                self.provider, f"asked for {len(texts)} embeddings, got {len(raw.vectors)}"
            )
        for vector in raw.vectors:
            if len(vector) != self.dimensions:
                raise ProviderError(
                    self.provider, f"expected {self.dimensions}-d vectors, got {len(vector)}-d"
                )
        usage = RawCompletion(text="", input_tokens=raw.input_tokens, output_tokens=0)
        self.log.append(
            CallRecord(
                provider=self.provider,
                model=self.model,
                input_tokens=raw.input_tokens,
                output_tokens=0,
                cached_input_tokens=0,
                cost_usd=self.pricing.cost(self._pricing_provider(), self.model, usage),
                latency_ms=latency_ms,
                started_at=started_at,
                request_id=raw.request_id,
                fixture_key=self._fixture_key(texts),
            )
        )
        return raw.vectors

    def _pricing_provider(self) -> str:
        return self.provider

    def _fixture_key(self, texts: list[str]) -> str | None:
        return None

    def fingerprint(self, texts: list[str]) -> str:
        payload = json.dumps(
            {"model": self.model, "texts": texts}, ensure_ascii=True, sort_keys=True
        )
        return hashlib.sha256(payload.encode()).hexdigest()


class OllamaEmbedder(Embedder):
    provider: ClassVar[str] = "ollama"

    def __init__(
        self,
        model: str = "nomic-embed-text",
        dimensions: int = 768,
        host: str | None = None,
        **kw: object,
    ) -> None:
        import ollama

        super().__init__(model, dimensions, **kw)  # type: ignore[arg-type]
        self._client = ollama.AsyncClient(host=host)

    async def _embed_raw(self, texts: list[str]) -> RawEmbeddings:
        import ollama

        try:
            response = await self._client.embed(model=self.model, input=texts)
        except ollama.ResponseError as error:
            raise ProviderError(self.provider, error.error, error.status_code) from error
        except ConnectionError as error:
            raise ProviderError(self.provider, f"cannot reach Ollama: {error}") from error
        return RawEmbeddings(
            vectors=[list(map(float, vector)) for vector in response.embeddings],
            input_tokens=response.prompt_eval_count or 0,
        )


class OpenAIEmbedder(Embedder):
    provider: ClassVar[str] = "openai"

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        dimensions: int = 768,
        api_key: str | None = None,
        **kw: object,
    ) -> None:
        import openai

        super().__init__(model, dimensions, **kw)  # type: ignore[arg-type]
        self._client = openai.AsyncOpenAI(api_key=api_key)

    async def _embed_raw(self, texts: list[str]) -> RawEmbeddings:
        import openai

        try:
            response = await self._client.embeddings.create(
                model=self.model, input=texts, dimensions=self.dimensions
            )
        except openai.APIStatusError as error:
            raise ProviderError(self.provider, error.message, error.status_code) from error
        except openai.APIConnectionError as error:
            raise ProviderError(self.provider, str(error)) from error
        ordered = sorted(response.data, key=lambda item: item.index)
        return RawEmbeddings(
            vectors=[list(item.embedding) for item in ordered],
            input_tokens=response.usage.prompt_tokens,
            request_id=response._request_id,
        )


class EmbeddingFixture(BaseModel):
    model_config = ConfigDict(extra="forbid")

    key: str
    recorded_at: datetime
    provider: str
    model: str
    texts_preview: list[str]
    input_tokens: int
    vectors: list[list[float]]


class RecordedEmbedder(Embedder):
    provider: ClassVar[str] = "recorded"

    def __init__(
        self,
        fixtures_dir: Path,
        inner: Embedder | None = None,
        model: str | None = None,
        dimensions: int | None = None,
        priced_as: str | None = None,
        **kw: object,
    ) -> None:
        if inner is None and (model is None or dimensions is None):
            raise ValueError("a replay-only RecordedEmbedder needs model and dimensions")
        super().__init__(
            model or (inner.model if inner else ""),
            dimensions or (inner.dimensions if inner else 0),
            **kw,  # type: ignore[arg-type]
        )
        self.fixtures_dir = fixtures_dir
        self.inner = inner
        self.priced_as = priced_as or (inner.provider if inner else "recorded")

    def path_for(self, key: str) -> Path:
        return self.fixtures_dir / f"{key[:16]}.json"

    async def _embed_raw(self, texts: list[str]) -> RawEmbeddings:
        key = self.fingerprint(texts)
        path = self.path_for(key)
        if path.exists():
            fixture = EmbeddingFixture.model_validate_json(path.read_text())
            return RawEmbeddings(vectors=fixture.vectors, input_tokens=fixture.input_tokens)
        if self.inner is None:
            raise FixtureMissingError(key, path)
        raw = await self.inner._embed_raw(texts)
        self.fixtures_dir.mkdir(parents=True, exist_ok=True)
        fixture = EmbeddingFixture(
            key=key,
            recorded_at=datetime.now(UTC),
            provider=self.inner.provider,
            model=self.inner.model,
            texts_preview=[t[:80] for t in texts],
            input_tokens=raw.input_tokens,
            vectors=[[round(x, 6) for x in v] for v in raw.vectors],
        )
        path.write_text(fixture.model_dump_json() + "\n")
        return raw

    def _pricing_provider(self) -> str:
        return self.priced_as

    def _fixture_key(self, texts: list[str]) -> str | None:
        return self.fingerprint(texts)


def cost_of(records: list[CallRecord]) -> Decimal:
    return sum((r.cost_usd for r in records), Decimal(0))
