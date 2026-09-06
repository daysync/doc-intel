import os
from pathlib import Path
from typing import ClassVar

import pytest

from doc_intel.llm.embeddings import Embedder, OllamaEmbedder, RawEmbeddings, RecordedEmbedder
from doc_intel.llm.errors import FixtureMissingError, ProviderError

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm" / "embeddings"


class FakeEmbedder(Embedder):
    provider: ClassVar[str] = "fake"

    def __init__(self, dims: int = 4, tokens: int = 7) -> None:
        super().__init__("fake-embed", dims)
        self.tokens = tokens
        self.calls = 0

    async def _embed_raw(self, texts: list[str]) -> RawEmbeddings:
        self.calls += 1
        return RawEmbeddings(
            vectors=[[float(len(t))] * self.dimensions for t in texts], input_tokens=self.tokens
        )


async def test_embed_returns_one_vector_per_text_and_logs_a_record() -> None:
    embedder = FakeEmbedder()
    vectors = await embedder.embed(["ab", "abcd"])
    assert vectors == [[2.0] * 4, [4.0] * 4]
    record = embedder.log.records[0]
    assert (record.provider, record.model, record.input_tokens, record.output_tokens) == (
        "fake",
        "fake-embed",
        7,
        0,
    )
    assert record.cost_usd == 0
    assert await embedder.embed([]) == []


async def test_wrong_dimensions_are_rejected() -> None:
    class Wrong(FakeEmbedder):
        async def _embed_raw(self, texts: list[str]) -> RawEmbeddings:
            return RawEmbeddings(vectors=[[1.0, 2.0]] * len(texts), input_tokens=1)

    with pytest.raises(ProviderError, match="4-d"):
        await Wrong().embed(["x"])


async def test_recorded_embedder_records_then_replays(tmp_path: Path) -> None:
    inner = FakeEmbedder()
    recorder = RecordedEmbedder(tmp_path, inner=inner)
    first = await recorder.embed(["hello", "world"])
    assert inner.calls == 1 and len(list(tmp_path.glob("*.json"))) == 1
    replay = RecordedEmbedder(
        tmp_path, inner=None, model="fake-embed", dimensions=4, priced_as="fake"
    )
    assert await replay.embed(["hello", "world"]) == first
    assert inner.calls == 1
    with pytest.raises(FixtureMissingError):
        await replay.embed(["other"])


def _ollama_recorded() -> RecordedEmbedder:
    if os.environ.get("LLM_RECORD") == "1":
        return RecordedEmbedder(FIXTURES, inner=OllamaEmbedder())
    return RecordedEmbedder(
        FIXTURES, inner=None, model="nomic-embed-text", dimensions=768, priced_as="ollama"
    )


async def test_local_embeddings_place_similar_texts_closer() -> None:
    """Recorded from Ollama nomic-embed-text; replayed in CI."""
    import math

    texts = [
        "Invoice INV-2026-0917 total due 78.60 EUR",
        "Total amount payable: 78.60 euros",
        "Argan shampoo 1L",
    ]
    try:
        a, b, c = await _ollama_recorded().embed(texts)
    except FixtureMissingError as error:
        pytest.skip(f"no fixture yet: {error}")

    def cosine(x: list[float], y: list[float]) -> float:
        dot = sum(i * j for i, j in zip(x, y, strict=True))
        return dot / (math.sqrt(sum(i * i for i in x)) * math.sqrt(sum(j * j for j in y)))

    assert len(a) == 768
    assert cosine(a, b) > cosine(a, c)
