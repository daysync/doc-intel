"""A deterministic bag-of-words embedder: overlapping words give overlapping vectors."""

import hashlib
import math
import re
from typing import ClassVar

from doc_intel.llm.embeddings import Embedder, RawEmbeddings

DIMS = 768


class HashEmbedder(Embedder):
    provider: ClassVar[str] = "fake"

    def __init__(self) -> None:
        super().__init__("hash-embed", DIMS)

    async def _embed_raw(self, texts: list[str]) -> RawEmbeddings:
        return RawEmbeddings(
            vectors=[_vector(t) for t in texts], input_tokens=sum(len(t.split()) for t in texts)
        )


def _vector(text: str) -> list[float]:
    vector = [0.0] * DIMS
    for token in re.findall(r"\w+", text.casefold()):
        slot = int(hashlib.md5(token.encode()).hexdigest(), 16) % DIMS
        vector[slot] += 1.0
    norm = math.sqrt(sum(x * x for x in vector)) or 1.0
    return [x / norm for x in vector]
