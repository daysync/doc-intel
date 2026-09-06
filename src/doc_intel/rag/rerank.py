"""Optional listwise rerank of the fused candidates by the LLM: one call, all chunks scored.

Costs one model call per question; measured against fusion alone by the eval.
"""

from pydantic import BaseModel, ConfigDict, Field

from doc_intel.llm.base import LLM
from doc_intel.llm.types import LLMRequest, Message, TextPart
from doc_intel.rag.answer import render_excerpts
from doc_intel.rag.index import Hit

SYSTEM = (
    "You judge how useful each numbered excerpt is for answering the question. Score every "
    "excerpt id given, from 0 (irrelevant) to 1 (contains the answer). Judge only relevance, "
    "do not answer the question."
)


class ChunkScore(BaseModel):
    model_config = ConfigDict(extra="forbid")

    excerpt: int = Field(ge=1, description="The excerpt number as shown in the prompt")
    relevance: float = Field(ge=0.0, le=1.0)


class Scores(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scores: list[ChunkScore]


class Reranker:
    def __init__(self, llm: LLM, model: str, max_tokens: int = 1024) -> None:
        self.llm = llm
        self.model = model
        self.max_tokens = max_tokens

    async def rerank(self, question: str, hits: list[Hit], k: int) -> list[Hit]:
        if len(hits) <= 1:
            return hits[:k]
        request = LLMRequest(
            model=self.model,
            system=SYSTEM,
            messages=[
                Message(
                    role="user",
                    parts=[
                        TextPart(
                            text=f"Excerpts:\n\n{render_excerpts(hits)}\n\nQuestion: {question}"
                        )
                    ],
                )
            ],
            max_tokens=self.max_tokens,
        )
        response = await self.llm.complete(request, Scores)
        relevance = {score.excerpt: score.relevance for score in response.output.scores}
        ordered = sorted(
            enumerate(hits, start=1),
            key=lambda item: (-relevance.get(item[0], 0.0), -item[1].score, item[1].chunk_id),
        )
        return [hit for _, hit in ordered][:k]
