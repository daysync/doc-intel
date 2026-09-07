"""Faithfulness and answer relevancy, by the Ragas definitions, over our own interfaces.

* Faithfulness: the answer is split into atomic claims by the judge; the score is the share
  of claims the retrieved contexts support. 1.0 means nothing was made up.
* Answer relevancy: the judge writes N questions the answer would be a good answer to; the
  score is the mean cosine similarity between those and the real question. A vague or
  off-topic answer yields questions unlike the one asked.

The Ragas library computes the same two numbers, but it pins the OpenAI SDK below 3 via
`instructor`; this repo is on OpenAI 3.x, so the metrics live here and the library can be
swapped in when its pin moves. Declines ("Not in the documents.") are scored as faithful
(no claims) and are excluded from relevancy.
"""

import math
from dataclasses import dataclass

from pydantic import BaseModel, ConfigDict

from doc_intel.eval.judge import Judge
from doc_intel.llm.embeddings import Embedder
from doc_intel.llm.types import LLMRequest, Message, TextPart
from doc_intel.rag.answer import NOT_IN_DOCUMENTS


class GeneratedQuestions(BaseModel):
    model_config = ConfigDict(extra="forbid")

    questions: list[str]


@dataclass
class RagasScores:
    faithfulness: float | None
    """None when the answer makes no claims (a decline)."""
    answer_relevancy: float | None
    claims_total: int = 0
    claims_supported: int = 0


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b, strict=True))
    norm = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return dot / norm if norm else 0.0


class RagasStyle:
    def __init__(self, judge: Judge, embedder: Embedder, questions_per_answer: int = 3) -> None:
        self.judge = judge
        self.embedder = embedder
        self.questions_per_answer = questions_per_answer

    async def faithfulness(self, answer: str, contexts: list[str]) -> tuple[float | None, int, int]:
        if answer.strip() == NOT_IN_DOCUMENTS or not contexts:
            return None, 0, 0
        checks = await self.judge.check_claims(answer, contexts)
        total = len(checks.claims)
        supported = sum(1 for c in checks.claims if c.supported)
        return (supported / total if total else None), total, supported

    async def answer_relevancy(self, question: str, answer: str) -> float | None:
        if answer.strip() == NOT_IN_DOCUMENTS:
            return None
        system = (
            f"Write {self.questions_per_answer} different questions that the given answer would "
            "directly answer. Keep them short and specific; do not answer them."
        )
        request = LLMRequest(
            model=self.judge.model,
            system=system,
            messages=[Message(role="user", parts=[TextPart(text=f"Answer: {answer}")])],
            max_tokens=512,
        )
        generated = (await self.judge.llm.complete(request, GeneratedQuestions)).output.questions
        generated = [q for q in generated if q.strip()][: self.questions_per_answer]
        if not generated:
            return 0.0
        vectors = await self.embedder.embed([question, *generated])
        return sum(_cosine(vectors[0], v) for v in vectors[1:]) / len(generated)

    async def score(self, question: str, answer: str, contexts: list[str]) -> RagasScores:
        faith, total, supported = await self.faithfulness(answer, contexts)
        relevancy = await self.answer_relevancy(question, answer)
        return RagasScores(faith, relevancy, total, supported)
