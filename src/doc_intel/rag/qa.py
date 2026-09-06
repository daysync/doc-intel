"""Question answering end to end: scope, retrieve, optionally rerank, answer, cite."""

from dataclasses import dataclass
from decimal import Decimal

from pydantic import BaseModel, ConfigDict

from doc_intel.rag.answer import Answerer, AnswerResult, VerifiedCitation
from doc_intel.rag.index import Hit, Retriever
from doc_intel.rag.rerank import Reranker
from doc_intel.rag.scope import resolve_scope


class RagConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    k: int = 5
    candidates: int = 20
    scope_by_number: bool = True
    rerank: bool = False


@dataclass
class QaResult:
    answer: str
    citations: list[VerifiedCitation]
    not_in_documents: bool
    supported: bool
    confidence: float
    scope: list[str] | None
    hits: list[Hit]
    cost_usd: Decimal


class QuestionAnswering:
    def __init__(
        self,
        retriever: Retriever,
        answerer: Answerer,
        reranker: Reranker | None = None,
        config: RagConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.answerer = answerer
        self.reranker = reranker
        self.config = config or RagConfig()

    async def ask(self, question: str, k: int | None = None) -> QaResult:
        k = k or self.config.k
        scope = (
            await resolve_scope(question, self.retriever) if self.config.scope_by_number else None
        )
        candidates = self.config.candidates if self.reranker else k
        hits = await self.retriever.search(question, k=candidates, document_ids=scope)
        cost = Decimal(0)
        if self.reranker and hits:
            before = self.reranker.llm.log.total_cost()
            hits = await self.reranker.rerank(question, hits, k)
            cost += self.reranker.llm.log.total_cost() - before
        result: AnswerResult = await self.answerer.answer(question, hits[:k])
        return QaResult(
            answer=result.answer,
            citations=result.citations,
            not_in_documents=result.not_in_documents,
            supported=result.supported,
            confidence=result.confidence,
            scope=scope,
            hits=hits[:k],
            cost_usd=cost + result.cost_usd,
        )
