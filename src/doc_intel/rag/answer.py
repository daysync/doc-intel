"""Answer a question from retrieved chunks, with citations that are checked, not trusted.

The model must cite chunk ids and quote the text it relied on. After the call, every quote
is looked up in its chunk; a citation whose quote is not there is dropped, and an answer
left with no surviving citation is marked unsupported. "Not in the documents" is a flag in
the schema, so declining is a first-class, measurable answer rather than a guess.
"""

import re
from dataclasses import dataclass, field
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from doc_intel.llm.base import LLM
from doc_intel.llm.types import CallRecord, LLMRequest, Message, TextPart
from doc_intel.rag.index import Hit

NOT_IN_DOCUMENTS = "Not in the documents."

SYSTEM = """You answer questions about a set of supplier invoices using only the numbered
excerpts provided. Rules:
- Use only the excerpts. Do not use outside knowledge and do not guess.
- Cite every fact with the excerpt id it comes from and a short verbatim quote from that
  excerpt. Quotes must be copied exactly, including numbers and formatting.
- If the excerpts do not contain the answer, set not_in_documents to true, give no
  citations, and answer exactly "Not in the documents."
- Keep the answer to one or two sentences. Give amounts with their currency when known.
- confidence is 0 to 1: how sure you are that the excerpts support the answer."""


class Citation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    excerpt: int = Field(ge=1, description="The excerpt number as shown in the prompt")
    quote: str


class Answer(BaseModel):
    model_config = ConfigDict(extra="forbid")

    answer: str
    citations: list[Citation]
    not_in_documents: bool
    confidence: float = Field(ge=0.0, le=1.0)


class VerifiedCitation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    chunk_id: int
    document_id: str
    page: int
    kind: str
    snippet: str


@dataclass
class AnswerResult:
    answer: str
    citations: list[VerifiedCitation]
    not_in_documents: bool
    supported: bool
    confidence: float
    dropped_citations: int = 0
    records: list[CallRecord] = field(default_factory=list)

    @property
    def cost_usd(self) -> Decimal:
        return sum((r.cost_usd for r in self.records), Decimal(0))


def render_excerpts(hits: list[Hit]) -> str:
    """Excerpts are numbered by position, 1..N, never by database id.

    Positional numbers keep the prompt identical across re-indexing, so recorded fixtures
    stay valid and the model sees small stable labels. ``verify_citations`` maps them back.
    """
    return "\n\n".join(
        f"[{index}] document {hit.document_id}, page {hit.page}, {hit.kind}:\n{hit.text}"
        for index, hit in enumerate(hits, start=1)
    )


def _norm(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().casefold()


def verify_citations(answer: Answer, hits: list[Hit]) -> tuple[list[VerifiedCitation], int]:
    """Keep citations whose quote appears in the cited excerpt; count the rest as dropped.

    ``citation.excerpt`` is the 1-based position in the prompt; it maps back to the hit.
    """
    kept: list[VerifiedCitation] = []
    dropped = 0
    for citation in answer.citations:
        hit = hits[citation.excerpt - 1] if 0 < citation.excerpt <= len(hits) else None
        if (
            hit is None
            or not citation.quote.strip()
            or _norm(citation.quote) not in _norm(hit.text)
        ):
            dropped += 1
            continue
        kept.append(
            VerifiedCitation(
                chunk_id=hit.chunk_id,
                document_id=hit.document_id,
                page=hit.page,
                kind=hit.kind,
                snippet=citation.quote.strip(),
            )
        )
    return kept, dropped


class Answerer:
    def __init__(self, llm: LLM, model: str, max_tokens: int = 1024) -> None:
        self.llm = llm
        self.model = model
        self.max_tokens = max_tokens

    def build_request(self, question: str, hits: list[Hit]) -> LLMRequest:
        text = f"Excerpts:\n\n{render_excerpts(hits)}\n\nQuestion: {question}"
        return LLMRequest(
            model=self.model,
            system=SYSTEM,
            messages=[Message(role="user", parts=[TextPart(text=text)])],
            max_tokens=self.max_tokens,
        )

    async def answer(self, question: str, hits: list[Hit]) -> AnswerResult:
        if not hits:
            return AnswerResult(NOT_IN_DOCUMENTS, [], True, True, 1.0)
        response = await self.llm.complete(self.build_request(question, hits), Answer)
        parsed = response.output
        if parsed.not_in_documents:
            return AnswerResult(
                NOT_IN_DOCUMENTS, [], True, True, parsed.confidence, records=[response.record]
            )
        citations, dropped = verify_citations(parsed, hits)
        return AnswerResult(
            answer=parsed.answer,
            citations=citations,
            not_in_documents=False,
            supported=bool(citations),
            confidence=parsed.confidence if citations else min(parsed.confidence, 0.3),
            dropped_citations=dropped,
            records=[response.record],
        )
