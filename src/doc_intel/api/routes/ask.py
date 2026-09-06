from typing import Annotated, Protocol

from fastapi import APIRouter, Depends, HTTPException, Request, status

from doc_intel.api.schemas import AskRequest, AskResponse, Citation
from doc_intel.rag.qa import QaResult

router = APIRouter()


class QuestionAnswerer(Protocol):
    async def ask(self, question: str, k: int | None = None) -> QaResult: ...


def get_qa(request: Request) -> QuestionAnswerer:
    qa: QuestionAnswerer | None = request.app.state.qa
    if qa is None:
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE, "question answering is not configured"
        )
    return qa


@router.post("/ask")
async def ask(body: AskRequest, qa: Annotated[QuestionAnswerer, Depends(get_qa)]) -> AskResponse:
    """Answer over the indexed documents with verified citations.

    "Not in the documents." is a real answer: the model declined because the excerpts do
    not contain it, and no citation is attached.
    """
    result = await qa.ask(body.question, k=body.k)
    return AskResponse(
        answer=result.answer,
        citations=[
            Citation(
                document_id=c.document_id,
                page=c.page,
                snippet=c.snippet,
                chunk_id=c.chunk_id,
                kind=c.kind,
            )
            for c in result.citations
        ],
        cost_usd=result.cost_usd,
        not_in_documents=result.not_in_documents,
        supported=result.supported,
        confidence=result.confidence,
        scope=result.scope,
    )
