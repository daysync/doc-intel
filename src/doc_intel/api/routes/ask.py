from decimal import Decimal

from fastapi import APIRouter

from doc_intel.api.schemas import AskRequest, AskResponse

router = APIRouter()

NOT_IN_DOCUMENTS = "Not in the documents."


@router.post("/ask")
def ask(body: AskRequest) -> AskResponse:
    """Answer a question over the corpus with citations. Retrieval arrives in Stage 3.

    Until then every question gets the honest answer, which is also the answer the real
    pipeline must be able to give: the documents do not contain it.
    """
    return AskResponse(answer=NOT_IN_DOCUMENTS, citations=[], cost_usd=Decimal(0))
