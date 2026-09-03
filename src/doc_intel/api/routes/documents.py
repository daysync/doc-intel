from typing import Annotated

from fastapi import APIRouter, Depends

from doc_intel.api.deps import get_store
from doc_intel.api.jobs import InMemoryJobStore, Job
from doc_intel.api.schemas import DocumentOut, DocumentsResponse, IssueOut, IssuesResponse

router = APIRouter()


def _to_document(job: Job) -> DocumentOut:
    result = job.result
    return DocumentOut(
        id=job.id,
        filename=job.filename,
        status=job.status,
        created_at=job.created_at,
        invoice=result.invoice if result else None,
        issues=result.issues if result else [],
        cost_usd=result.cost_usd if result else None,
        error=job.error,
    )


@router.get("/documents")
def list_documents(store: Annotated[InMemoryJobStore, Depends(get_store)]) -> DocumentsResponse:
    return DocumentsResponse(documents=[_to_document(job) for job in store.list()])


@router.get("/issues")
def list_issues(store: Annotated[InMemoryJobStore, Depends(get_store)]) -> IssuesResponse:
    """Every validation issue across processed documents, flattened."""
    issues = [
        IssueOut(document_id=job.id, issue=issue)
        for job in store.list()
        if job.result
        for issue in job.result.issues
    ]
    return IssuesResponse(issues=issues)
