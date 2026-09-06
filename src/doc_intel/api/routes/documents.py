from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from doc_intel.api.deps import get_processor, get_store
from doc_intel.api.jobs import Job, JobStatus, JobStore
from doc_intel.api.processing import DocumentProcessor
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
        confidence=result.confidence if result else None,
        error=job.error,
    )


@router.get("/documents")
async def list_documents(store: Annotated[JobStore, Depends(get_store)]) -> DocumentsResponse:
    return DocumentsResponse(documents=[_to_document(job) for job in await store.list()])


@router.get("/documents/{job_id}")
async def get_document(job_id: str, store: Annotated[JobStore, Depends(get_store)]) -> DocumentOut:
    job = await store.get(job_id)
    if job is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "no such document")
    return _to_document(job)


@router.get("/issues")
async def list_issues(
    store: Annotated[JobStore, Depends(get_store)],
    processor: Annotated[DocumentProcessor, Depends(get_processor)],
) -> IssuesResponse:
    """Per-document issues plus cross-document ones (duplicates, supplier mismatch)."""
    jobs = await store.list()
    done = {job.id: job.result for job in jobs if job.status is JobStatus.DONE and job.result}
    issues = [
        IssueOut(document_id=doc_id, issue=issue)
        for doc_id, result in done.items()
        for issue in result.issues
    ]
    for doc_id, found in processor.cross_check(done).items():
        issues.extend(IssueOut(document_id=doc_id, issue=issue) for issue in found)
    return IssuesResponse(issues=issues)
