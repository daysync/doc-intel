from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status

from doc_intel.api.deps import get_store
from doc_intel.api.jobs import InMemoryJobStore
from doc_intel.api.schemas import IngestResponse

router = APIRouter()

ACCEPTED_MIME_TYPES = frozenset({"application/pdf", "image/jpeg", "image/png", "image/heic"})


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    file: UploadFile,
    store: Annotated[InMemoryJobStore, Depends(get_store)],
) -> IngestResponse:
    """Accept one document and queue it. Processing arrives in Stage 2."""
    mime = file.content_type or ""
    if mime not in ACCEPTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type {mime!r}; accepted: {sorted(ACCEPTED_MIME_TYPES)}",
        )
    payload = await file.read()
    job = store.create(filename=file.filename or "upload", mime=mime, size_bytes=len(payload))
    return IngestResponse(job_id=job.id, status=job.status)
