from typing import Annotated

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, status

from doc_intel.api.deps import get_indexer, get_processor, get_store
from doc_intel.api.jobs import JobStore
from doc_intel.api.processing import DocumentIndexer, DocumentProcessor, process_job
from doc_intel.api.schemas import IngestResponse
from doc_intel.api.security import get_request_settings
from doc_intel.api.settings import Settings
from doc_intel.ocr.image import IMAGE_MIMES, PDF_MIME

router = APIRouter()

ACCEPTED_MIME_TYPES = frozenset({PDF_MIME, *IMAGE_MIMES})


@router.post("/ingest", status_code=status.HTTP_202_ACCEPTED)
async def ingest(
    file: UploadFile,
    background: BackgroundTasks,
    store: Annotated[JobStore, Depends(get_store)],
    processor: Annotated[DocumentProcessor, Depends(get_processor)],
    indexer: Annotated[DocumentIndexer | None, Depends(get_indexer)],
    settings: Annotated[Settings, Depends(get_request_settings)],
) -> IngestResponse:
    """Accept one document, queue it, and process it in the background."""
    mime = file.content_type or ""
    if mime not in ACCEPTED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=f"Unsupported type {mime!r}; accepted: {sorted(ACCEPTED_MIME_TYPES)}",
        )
    limit = settings.max_upload_mb * 1024 * 1024
    payload = await file.read(limit + 1)
    if len(payload) > limit:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=f"file exceeds {settings.max_upload_mb} MB",
        )
    job = await store.create(filename=file.filename or "upload", mime=mime, size_bytes=len(payload))
    background.add_task(process_job, store, processor, job.id, payload, mime, indexer)
    return IngestResponse(job_id=job.id, status=job.status)
