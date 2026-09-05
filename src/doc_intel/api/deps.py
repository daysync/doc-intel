"""Request-scoped dependencies. Routes ask for these instead of touching globals."""

from fastapi import HTTPException, Request, status

from doc_intel.api.jobs import InMemoryJobStore
from doc_intel.api.processing import DocumentProcessor


def get_store(request: Request) -> InMemoryJobStore:
    store: InMemoryJobStore = request.app.state.jobs
    return store


def get_processor(request: Request) -> DocumentProcessor:
    processor: DocumentProcessor | None = request.app.state.processor
    if processor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no pipeline configured")
    return processor
