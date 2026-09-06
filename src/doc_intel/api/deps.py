"""Request-scoped dependencies. Routes ask for these instead of touching globals."""

from fastapi import HTTPException, Request, status

from doc_intel.api.jobs import JobStore
from doc_intel.api.processing import DocumentIndexer, DocumentProcessor


def get_store(request: Request) -> JobStore:
    store: JobStore = request.app.state.jobs
    return store


def get_processor(request: Request) -> DocumentProcessor:
    processor: DocumentProcessor | None = request.app.state.processor
    if processor is None:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "no pipeline configured")
    return processor


def get_indexer(request: Request) -> DocumentIndexer | None:
    indexer: DocumentIndexer | None = request.app.state.indexer
    return indexer
