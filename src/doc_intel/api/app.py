"""Application factory.

``create_app()`` builds a fresh app with its own job store, so tests never share state and
``make api`` (``uvicorn doc_intel.api.app:app --factory``) uses the same code path. The
pipeline is injected: the real one from ``configs/default.yaml`` in production, a fake in tests.
"""

from fastapi import FastAPI

from doc_intel import __version__
from doc_intel.api.jobs import InMemoryJobStore
from doc_intel.api.processing import DocumentProcessor
from doc_intel.api.routes import ask, documents, health, ingest, metrics
from doc_intel.api.settings import get_settings


def create_app(processor: DocumentProcessor | None = None) -> FastAPI:
    application = FastAPI(
        title="doc-intel",
        version=__version__,
        description="Document intelligence for supplier invoices and contracts.",
    )
    application.state.jobs = InMemoryJobStore()
    application.state.processor = processor
    for module in (health, ingest, documents, ask, metrics):
        application.include_router(module.router)
    return application


def app() -> FastAPI:
    """Entry point for ``uvicorn --factory``: the real pipeline from the configured YAML."""
    from doc_intel.pipeline import Pipeline

    settings = get_settings()
    return create_app(Pipeline.from_config(settings.pipeline_config, settings))
