"""Application factory.

``create_app()`` builds a fresh app with its own job store, so tests never share state and
``make api`` (``uvicorn doc_intel.api.app:app --factory``) uses the same code path.
"""

from fastapi import FastAPI

from doc_intel import __version__
from doc_intel.api.jobs import InMemoryJobStore
from doc_intel.api.routes import health


def create_app() -> FastAPI:
    application = FastAPI(
        title="doc-intel",
        version=__version__,
        description="Document intelligence for supplier invoices and contracts.",
    )
    application.state.jobs = InMemoryJobStore()
    application.include_router(health.router)
    return application


def app() -> FastAPI:
    """Entry point for ``uvicorn --factory``."""
    return create_app()
