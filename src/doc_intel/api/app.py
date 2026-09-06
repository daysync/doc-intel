"""Application factory.

``create_app()`` builds an app around whatever store, processor and indexer it is given, so
tests inject fakes and never share state. ``app()``, the uvicorn entry point, wires the real
thing: the pipeline from the configured YAML, Postgres when DATABASE_URL points at one, and
an indexer so every processed document becomes searchable.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from doc_intel import __version__
from doc_intel.api.jobs import InMemoryJobStore, JobStore
from doc_intel.api.processing import DocumentIndexer, DocumentProcessor
from doc_intel.api.routes import ask, documents, health, ingest, metrics
from doc_intel.api.settings import get_settings


def create_app(
    processor: DocumentProcessor | None = None,
    store: JobStore | None = None,
    indexer: DocumentIndexer | None = None,
    qa: object | None = None,
) -> FastAPI:
    application = FastAPI(
        title="doc-intel",
        version=__version__,
        description="Document intelligence for supplier invoices and contracts.",
    )
    application.state.jobs = store or InMemoryJobStore()
    application.state.processor = processor
    application.state.indexer = indexer
    application.state.qa = qa
    for module in (health, ingest, documents, ask, metrics):
        application.include_router(module.router)
    return application


def app() -> FastAPI:
    """Entry point for ``uvicorn --factory``: real pipeline, Postgres store and indexer."""
    from doc_intel.db.connection import apply_schema, make_pool
    from doc_intel.db.store import PostgresJobStore
    from doc_intel.llm.factory import build_embedder
    from doc_intel.pipeline import Pipeline
    from doc_intel.rag.answer import Answerer
    from doc_intel.rag.index import Indexer, Retriever
    from doc_intel.rag.qa import QuestionAnswering
    from doc_intel.rag.rerank import Reranker

    settings = get_settings()
    pipeline = Pipeline.from_config(settings.pipeline_config, settings)
    pool = make_pool(settings.database_url)
    embeddings = pipeline.config.embeddings
    embedder = build_embedder(
        settings, embeddings.provider, embeddings.model, embeddings.dimensions
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await pool.open()
        await apply_schema(pool)
        yield
        await pool.close()

    rag = pipeline.config.rag
    qa = QuestionAnswering(
        Retriever(pool, embedder, candidates=rag.candidates),
        Answerer(pipeline.llm, pipeline.config.llm.model),
        Reranker(pipeline.llm, pipeline.config.llm.model) if rag.rerank else None,
        rag,
    )
    application = create_app(pipeline, PostgresJobStore(pool), Indexer(pool, embedder), qa)
    application.router.lifespan_context = lifespan
    return application
