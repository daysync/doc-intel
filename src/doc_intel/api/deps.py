"""Request-scoped dependencies. Routes ask for these instead of touching globals."""

from fastapi import Request

from doc_intel.api.jobs import InMemoryJobStore


def get_store(request: Request) -> InMemoryJobStore:
    store: InMemoryJobStore = request.app.state.jobs
    return store
