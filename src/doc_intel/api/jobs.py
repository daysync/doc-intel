"""Ingestion jobs and the store that tracks them.

Two stores implement the same async ``JobStore`` protocol: ``InMemoryJobStore`` for tests
and keyless local runs, ``db.store.PostgresJobStore`` for everything that should survive a
restart. The routes only see the protocol.
"""

from datetime import UTC, datetime
from enum import StrEnum
from typing import Protocol
from uuid import uuid4

from pydantic import BaseModel, ConfigDict

from doc_intel.models import ProcessResult


class JobStatus(StrEnum):
    QUEUED = "queued"
    PROCESSING = "processing"
    DONE = "done"
    FAILED = "failed"


class Job(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    filename: str
    mime: str
    size_bytes: int
    status: JobStatus
    created_at: datetime
    result: ProcessResult | None = None
    error: str | None = None


class JobStore(Protocol):
    async def create(self, filename: str, mime: str, size_bytes: int) -> Job: ...

    async def get(self, job_id: str) -> Job | None: ...

    async def list(self) -> list[Job]: ...

    async def set_status(
        self,
        job_id: str,
        status: JobStatus,
        result: ProcessResult | None = None,
        error: str | None = None,
    ) -> Job: ...


class InMemoryJobStore:
    def __init__(self) -> None:
        self._jobs: dict[str, Job] = {}

    async def create(self, filename: str, mime: str, size_bytes: int) -> Job:
        job = Job(
            id=uuid4().hex,
            filename=filename,
            mime=mime,
            size_bytes=size_bytes,
            status=JobStatus.QUEUED,
            created_at=datetime.now(UTC),
        )
        self._jobs[job.id] = job
        return job

    async def get(self, job_id: str) -> Job | None:
        return self._jobs.get(job_id)

    async def list(self) -> list[Job]:
        return sorted(self._jobs.values(), key=lambda job: job.created_at)

    async def set_status(
        self,
        job_id: str,
        status: JobStatus,
        result: ProcessResult | None = None,
        error: str | None = None,
    ) -> Job:
        job = self._jobs[job_id]
        updated = job.model_copy(update={"status": status, "result": result, "error": error})
        self._jobs[job_id] = updated
        return updated
