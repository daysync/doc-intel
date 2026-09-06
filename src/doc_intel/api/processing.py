"""Background processing of ingested documents.

``POST /ingest`` returns 202 immediately; this module runs the pipeline afterwards and
records the outcome on the job. A failure is stored, never raised, so a bad document can
never take the API down.
"""

import logging
from collections.abc import Mapping
from typing import Protocol

from doc_intel.api.jobs import JobStatus, JobStore
from doc_intel.models import ProcessResult, ValidationIssue

logger = logging.getLogger("doc_intel.api")


class DocumentProcessor(Protocol):
    """What the API needs from a pipeline. ``Pipeline`` satisfies it; tests use a fake."""

    async def process(
        self, data: bytes, mime: str, document_id: str | None = None
    ) -> ProcessResult: ...

    def cross_check(
        self, results: Mapping[str, ProcessResult]
    ) -> dict[str, list[ValidationIssue]]: ...


async def process_job(
    store: JobStore, processor: DocumentProcessor, job_id: str, data: bytes, mime: str
) -> None:
    await store.set_status(job_id, JobStatus.PROCESSING)
    try:
        result = await processor.process(data, mime, document_id=job_id)
    except Exception as error:
        logger.exception("job %s failed", job_id)
        await store.set_status(job_id, JobStatus.FAILED, error=f"{type(error).__name__}: {error}")
        return
    await store.set_status(job_id, JobStatus.DONE, result=result)
