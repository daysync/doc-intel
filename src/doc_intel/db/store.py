"""PostgresJobStore: the same four methods as the in-memory store, backed by the documents table."""

import json
from datetime import UTC, datetime
from uuid import uuid4

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from doc_intel.api.jobs import Job, JobStatus
from doc_intel.models import ProcessResult


class PostgresJobStore:
    def __init__(self, pool: AsyncConnectionPool) -> None:
        self.pool = pool

    async def create(self, filename: str, mime: str, size_bytes: int) -> Job:
        job = Job(
            id=uuid4().hex,
            filename=filename,
            mime=mime,
            size_bytes=size_bytes,
            status=JobStatus.QUEUED,
            created_at=datetime.now(UTC),
        )
        async with self.pool.connection() as connection:
            await connection.execute(
                "INSERT INTO documents (id, filename, mime, size_bytes, status, created_at) "
                "VALUES (%s, %s, %s, %s, %s, %s)",
                (job.id, job.filename, job.mime, job.size_bytes, job.status.value, job.created_at),
            )
            await connection.commit()
        return job

    async def get(self, job_id: str) -> Job | None:
        async with (
            self.pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute("SELECT * FROM documents WHERE id = %s", (job_id,))
            row = await cursor.fetchone()
        return _to_job(row) if row else None

    async def list(self) -> list[Job]:
        async with (
            self.pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute("SELECT * FROM documents ORDER BY created_at, id")
            rows = await cursor.fetchall()
        return [_to_job(row) for row in rows]

    async def set_status(
        self,
        job_id: str,
        status: JobStatus,
        result: ProcessResult | None = None,
        error: str | None = None,
    ) -> Job:
        async with self.pool.connection() as connection:
            await connection.execute(
                "UPDATE documents SET status = %s, result = %s::jsonb, error = %s WHERE id = %s",
                (status.value, result.model_dump_json() if result else None, error, job_id),
            )
            await connection.commit()
        job = await self.get(job_id)
        if job is None:
            raise KeyError(job_id)
        return job

    async def clear(self) -> None:
        async with self.pool.connection() as connection:
            await connection.execute("DELETE FROM documents")
            await connection.commit()


def _to_job(row: dict[str, object]) -> Job:
    result = row["result"]
    if isinstance(result, str):
        result = json.loads(result)
    return Job(
        id=str(row["id"]),
        filename=str(row["filename"]),
        mime=str(row["mime"]),
        size_bytes=int(row["size_bytes"]),  # type: ignore[call-overload]
        status=JobStatus(str(row["status"])),
        created_at=row["created_at"],  # type: ignore[arg-type]
        result=ProcessResult.model_validate(result) if result else None,
        error=str(row["error"]) if row["error"] else None,
    )
