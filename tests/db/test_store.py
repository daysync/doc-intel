from decimal import Decimal

from psycopg_pool import AsyncConnectionPool

from doc_intel.api.jobs import JobStatus
from doc_intel.db.store import PostgresJobStore
from doc_intel.models import ProcessResult, Timings
from tests.conftest import requires_postgres
from tests.models.factories import invoice

pytestmark = requires_postgres


async def test_create_get_list_and_status_survive_the_round_trip(pool: AsyncConnectionPool) -> None:
    store = PostgresJobStore(pool)
    a = await store.create("a.pdf", "application/pdf", 10)
    b = await store.create("b.jpg", "image/jpeg", 20)
    assert await store.get(a.id) == a
    assert await store.get("missing") is None
    assert [job.id for job in await store.list()] == [a.id, b.id]

    result = ProcessResult(
        document_id=a.id,
        invoice=invoice(),
        issues=[],
        cost_usd=Decimal("0.0042"),
        timings=Timings(total_ms=5),
        ocr_engine="tesseract",
        model="fake",
    )
    done = await store.set_status(a.id, JobStatus.DONE, result=result)
    assert done.status is JobStatus.DONE
    assert done.result is not None and done.result.invoice.number.value == "INV-1042"
    assert done.result.cost_usd == Decimal("0.0042")

    failed = await store.set_status(b.id, JobStatus.FAILED, error="boom")
    assert failed.error == "boom" and failed.result is None
