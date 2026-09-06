from decimal import Decimal

from psycopg_pool import AsyncConnectionPool

from doc_intel.api.jobs import JobStatus
from doc_intel.db.store import PostgresJobStore
from doc_intel.models import ProcessResult, Timings
from doc_intel.rag.index import Hit, Indexer, Retriever, fuse
from tests.conftest import requires_postgres
from tests.models.factories import ex, invoice
from tests.rag.fakes import HashEmbedder

pytestmark = requires_postgres


async def _stored(pool: AsyncConnectionPool, number: str, supplier: str) -> ProcessResult:
    store = PostgresJobStore(pool)
    job = await store.create(f"{number}.pdf", "application/pdf", 1)
    inv = invoice()
    inv.number = ex(number)
    inv.supplier.name = ex(supplier)
    result = ProcessResult(
        document_id=job.id,
        invoice=inv,
        issues=[],
        cost_usd=Decimal(0),
        timings=Timings(total_ms=1),
        ocr_pages=[f"{supplier}\nInvoice No {number}\nthank you for your business"],
    )
    await store.set_status(job.id, JobStatus.DONE, result=result)
    return result


async def test_index_then_hybrid_search_finds_the_right_chunks(pool: AsyncConnectionPool) -> None:
    embedder = HashEmbedder()
    a = await _stored(pool, "INV-7001", "Beauty Supplies Ltd")
    b = await _stored(pool, "INV-7002", "Glow Wholesale")
    indexer = Indexer(pool, embedder)
    assert len(await indexer.index(a)) == 5  # header, 2 items, totals, 1 ocr block
    await indexer.index(b)
    await indexer.index(b)  # re-indexing replaces, never duplicates

    retriever = Retriever(pool, embedder)
    hits = await retriever.search("grand total of invoice INV-7002", k=3)
    assert hits[0].document_id == b.document_id
    assert hits[0].kind in ("totals", "header")
    assert all(isinstance(h, Hit) for h in hits)

    hits = await retriever.search("Glow Wholesale supplier", k=3)
    assert hits[0].document_id == b.document_id
    assert hits[0].text_rank is not None  # exact tokens matched by full-text search

    hits = await retriever.search("line total for Shampoo 1L on INV-7001", k=3)
    assert hits[0].document_id == a.document_id


def test_rrf_fusion_prefers_chunks_ranked_by_both_lists() -> None:
    def row(i: int) -> dict[str, object]:
        return {
            "id": i,
            "document_id": "d",
            "page": 1,
            "kind": "ocr",
            "field_path": None,
            "text": str(i),
        }

    hits = fuse([row(1), row(2), row(3)], [row(3), row(4)], k=4)
    assert [h.chunk_id for h in hits] == [3, 1, 2, 4]
    assert hits[0].vector_rank == 3 and hits[0].text_rank == 1
    assert hits[3].vector_rank is None and hits[3].text_rank == 2
