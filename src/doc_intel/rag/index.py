"""Write chunks with embeddings into Postgres; hybrid retrieval out of it.

Two rankings per query: cosine similarity over pgvector (meaning) and Postgres full-text
rank (exact tokens such as invoice numbers). Reciprocal rank fusion merges them without
tuning weights: score = sum over lists of 1 / (k + rank).
"""

import re
from dataclasses import dataclass

from pgvector import Vector
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

from doc_intel.llm.embeddings import Embedder
from doc_intel.models import ProcessResult
from doc_intel.rag.chunking import Chunk, chunk_document

RRF_K = 60


@dataclass
class Hit:
    chunk_id: int
    document_id: str
    page: int
    kind: str
    field_path: str | None
    text: str
    score: float
    vector_rank: int | None = None
    text_rank: int | None = None


class Indexer:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder, batch_size: int = 32) -> None:
        self.pool = pool
        self.embedder = embedder
        self.batch_size = batch_size

    async def index(self, result: ProcessResult) -> list[Chunk]:
        chunks = chunk_document(result)
        vectors: list[list[float]] = []
        for start in range(0, len(chunks), self.batch_size):
            vectors += await self.embedder.embed(
                [c.text for c in chunks[start : start + self.batch_size]]
            )
        async with self.pool.connection() as connection:
            await connection.execute(
                "DELETE FROM chunks WHERE document_id = %s", (result.document_id,)
            )
            async with connection.cursor() as cursor:
                await cursor.executemany(
                    "INSERT INTO chunks "
                    "(document_id, page, kind, field_path, text, metadata, embedding) "
                    "VALUES (%s, %s, %s, %s, %s, %s, %s)",
                    [
                        (
                            c.document_id,
                            c.page,
                            c.kind,
                            c.field_path,
                            c.text,
                            Jsonb(c.metadata),
                            Vector(v),
                        )
                        for c, v in zip(chunks, vectors, strict=True)
                    ],
                )
            await connection.commit()
        return chunks


class Retriever:
    def __init__(self, pool: AsyncConnectionPool, embedder: Embedder, candidates: int = 20) -> None:
        self.pool = pool
        self.embedder = embedder
        self.candidates = candidates

    async def search(
        self, query: str, k: int = 5, document_ids: list[str] | None = None
    ) -> list[Hit]:
        """Fused top-k. ``document_ids`` restricts both rankings to those documents."""
        (vector,) = await self.embedder.embed([query])
        scope_sql = " AND document_id = ANY(%s)" if document_ids else ""
        scope_params: tuple[object, ...] = (document_ids,) if document_ids else ()
        async with (
            self.pool.connection() as connection,
            connection.cursor(row_factory=dict_row) as cursor,
        ):
            await cursor.execute(
                "SELECT id, document_id, page, kind, field_path, text FROM chunks "
                f"WHERE TRUE{scope_sql} ORDER BY embedding <=> %s LIMIT %s",
                (*scope_params, Vector(vector), self.candidates),
            )
            by_vector = await cursor.fetchall()
            await cursor.execute(
                "SELECT id, document_id, page, kind, field_path, text, ts_rank(tsv, q) AS rank "
                f"FROM chunks, websearch_to_tsquery('simple', %s) q WHERE tsv @@ q{scope_sql} "
                "ORDER BY rank DESC LIMIT %s",
                (query, *scope_params, self.candidates),
            )
            by_text = await cursor.fetchall()
        return fuse(by_vector, by_text, k)

    async def documents_with_number(self, number: str) -> list[str]:
        """Documents whose chunks carry this invoice number (separators and case ignored)."""
        async with self.pool.connection() as connection, connection.cursor() as cursor:
            await cursor.execute(
                "SELECT DISTINCT document_id FROM chunks "
                "WHERE regexp_replace(upper(metadata->>'number'), '[^A-Z0-9]', '', 'g') = %s",
                (normalise_number(number),),
            )
            rows = await cursor.fetchall()
        return [str(row[0]) for row in rows]


def normalise_number(number: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", number.upper())


def fuse(by_vector: list[dict[str, object]], by_text: list[dict[str, object]], k: int) -> list[Hit]:
    hits: dict[int, Hit] = {}
    for rank, row in enumerate(by_vector, start=1):
        hit = hits.setdefault(int(row["id"]), _hit(row))  # type: ignore[call-overload]
        hit.vector_rank = rank
        hit.score += 1 / (RRF_K + rank)
    for rank, row in enumerate(by_text, start=1):
        hit = hits.setdefault(int(row["id"]), _hit(row))  # type: ignore[call-overload]
        hit.text_rank = rank
        hit.score += 1 / (RRF_K + rank)
    return sorted(hits.values(), key=lambda h: (-h.score, h.chunk_id))[:k]


def _hit(row: dict[str, object]) -> Hit:
    return Hit(
        chunk_id=int(row["id"]),  # type: ignore[call-overload]
        document_id=str(row["document_id"]),
        page=int(row["page"]),  # type: ignore[call-overload]
        kind=str(row["kind"]),
        field_path=str(row["field_path"]) if row["field_path"] else None,
        text=str(row["text"]),
        score=0.0,
    )
