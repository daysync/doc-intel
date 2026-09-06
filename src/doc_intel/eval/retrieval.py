"""Retrieval quality on its own: recall@k and MRR over questions with a known answer chunk.

Questions are generated from ground truth, so the correct document and chunk kind are
known by construction. Ground-truth invoices are indexed directly (no OCR, no extraction),
which isolates retrieval from upstream errors. Run: ``make eval-retrieval``.
"""

import argparse
import asyncio
import json
import statistics
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from doc_intel.api.settings import get_settings
from doc_intel.dataset.generate import Manifest, load_truth
from doc_intel.db.connection import apply_schema, make_pool
from doc_intel.llm.embeddings import Embedder
from doc_intel.llm.factory import build_embedder
from doc_intel.models import Invoice, ProcessResult, Timings
from doc_intel.pipeline import PipelineConfig
from doc_intel.rag.index import Hit, Indexer, Retriever


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    document_id: str
    kind: str
    language: str


class QuestionResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Question
    rank: int | None
    top_document: str | None


class RetrievalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    embeddings_model: str
    k: int
    results: list[QuestionResult]

    def recall_at_k(self) -> float:
        return sum(1 for r in self.results if r.rank is not None and r.rank <= self.k) / len(
            self.results
        )

    def mrr(self) -> float:
        return statistics.mean(1 / r.rank if r.rank else 0.0 for r in self.results)

    def by_kind(self) -> dict[str, float]:
        kinds = sorted({r.question.kind for r in self.results})
        return {
            kind: sum(
                1
                for r in self.results
                if r.question.kind == kind and r.rank is not None and r.rank <= self.k
            )
            / sum(1 for r in self.results if r.question.kind == kind)
            for kind in kinds
        }

    def by_language(self) -> dict[str, float]:
        langs = sorted({r.question.language for r in self.results})
        return {
            lang: sum(
                1
                for r in self.results
                if r.question.language == lang and r.rank is not None and r.rank <= self.k
            )
            / sum(1 for r in self.results if r.question.language == lang)
            for lang in langs
        }


def questions_for(document_id: str, truth: Invoice) -> list[Question]:
    number = truth.number.value or ""
    language = truth.language.value or "en"
    questions = [
        Question(
            text=f"What is the grand total of invoice {number}?",
            document_id=document_id,
            kind="totals",
            language=language,
        ),
        Question(
            text=f"Who is the supplier on invoice {number}?",
            document_id=document_id,
            kind="header",
            language=language,
        ),
        Question(
            text=f"When is invoice {number} due?",
            document_id=document_id,
            kind="header",
            language=language,
        ),
    ]
    if truth.line_items:
        item = truth.line_items[0]
        questions.append(
            Question(
                text=f"How many {item.description.value} are on invoice {number}?",
                document_id=document_id,
                kind="line_item",
                language=language,
            )
        )
    return questions


def _as_result(document_id: str, truth: Invoice) -> ProcessResult:
    return ProcessResult(
        document_id=document_id,
        invoice=truth,
        issues=[],
        cost_usd=Decimal(0),
        timings=Timings(total_ms=0),
    )


def _rank(hits: list[Hit], question: Question) -> int | None:
    for index, hit in enumerate(hits, start=1):
        if hit.document_id == question.document_id and hit.kind == question.kind:
            return index
    return None


async def run(
    samples: Path, config_path: Path, k: int, embedder: Embedder | None = None
) -> RetrievalReport:
    settings = get_settings()
    config = PipelineConfig.from_yaml(config_path)
    embeddings = config.embeddings
    embedder = embedder or build_embedder(
        settings, embeddings.provider, embeddings.model, embeddings.dimensions
    )
    manifest = Manifest.model_validate_json((samples / "manifest.json").read_text())

    pool = make_pool(settings.database_url)
    await pool.open()
    try:
        await apply_schema(pool)
        async with pool.connection() as connection:
            await connection.execute("DELETE FROM documents WHERE id LIKE 'eval-%'")
            await connection.execute(
                "INSERT INTO documents (id, filename, mime, size_bytes, status) "
                "SELECT 'eval-' || d, d, 'application/pdf', 0, 'done' FROM unnest(%s::text[]) d",
                ([m.id for m in manifest.documents],),
            )
            await connection.commit()
        indexer = Indexer(pool, embedder)
        truths = {f"eval-{m.id}": load_truth(samples / m.id) for m in manifest.documents}
        for document_id, truth in truths.items():
            await indexer.index(_as_result(document_id, truth))

        retriever = Retriever(pool, embedder)
        results = []
        for document_id, truth in truths.items():
            for question in questions_for(document_id, truth):
                hits = await retriever.search(question.text, k=k)
                results.append(
                    QuestionResult(
                        question=question,
                        rank=_rank(hits, question),
                        top_document=hits[0].document_id if hits else None,
                    )
                )
    finally:
        await pool.close()
    return RetrievalReport(
        config=config.name, embeddings_model=embeddings.model, k=k, results=results
    )


def print_report(report: RetrievalReport) -> None:
    print(f"== {report.config} / {report.embeddings_model}: {len(report.results)} questions")
    print(f"  recall@{report.k}   {report.recall_at_k():6.0%}")
    print(f"  MRR        {report.mrr():6.2f}")
    print("  by kind    " + "  ".join(f"{k} {v:.0%}" for k, v in report.by_kind().items()))
    print("  by language" + "  ".join(f" {k} {v:.0%}" for k, v in report.by_language().items()))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retrieval recall@k and MRR over ground-truth questions."
    )
    parser.add_argument("--samples", type=Path, default=Path("data/samples"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    report = asyncio.run(run(args.samples, args.config, args.k))
    out = args.samples / f"eval-retrieval-{report.config}.json"
    out.write_text(report.model_dump_json(indent=2) + "\n")
    print_report(report)
    print(
        json.dumps({"recall_at_k": round(report.recall_at_k(), 3), "mrr": round(report.mrr(), 3)})
    )
    _ = date.today()


if __name__ == "__main__":
    main()
