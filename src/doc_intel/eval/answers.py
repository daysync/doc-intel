"""End-to-end question answering over the ground-truth index: ``make eval-answers``.

Measures what the README calls "end to end": the share of questions answered correctly
with verified citations, and the share of unanswerable questions correctly declined.
Also reports scoped retrieval recall so the effect of scope_by_number is visible next to
the unscoped 3a number.
"""

import argparse
import asyncio
import json
import re
import statistics
import time
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from doc_intel.api.settings import get_settings
from doc_intel.dataset.generate import Manifest, load_truth
from doc_intel.db.connection import apply_schema, make_pool
from doc_intel.eval.accuracy import _NOT_LETTER
from doc_intel.eval.judge import Judge
from doc_intel.eval.ragas_style import RagasStyle
from doc_intel.eval.retrieval import _as_result, _rank
from doc_intel.eval.stats import bootstrap_mean
from doc_intel.llm.factory import build_embedder, build_llm
from doc_intel.models import Invoice
from doc_intel.pipeline import PipelineConfig
from doc_intel.rag.answer import Answerer
from doc_intel.rag.index import Indexer, Retriever
from doc_intel.rag.qa import QuestionAnswering
from doc_intel.rag.rerank import Reranker


class AnswerCase(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str
    document_id: str | None
    kind: str
    expected: str | None = None
    """Substring (normalised) the answer must contain; None for unanswerable questions."""
    language: str


class AnswerOutcome(BaseModel):
    model_config = ConfigDict(extra="forbid")

    case: AnswerCase
    answer: str
    correct: bool
    supported: bool
    not_in_documents: bool
    citations: int
    retrieval_rank: int | None
    scoped: bool
    cost_usd: Decimal
    latency_ms: int
    error: str | None = None
    judge_grade: str | None = None
    judge_reason: str | None = None
    faithfulness: float | None = None
    answer_relevancy: float | None = None


class AnswersReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    model: str
    rerank: bool
    outcomes: list[AnswerOutcome]

    def answerable(self) -> list[AnswerOutcome]:
        return [o for o in self.outcomes if o.case.expected is not None]

    def unanswerable(self) -> list[AnswerOutcome]:
        return [o for o in self.outcomes if o.case.expected is None]

    def answer_accuracy(self) -> float:
        cases = self.answerable()
        return sum(1 for o in cases if o.correct and o.supported) / len(cases) if cases else 0.0

    def decline_rate(self) -> float:
        cases = self.unanswerable()
        return sum(1 for o in cases if o.not_in_documents) / len(cases) if cases else 0.0

    def citation_support_rate(self) -> float:
        cases = [o for o in self.answerable() if not o.not_in_documents and o.error is None]
        return sum(1 for o in cases if o.supported) / len(cases) if cases else 0.0

    def scoped_recall_at_k(self) -> float:
        cases = [o for o in self.answerable() if o.error is None]
        return sum(1 for o in cases if o.retrieval_rank is not None) / len(cases) if cases else 0.0

    def by_kind(self) -> dict[str, float]:
        kinds = sorted({o.case.kind for o in self.answerable()})
        out = {}
        for kind in kinds:
            cases = [o for o in self.answerable() if o.case.kind == kind]
            out[kind] = sum(1 for o in cases if o.correct and o.supported) / len(cases)
        return out

    def judge_accuracy(self) -> float | None:
        graded = [o for o in self.answerable() if o.judge_grade]
        if not graded:
            return None
        return sum(1 for o in graded if o.judge_grade == "correct") / len(graded)

    def mean_faithfulness(self) -> float | None:
        values = [o.faithfulness for o in self.outcomes if o.faithfulness is not None]
        return statistics.mean(values) if values else None

    def mean_relevancy(self) -> float | None:
        values = [o.answer_relevancy for o in self.outcomes if o.answer_relevancy is not None]
        return statistics.mean(values) if values else None

    def metrics(self) -> dict[str, float]:
        out = {
            "answer_accuracy": self.answer_accuracy(),
            "decline_rate": self.decline_rate(),
            "citation_support_rate": self.citation_support_rate(),
            "scoped_recall_at_k": self.scoped_recall_at_k(),
        }
        for name, value in (
            ("judge_accuracy", self.judge_accuracy()),
            ("faithfulness", self.mean_faithfulness()),
            ("answer_relevancy", self.mean_relevancy()),
        ):
            if value is not None:
                out[name] = value
        answerable = [o for o in self.answerable() if o.error is None]
        if answerable:
            out["p50_latency_ms"] = float(statistics.median(o.latency_ms for o in answerable))
        return out


def _norm(text: str) -> str:
    return _NOT_LETTER.sub("", text.casefold())


def _amount(value: Decimal | None) -> str:
    return f"{value:.2f}" if value is not None else ""


def cases_for(document_id: str, truth: Invoice) -> list[AnswerCase]:
    number = truth.number.value or ""
    language = truth.language.value or "en"
    cases = [
        AnswerCase(
            question=f"What is the grand total of invoice {number}?",
            document_id=document_id,
            kind="totals",
            expected=_amount(truth.totals.grand_total.value),
            language=language,
        ),
        AnswerCase(
            question=f"Who is the supplier on invoice {number}?",
            document_id=document_id,
            kind="header",
            expected=truth.supplier.name.value,
            language=language,
        ),
        AnswerCase(
            question=f"When is invoice {number} due?",
            document_id=document_id,
            kind="header",
            expected=truth.due_date.quote,
            language=language,
        ),
    ]
    if truth.line_items:
        item = truth.line_items[0]
        cases.append(
            AnswerCase(
                question=f"How many {item.description.value} are on invoice {number}?",
                document_id=document_id,
                kind="line_item",
                expected=f"{item.quantity.value:f}".rstrip("0").rstrip(".")
                if item.quantity.value is not None
                else None,
                language=language,
            )
        )
    return cases


UNANSWERABLE = [
    AnswerCase(
        question="What is the grand total of invoice INV-9999-0001?",
        document_id=None,
        kind="missing_document",
        language="en",
    ),
    AnswerCase(
        question="Who is the CEO of Beauty Supplies Ltd?",
        document_id=None,
        kind="outside_scope",
        language="en",
    ),
    AnswerCase(
        question="What is the bank account number on invoice INV-2026-0001?",
        document_id=None,
        kind="missing_field",
        language="en",
    ),
    AnswerCase(
        question="Яка адреса електронної пошти постачальника?",
        document_id=None,
        kind="missing_field",
        language="uk",
    ),
]


def is_correct(case: AnswerCase, answer: str) -> bool:
    if case.expected is None:
        return False
    expected = _norm(case.expected)
    got = _norm(answer)
    if expected in got:
        return True
    # amounts: accept 78.60 / 78,60 / 78.6 / 78 60
    if case.kind == "totals" and re.fullmatch(r"\d+\.\d\d", case.expected):
        digits = case.expected.replace(".", "")
        return digits in got or digits[:-1] in got.replace(" ", "")
    return False


async def run(
    samples: Path, config_path: Path, limit: int | None = None, judge_enabled: bool = True
) -> AnswersReport:
    settings = get_settings()
    config = PipelineConfig.from_yaml(config_path)
    llm = build_llm(settings, config.llm.provider)
    judge: Judge | None = None
    ragas: RagasStyle | None = None
    if judge_enabled:
        judge_llm = build_llm(settings, config.eval.judge_provider)
        judge = Judge(judge_llm, config.eval.judge_model, model_under_test=config.llm.model)
    emb = config.embeddings
    embedder = build_embedder(settings, emb.provider, emb.model, emb.dimensions)
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

        if judge is not None and config.eval.ragas:
            ragas = RagasStyle(judge, embedder)
        retriever = Retriever(pool, embedder, candidates=config.rag.candidates)
        reranker = Reranker(llm, config.llm.model) if config.rag.rerank else None
        qa = QuestionAnswering(retriever, Answerer(llm, config.llm.model), reranker, config.rag)

        cases = [c for document_id, truth in truths.items() for c in cases_for(document_id, truth)]
        cases = cases[:limit] + UNANSWERABLE if limit else cases + UNANSWERABLE
        outcomes: list[AnswerOutcome] = []
        for case in cases:
            started = time.perf_counter()
            try:
                result = await qa.ask(case.question)
            except Exception as error:
                outcomes.append(
                    AnswerOutcome(
                        case=case,
                        answer="",
                        correct=False,
                        supported=False,
                        not_in_documents=False,
                        citations=0,
                        retrieval_rank=None,
                        scoped=False,
                        cost_usd=Decimal(0),
                        latency_ms=int((time.perf_counter() - started) * 1000),
                        error=f"{type(error).__name__}: {error}"[:200],
                    )
                )
                continue
            rank = None
            if case.document_id:
                from doc_intel.eval.retrieval import Question

                rank = _rank(
                    result.hits,
                    Question(
                        text=case.question,
                        document_id=case.document_id,
                        kind=case.kind,
                        language=case.language,
                    ),
                )
            outcome = AnswerOutcome(
                case=case,
                answer=result.answer,
                correct=is_correct(case, result.answer),
                supported=result.supported,
                not_in_documents=result.not_in_documents,
                citations=len(result.citations),
                retrieval_rank=rank,
                scoped=result.scope is not None,
                cost_usd=result.cost_usd,
                latency_ms=int((time.perf_counter() - started) * 1000),
            )
            if judge is not None and case.expected is not None and not result.not_in_documents:
                verdict = await judge.grade(case.question, case.expected, result.answer)
                outcome.judge_grade, outcome.judge_reason = verdict.grade, verdict.reason
            if ragas is not None:
                contexts = [h.text for h in result.hits]
                scores = await ragas.score(case.question, result.answer, contexts)
                outcome.faithfulness = scores.faithfulness
                outcome.answer_relevancy = scores.answer_relevancy
            outcomes.append(outcome)
            mark = (
                "ok "
                if (outcome.correct and outcome.supported)
                or (case.expected is None and outcome.not_in_documents)
                else "no "
            )
            print(
                f"{mark} {case.kind:16} {outcome.latency_ms:6d}ms  {case.question[:60]:60}"
                f"  -> {result.answer[:70]}",
                flush=True,
            )
    finally:
        await pool.close()
    return AnswersReport(
        config=config.name, model=config.llm.model, rerank=config.rag.rerank, outcomes=outcomes
    )


def print_report(report: AnswersReport) -> None:
    answerable = [o for o in report.answerable() if o.error is None]
    rerank = "on" if report.rerank else "off"
    print(
        f"\n== {report.config} / {report.model} (rerank={rerank}): {len(report.outcomes)} questions"
    )
    print(f"  answer accuracy (correct + cited)  {report.answer_accuracy():6.0%}")
    print(f"  correct declines (unanswerable)    {report.decline_rate():6.0%}")
    print(f"  citation support rate              {report.citation_support_rate():6.0%}")
    print(f"  scoped retrieval recall@k          {report.scoped_recall_at_k():6.0%}")
    print("  by kind  " + "  ".join(f"{k} {v:.0%}" for k, v in report.by_kind().items()))
    ci = bootstrap_mean([1.0 if (o.correct and o.supported) else 0.0 for o in answerable])
    print(f"  accuracy 95% CI       [{ci.low:.0%}, {ci.high:.0%}] over {ci.n} questions")
    if report.judge_accuracy() is not None:
        print(f"  judge says correct    {report.judge_accuracy():6.0%}")
    if report.mean_faithfulness() is not None:
        print(f"  faithfulness          {report.mean_faithfulness():6.2f}")
    if report.mean_relevancy() is not None:
        print(f"  answer relevancy      {report.mean_relevancy():6.2f}")
    if answerable:
        latencies = sorted(o.latency_ms for o in answerable)
        p95 = latencies[int(0.95 * (len(latencies) - 1))]
        print(f"  mean cost / question  ${statistics.mean(o.cost_usd for o in answerable):.5f}")
        print(f"  p50 / p95 latency     {statistics.median(latencies):.0f} ms / {p95:.0f} ms")
    errors = [o for o in report.outcomes if o.error]
    if errors:
        print(f"  errors                {len(errors)} (first: {errors[0].error})")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="End-to-end QA accuracy over the ground-truth index."
    )
    parser.add_argument("--samples", type=Path, default=Path("data/samples"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()
    report = asyncio.run(run(args.samples, args.config, args.limit))
    out = args.samples / f"eval-answers-{report.config}.json"
    out.write_text(report.model_dump_json(indent=2) + "\n")
    print_report(report)
    print(
        json.dumps(
            {
                "answer_accuracy": round(report.answer_accuracy(), 3),
                "decline_rate": round(report.decline_rate(), 3),
            }
        )
    )


if __name__ == "__main__":
    main()
