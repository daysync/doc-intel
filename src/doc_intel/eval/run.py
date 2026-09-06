"""``make eval``: run the pipeline over the synthetic dataset and report accuracy.

Writes ``data/samples/eval-<config>.json`` with every document's scores, cost, latency and
issues found, then prints per-field accuracy and breakdowns by language, layout and
degradation level. Stage 4 will log the same numbers to MLflow and add retrieval and
generation metrics; the per-document records here are already shaped for that.
"""

import argparse
import asyncio
import json
import statistics
import time
from collections import defaultdict
from datetime import date
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from doc_intel.dataset.generate import DocumentMeta, Manifest, load_truth
from doc_intel.eval.accuracy import accuracy, compare
from doc_intel.eval.tracking import flatten_config, format_diff, log_run, previous_metrics
from doc_intel.extract.rules import error_codes
from doc_intel.pipeline import Pipeline, PipelineConfig


class DocumentEval(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    language: str
    layout: str
    level: str
    ok: bool
    error: str | None = None
    scores: dict[str, bool] = {}
    issues_found: list[str] = []
    expected_issues: list[str] = []
    planted_found: bool | None = None
    ocr_engine: str | None = None
    repairs: int = 0
    cost_usd: Decimal = Decimal(0)
    latency_ms: int = 0


class EvalReport(BaseModel):
    model_config = ConfigDict(extra="forbid")

    config: str
    model: str
    documents: list[DocumentEval]

    @property
    def succeeded(self) -> list[DocumentEval]:
        return [d for d in self.documents if d.ok]

    def overall(self) -> dict[str, float]:
        return accuracy([d.scores for d in self.succeeded])

    def by(self, attribute: str) -> dict[str, float]:
        groups: dict[str, list[dict[str, bool]]] = defaultdict(list)
        for d in self.succeeded:
            groups[getattr(d, attribute)].append(d.scores)
        return {key: accuracy(scores)["all_fields"] for key, scores in sorted(groups.items())}


async def evaluate_document(
    pipeline: Pipeline, samples: Path, meta: DocumentMeta, today: date
) -> DocumentEval:
    folder = samples / meta.id
    record = DocumentEval(
        id=meta.id,
        language=meta.language,
        layout=meta.layout,
        level=meta.level,
        ok=False,
        expected_issues=meta.expected_issues,
    )
    started = time.perf_counter()
    try:
        result = await pipeline.process(
            (folder / "photo.jpg").read_bytes(), "image/jpeg", document_id=meta.id
        )
    except Exception as error:
        record.error = f"{type(error).__name__}: {error}"[:300]
        record.latency_ms = int((time.perf_counter() - started) * 1000)
        return record
    truth = load_truth(folder)
    record.ok = True
    record.scores = compare(truth, result.invoice)
    record.issues_found = sorted(error_codes(result.issues))
    single_doc_expected = [i for i in meta.expected_issues if i != "duplicate_number"]
    if single_doc_expected:
        found = record.issues_found
        record.planted_found = all(code in found for code in single_doc_expected)
    record.ocr_engine = result.ocr_engine
    record.repairs = result.repairs
    record.cost_usd = result.cost_usd
    record.latency_ms = result.timings.total_ms
    return record


async def run(samples: Path, config: Path, limit: int | None, today: date) -> EvalReport:
    manifest = Manifest.model_validate_json((samples / "manifest.json").read_text())
    pipeline = Pipeline.from_config(config)
    report = EvalReport(config=pipeline.config.name, model=pipeline.config.llm.model, documents=[])
    for meta in manifest.documents[:limit]:
        record = await evaluate_document(pipeline, samples, meta, today)
        report.documents.append(record)
        status = (
            f"{accuracy([record.scores])['all_fields']:.0%}"
            if record.ok
            else f"FAILED {record.error}"
        )
        print(
            f"{meta.id:32} {record.latency_ms:6d}ms  repairs={record.repairs}  {status}", flush=True
        )
    return report


def print_report(report: EvalReport) -> None:
    docs = report.succeeded
    processed = f"{len(docs)}/{len(report.documents)} documents processed"
    print(f"\n== {report.config} / {report.model}: {processed}")
    if not docs:
        return
    overall = report.overall()
    print("\nfield accuracy")
    for field, value in overall.items():
        print(f"  {field:28} {value:6.0%}")
    for attribute in ("language", "layout", "level"):
        print(f"\nby {attribute}")
        for key, value in report.by(attribute).items():
            print(f"  {key:12} {value:6.0%}")
    latencies = [d.latency_ms for d in docs]
    planted = [d for d in docs if d.planted_found is not None]
    print("\nrun")
    print(f"  mean cost / doc     ${statistics.mean(d.cost_usd for d in docs):.5f}")
    p95 = sorted(latencies)[int(0.95 * (len(latencies) - 1))]
    print(f"  p50 / p95 latency   {statistics.median(latencies):.0f} ms / {p95:.0f} ms")
    print(f"  repair rate         {sum(1 for d in docs if d.repairs) / len(docs):.0%}")
    print(
        f"  vision fallback     {sum(1 for d in docs if d.ocr_engine == 'vision') / len(docs):.0%}"
    )
    if planted:
        print(f"  planted issues found {sum(1 for d in planted if d.planted_found)}/{len(planted)}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Field accuracy of the pipeline on the synthetic dataset."
    )
    parser.add_argument("--samples", type=Path, default=Path("data/samples"))
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-track", action="store_true", help="do not log the run to MLflow")
    args = parser.parse_args()
    report = asyncio.run(run(args.samples, args.config, args.limit, date.today()))
    out = args.samples / f"eval-{report.config}.json"
    out.write_text(report.model_dump_json(indent=2) + "\n")
    print_report(report)
    print(f"\nwrote {out}")
    if not args.no_track and report.succeeded:
        overall = report.overall()
        metrics = {f"field.{k}": v for k, v in overall.items()}
        metrics["documents_processed"] = len(report.succeeded) / len(report.documents)
        previous = previous_metrics("accuracy", report.config)
        config = PipelineConfig.from_yaml(args.config)
        run_id = log_run(
            "accuracy", report.config, flatten_config(config.model_dump()), metrics, out
        )
        print(f"mlflow run {run_id}\n{format_diff(metrics, previous)}")
    print(json.dumps({"all_fields": round(report.overall().get("all_fields", 0), 3)}))


if __name__ == "__main__":
    main()
