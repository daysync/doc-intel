"""``make eval-ci``: the reduced eval that runs on every pull request, offline.

Four ground-truth documents from ``tests/fixtures/eval-samples`` are indexed and 16 + 4
questions are answered through recorded fixtures (``LLM_REPLAY=1``), so the job needs no
model and no keys, only Postgres. The result is compared with the committed baseline and a
Markdown summary is written for the job. A prompt or pipeline change that alters a request
invalidates its fixture, which fails the job loudly instead of silently skipping the check.
"""

import argparse
import asyncio
import os
from pathlib import Path

from doc_intel.eval.answers import AnswersReport, print_report, run
from doc_intel.eval.compare import compare

SAMPLES = Path("tests/fixtures/eval-samples")
BASELINE = SAMPLES / "baseline-answers.json"


def summary(current: AnswersReport, baseline: AnswersReport | None) -> str:
    lines = [
        "## Reduced eval (replayed fixtures)",
        "",
        f"- answer accuracy (correct + cited): **{current.answer_accuracy():.0%}**",
        f"- correct declines: **{current.decline_rate():.0%}**",
        f"- citation support: **{current.citation_support_rate():.0%}**",
        f"- scoped retrieval recall@5: **{current.scoped_recall_at_k():.0%}**",
        "",
    ]
    if baseline is not None:
        lines += ["```", compare(baseline, current), "```"]
    else:
        lines.append("_no committed baseline yet_")
    errors = [o for o in current.outcomes if o.error]
    if errors:
        lines += ["", f"**{len(errors)} question(s) errored** (first: `{errors[0].error}`)"]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Reduced, offline eval for CI.")
    parser.add_argument("--config", type=Path, default=Path("configs/default.yaml"))
    parser.add_argument(
        "--update-baseline", action="store_true", help="write the current report as the baseline"
    )
    args = parser.parse_args()
    os.environ.setdefault("LLM_REPLAY", "1")
    os.environ.setdefault("LLM_FIXTURES_DIR", "tests/fixtures/llm/eval")
    from doc_intel.api.settings import get_settings

    get_settings.cache_clear()
    report = asyncio.run(run(SAMPLES, args.config, judge_enabled=False))
    print_report(report)
    baseline = (
        AnswersReport.model_validate_json(BASELINE.read_text()) if BASELINE.exists() else None
    )
    text = summary(report, baseline)
    print("\n" + text)
    step_summary = os.environ.get("GITHUB_STEP_SUMMARY")
    if step_summary:
        Path(step_summary).write_text(text + "\n")
    if args.update_baseline:
        BASELINE.write_text(report.model_dump_json(indent=2) + "\n")
        print(f"\nbaseline written to {BASELINE}")
    errors = [o for o in report.outcomes if o.error]
    if errors:
        raise SystemExit(
            f"{len(errors)} question(s) errored; a missing fixture means a request changed"
        )


if __name__ == "__main__":
    main()
