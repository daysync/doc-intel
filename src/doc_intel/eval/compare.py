"""``make eval-compare A=… B=…``: is B better than A, or just lucky?

Loads two answer-eval reports over the same questions and prints the paired difference in
answer accuracy with a 95% bootstrap interval. An interval that excludes zero is a real
difference at this sample size; one that straddles zero means the eval cannot tell.
"""

import argparse
from pathlib import Path

from doc_intel.eval.answers import AnswersReport
from doc_intel.eval.stats import bootstrap_mean, paired_difference


def compare(a: AnswersReport, b: AnswersReport) -> str:
    by_question_a = {o.case.question: o for o in a.answerable()}
    by_question_b = {o.case.question: o for o in b.answerable()}
    shared = sorted(set(by_question_a) & set(by_question_b))
    if not shared:
        return "no shared questions between the two reports"
    xa = [1.0 if (by_question_a[q].correct and by_question_a[q].supported) else 0.0 for q in shared]
    xb = [1.0 if (by_question_b[q].correct and by_question_b[q].supported) else 0.0 for q in shared]
    diff = paired_difference(xa, xb)
    verdict = (
        f"{b.config} is better"
        if diff.low > 0
        else f"{a.config} is better"
        if diff.high < 0
        else "no measurable difference"
    )
    lines = [
        f"A = {a.config} ({a.model}, rerank={'on' if a.rerank else 'off'}): {bootstrap_mean(xa)}",
        f"B = {b.config} ({b.model}, rerank={'on' if b.rerank else 'off'}): {bootstrap_mean(xb)}",
        f"paired difference B - A: {diff.mean:+.1%} [{diff.low:+.1%}, {diff.high:+.1%}]"
        f" over {diff.n} questions",
        f"verdict: {verdict}",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Paired comparison of two answer-eval reports.")
    parser.add_argument("a", type=Path)
    parser.add_argument("b", type=Path)
    args = parser.parse_args()
    a = AnswersReport.model_validate_json(args.a.read_text())
    b = AnswersReport.model_validate_json(args.b.read_text())
    print(compare(a, b))


if __name__ == "__main__":
    main()
