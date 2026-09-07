# 0001 — LLM rerank of fused candidates: on vs off

**Date:** 2026-09-06 · **Configs:** `configs/default.yaml` (off) vs `configs/rerank.yaml` (on) · **Model:** Ollama `qwen2.5vl:3b` for both answering and reranking · **Embeddings:** `nomic-embed-text`

## Question

Hybrid retrieval (pgvector cosine + Postgres full-text, fused by reciprocal rank fusion) hands the
top 5 excerpts to the answer model. Does a listwise LLM rerank of the top 20 fused candidates,
one extra structured-output call per question, produce more correct, cited answers?

## Method

`make eval-answers` on the ground-truth index: 96 answerable questions (4 per synthetic
invoice: grand total, supplier, due date, first line item quantity) plus 4 unanswerable ones.
"Correct" means the expected value appears in the answer **and** at least one citation was
verified verbatim in its chunk. Paired comparison over the questions both runs answered
(`make eval-compare`), 2,000 bootstrap resamples, 95% interval.

## Results

| | rerank off | rerank on |
|---|---|---|
| answer accuracy (correct + cited) | 90.0% [83.3%, 95.6%] | 91.1% [85.6%, 96.7%] |
| paired difference on − off | **+1.1% [+0.0%, +3.3%]**, n = 90 | |
| citation support rate | 97% | 98% |
| correct declines | 75% | 75% |
| latency p50 / p95 | 2.0 s / 2.9 s | 3.3 s / 4.2 s |
| cost per question | $0 | $0 (would double on a paid model) |

## Decision

**No measurable difference.** The interval touches zero; one more correct answer in ninety is
within noise, and the rerank adds 60% latency and would double the per-question cost on a
paid model. Rerank stays off in `configs/default.yaml`. The config is kept so the question can
be re-asked when (a) the corpus is large enough that fusion's top 5 is often wrong, or (b) a
stronger reranker is cheap. Scoping search to the invoice a question names (Stage 3b) already
took scoped recall to 99%, which leaves the reranker little to fix.
