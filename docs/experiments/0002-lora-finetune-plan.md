# 0002 — LoRA fine-tune of a small extraction model (plan, not run)

**Status:** planned · **Stage:** 5 · **Decision 2026-09-06:** document now, run later.

## Question

The 3B local model extracts 69% of key fields correctly (Stage 2 baseline), with dates near
zero and receipts weakest. Would a LoRA fine-tune of a small open model on synthetic
invoice → JSON pairs beat prompting the same model, and by how much per language and layout?

## Why it is plausible

Extraction here is a narrow, well-specified task with a fixed output schema and an
unlimited supply of perfectly labeled training data (`make dataset` renders invoices whose
ground truth is known by construction). That is the textbook case where a small tuned model
catches up with a much larger prompted one.

## Plan

1. **Data.** `make dataset --n 2000 --seed 1` (about 4 minutes, ~1 GB) split 90/10 by document.
   Training pair = (page image *or* OCR text, target JSON = `truth.json`). Two variants:
   text-only (fine-tune a text model on OCR output) and vision (fine-tune a VL model on the
   photo). Hold out one layout family entirely to measure generalisation.
2. **Base models.** Text: Qwen2.5 3B / 7B. Vision: Qwen2.5-VL 3B. Apple Silicon path: MLX with
   `mlx-lm` LoRA (rank 8–16, ~1–3 h for 2k examples on a laptop); Linux path: PEFT + bitsandbytes.
3. **Training target.** The exact `Invoice` JSON, quotes included, so the model learns to quote
   and to produce the schema without structured-output constraints.
4. **Evaluation.** `make eval` with a new config pointing Ollama at the tuned model
   (`ollama create` from the merged weights), compared to the baseline with paired bootstrap
   intervals per field, per language, per layout. Also `make eval-answers` to check nothing
   downstream regressed.
5. **Success criterion.** ≥ +15 points on `all_fields` on the held-out layout, dates above 80%,
   no loss on the other layouts. Cost: zero per document at inference; the training time is the
   only spend.

## Risks

- Synthetic-only training may overfit to the generator's layouts; the held-out family is the
  guard, and a handful of hand-labeled real invoices (private, never in this repo) would be
  the true test.
- Structured-output constraints and a tuned model can fight each other; evaluate both with
  and without `format=schema`.
- 4-bit inference of the tuned model may lose some of the gain; measure the quantised variant.

## What to read first

`src/doc_intel/dataset/generate.py` (training data), `src/doc_intel/eval/run.py` (the metric),
`docs/experiments/0001-rerank-on-vs-off.md` (how a decision is written up here).
