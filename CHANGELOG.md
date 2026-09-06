# Changelog

All notable changes to this project. Format follows Keep a Changelog; versions follow SemVer.

## [0.1.0] — 2026-09-07

First release: the whole pipeline runs locally against free models and is measured.

### Added
- Core `Invoice` schema with a source quote and confidence on every field; lenient parsing
  of amounts and dates.
- One `LLM` interface with Anthropic, OpenAI, Ollama and recorded adapters; cost and latency
  per call; `Embedder` interface with the same shape.
- Synthetic invoice dataset in en/ru/uk/ka across three layouts, with phone-photo degradation
  and planted inconsistencies (`make dataset`).
- Preprocessing (deskew, adaptive threshold), Tesseract with script detection, vision-model
  fallback.
- Extraction with a repair round; thirteen validation rules incl. cross-document duplicates.
- Postgres persistence, structure-aware chunking, pgvector + full-text hybrid retrieval with
  reciprocal rank fusion, document scoping by invoice number.
- `POST /ask` with verified citations and a first-class "Not in the documents." answer.
- Evals: field accuracy, retrieval recall@k / MRR, end-to-end answer accuracy, LLM judge,
  Ragas-style faithfulness and relevancy, bootstrap intervals, MLflow tracking, paired A/B
  comparison, and a reduced offline eval on every pull request.
- API-key auth, request ids, JSON logging, `/ready`, upload limits, Dockerfile and compose
  stack.

### Known limits
- Baselines are on a 3B local model; see the README results table.
- Terraform/AWS deployment is not included.
