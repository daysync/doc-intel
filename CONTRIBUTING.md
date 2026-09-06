# Contributing

Thanks for looking. The most useful contributions right now: invoice layouts from
countries and languages the synthetic generator does not cover, OCR backends, and eval
cases where the pipeline fails.

## Setup

```bash
uv sync
brew install tesseract tesseract-lang   # Debian/Ubuntu: apt install tesseract-ocr tesseract-ocr-{eng,rus,ukr,kat}
docker compose up -d postgres
make dataset
make lint typecheck test
```

Ollama is optional but makes everything free: `brew install ollama`, then
`ollama pull qwen2.5vl:3b nomic-embed-text qwen2.5:7b`.

## How the code is organised

`src/doc_intel/` — `models/` (the schema everything derives from), `llm/` (one interface,
provider adapters, recorded fixtures), `dataset/` (synthetic invoices with ground truth),
`ocr/`, `extract/`, `rag/`, `eval/`, `db/`, `api/`, and `pipeline.py` that ties the stages
together from a YAML in `configs/`. Every stage has a note in the maintainers' learning log;
the README's layout block is the map.

## Rules that keep the repo healthy

- **Every commit passes `make lint typecheck test`.** mypy runs strict; do not weaken it.
- **Every external call goes through an interface** (`LLM`, `Embedder`) and is recorded as a
  fixture so tests and CI run offline. Record with `LLM_RECORD=1`; fixtures are keyed by the
  request *and* the JSON schema, so a changed prompt or model docstring means re-recording.
- **No real customer documents, ever.** Synthetic data only; the generator is the source.
- **Evals decide.** A change to a prompt, model or pipeline setting comes with a `make eval*`
  result and, for A/B decisions, a `make eval-compare` interval. Write it up in
  `docs/experiments/`.
- **The judge is never the model under test.**
- **Conventional commits**, small and single-purpose; branches off `main`.

## Pull requests

CI runs lint, types, tests against a pgvector Postgres, and the reduced offline eval
(`make eval-ci`) whose summary appears on the job. A PR that changes a request shape must
also update fixtures; a PR that changes quality must include the numbers.

## Reporting a security issue

See `SECURITY.md`. Please do not open a public issue for vulnerabilities.
