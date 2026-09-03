# doc-intel

**Open-source document intelligence by [DaySync](https://daysync.io).**

Turn photos and PDFs of supplier invoices and contracts into validated structured data, keep the documents searchable, and measure every step of the pipeline.

DaySync is a booking and business-management platform for salons. Salon owners receive supplier invoices on paper, photograph them with a phone, and expect products and expenses to appear in their ledger without manual entry. `doc-intel` is the engine behind that feature, published as a standalone Python library and service so any product with the same problem can use it, and so the quality of the pipeline is measured in the open.

## Status

| Stage | Scope | State |
|---|---|---|
| 0 | Project skeleton, models, API, CI | – |
| 1 | LLM wrapper: OpenAI / Anthropic / Ollama, structured output, cost and latency logging | – |
| 2 | OCR and extraction: synthetic dataset, field-level accuracy, cross-document validation | – |
| 3 | RAG: structure-aware chunking, pgvector hybrid search, reranking, cited answers | – |
| 4 | Evals: Ragas, LLM-as-judge, MLflow tracking, A/B of prompts and models | – |
| 5 | Open-source models: local inference, LoRA fine-tune on extraction | – |
| 6 | AWS deploy, eval in CI on PRs, monitoring | – |

Latest results (updated with each `make eval` run):

| Configuration | Field accuracy | Retrieval recall@5 | Faithfulness | Cost / doc | p95 latency |
|---|---|---|---|---|---|
| – | – | – | – | – | – |

Experiment write-ups live in `docs/experiments/`.

## What it does

```
POST /ingest        upload documents (PDF, JPG, PNG, HEIC) → job_id
GET  /documents     extracted fields per document with validation status
GET  /issues        inconsistencies across documents: totals, dates, counterparties
POST /ask           question over the documents → answer with citations (document, page, snippet)
GET  /metrics       quality metrics from the latest eval run
```

Pipeline per document:

1. **Preprocess** – phone photos are the primary input: deskew, denoise, glare and perspective correction before OCR.
2. **OCR** – text with bounding boxes and confidence per block; languages `eng`, `rus`, `ukr`, `kat`. Low-confidence pages fall back to a vision model.
3. **Extract** – LLM fills a pydantic schema (`Invoice`: supplier, buyer, number, date, currency, line items with quantity and unit price, totals, taxes) with a source quote per field.
4. **Validate** – field rules (totals reconcile with line items, dates plausible, currency known) and cross-document checks (invoice vs contract terms, duplicate numbers, supplier mismatch).
5. **Index** – structure-aware chunks with metadata into Postgres + pgvector; hybrid retrieval (vector + BM25) with reranking.
6. **Answer** – questions over the corpus with citations; "not in the documents" is a first-class answer.

Every external call goes through an interface, so any layer can be replaced or mocked. Evals run without network on recorded responses.

## Using it in your product

The service is the reference deployment; the same code is importable:

```bash
uv add daysync-doc-intel
```

```python
from doc_intel import Pipeline

pipeline = Pipeline.from_config("configs/default.yaml")
result = await pipeline.process(photo_bytes, mime="image/jpeg")

result.invoice        # Invoice, validated
result.issues         # list[ValidationIssue]
result.confidence     # per-field confidence for a review screen
result.cost_usd       # what this document cost to process
```

Design rules that keep it embeddable:

- Input is bytes plus MIME type; storage of originals is the host's concern.
- Output is a pydantic model plus per-field confidence, so the host can show a review screen instead of trusting the result blindly.
- Language and currency are detected, not configured, because one account can receive invoices in several languages.
- Cost per document is returned with the result; the host decides what to do with expensive documents.
- No host-specific fields in the schema; a host maps `Invoice` to its own products and expenses.

## Evaluation

Quality is measured per layer on a synthetic, labeled dataset (`make dataset`): invoices and contracts rendered with varied layouts, fonts and scan noise, with ground truth written at generation time and deliberate inconsistencies planted across documents.

- **Extraction** – exact match for identifiers and dates, tolerance for amounts, reported per field and per layout family.
- **Retrieval** – recall@k and MRR against known source blocks for each question.
- **Generation** – Ragas (faithfulness, answer relevancy, context precision) plus an LLM judge with a fixed rubric; the judge model is never the model under test.
- **End to end** – share of questions answered correctly with valid citations, share of correct "not in the documents".

Each `make eval` run logs parameters, metrics and error tables to MLflow and prints a diff against the previous run. A/B decisions between prompts, models or pipeline settings are made on bootstrap confidence intervals over documents, and each one is written up in `docs/experiments/`.

## Stack

Python 3.12, uv, FastAPI, pydantic v2, asyncio · Tesseract / PaddleOCR, OpenCV · Anthropic, OpenAI, Ollama · Postgres 16 + pgvector · Ragas, MLflow · pytest, ruff, mypy, GitHub Actions · Terraform, AWS (ECS Fargate, RDS, S3, CloudWatch)

## Running locally

```bash
uv sync
cp .env.example .env            # provider keys, DATABASE_URL
docker compose up -d postgres
make dataset                    # generate labeled synthetic documents
make api                        # http://localhost:8000
make eval                       # run the golden set, log to MLflow
make llm-smoke                  # same prompt across providers, compare cost and latency
```

## Layout

```
src/doc_intel/
  api/        FastAPI app
  llm/        provider adapters, structured output, cost logging
  ocr/        preprocessing, OCR backends, vision fallback
  extract/    schemas, prompts, validation rules
  rag/        chunking, indexing, retrieval, answering
  eval/       golden set, metrics, MLflow integration
tests/
data/samples/   generated documents and ground truth
configs/        pipeline configurations under comparison
docs/experiments/
infra/          Terraform
```

## Contributing

Issues and pull requests are welcome. The most useful contributions right now: invoice layouts from countries and languages not yet in the synthetic generator, OCR backends, and eval cases where the pipeline fails. Every PR runs lint, type checks and tests; PRs that touch prompts or pipeline settings also run the reduced eval set and post the metrics diff.

## License

MIT © DaySync LLC
