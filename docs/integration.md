# Integrating doc-intel into a product

The service is generic on purpose: it turns a document into an `Invoice` with a quote and a
confidence per field, plus issues, cost and timings. Mapping that to your own suppliers,
products and expenses is your side of the line.

## Two ways in

1. **HTTP.** `POST /ingest` (multipart `file`) returns a job id; poll `GET /documents/{id}`
   until `status` is `done` or `failed`. `GET /issues` adds cross-document findings;
   `POST /ask` answers questions with citations.
2. **Library.** `uv add daysync-doc-intel`, then:

```python
from doc_intel.pipeline import Pipeline

pipeline = Pipeline.from_config("configs/default.yaml")
result = await pipeline.process(photo_bytes, mime="image/jpeg")
result.invoice        # Invoice, validated
result.issues         # list[ValidationIssue]
result.confidence     # {"number": 0.9, "line_items[0].quantity": 0.6, ...}
result.cost_usd       # what this document cost
```

## What a host should do with the result

- **Show a review screen, do not auto-commit.** `confidence` is flat, keyed by field path, so a
  UI can highlight what to check. `issues` carry stable codes (`totals_mismatch`,
  `date_quote_mismatch`, `duplicate_number`, …) with `field`, `expected`, `actual`.
- **Match suppliers and products yourself.** `Invoice.supplier.name` and each
  `line_items[i].description` are what the document says; your catalogue decides what they
  map to. Keep that matching in your codebase; it is product-specific.
- **Meter by `cost_usd`.** Every result says what it cost; a per-account budget or a plan
  gate is a one-line check on the host side.
- **Store originals yourself.** The service takes bytes and keeps none.

## Choosing a configuration

Copy `configs/default.yaml`, change `llm.provider`/`llm.model`, `extraction.strategy` or
`rag.*`, and run `make eval` and `make eval-answers` against it. The README results table is
the format to report; `make eval-compare` says whether a change is real.
