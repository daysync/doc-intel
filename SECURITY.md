# Security

## Reporting

Email security@daysync.io with a description and, if possible, steps to reproduce. You will
get an acknowledgement within three working days. Please do not open a public issue.

## What the service does with data

- Documents arrive as bytes over HTTP and are processed in memory; the service does not
  store originals. Extracted data, OCR text and their embeddings are stored in Postgres.
- Processing may send page images and text to the configured model provider (Anthropic,
  OpenAI, or a local Ollama). Choose the provider with that in mind; the local path never
  leaves the machine.
- Nothing in this repository contains real documents; the dataset is synthetic.

## Hardening checklist for a deployment

- Set `API_KEYS`; the API is open when it is empty (development mode), and `/ready` says so.
- Terminate TLS in front of the container; the service speaks plain HTTP.
- Keep provider keys and `DATABASE_URL` in the environment or a secrets manager, never in
  the image or the repository.
- Cap uploads with `MAX_UPLOAD_MB`; put a request-rate limit at the edge.
- Run as the non-root user the image provides; mount nothing writable except what you need.
