# Deployment

One container, one Postgres with pgvector, one model provider. Anything that runs containers
works: a VM with Docker, Render, Fly.io, ECS, Kubernetes.

## Image

```bash
docker build -t doc-intel:0.1.0 .
```

Multi-stage, Python 3.12 slim, Tesseract with `eng rus ukr kat`, non-root user, `HEALTHCHECK`
on `/health`. Listens on 8000.

## Environment

| Variable | Required | Notes |
|---|---|---|
| `DATABASE_URL` | yes | `postgresql://user:pass@host:5432/db`, Postgres 16 with the `vector` extension available (the service creates it) |
| `API_KEYS` | production | comma-separated; empty = open API |
| `PIPELINE_CONFIG` | no | default `configs/default.yaml`; mount your own to change provider, model, strategy |
| `ANTHROPIC_API_KEY` / `OPENAI_API_KEY` | per provider | only the configured provider's key is needed |
| `OLLAMA_HOST` | if Ollama | e.g. `http://ollama:11434` |
| `MAX_UPLOAD_MB` | no | default 25 |
| `LOG_FORMAT` | no | `text` or `json` |

## Run

```bash
docker run --rm -p 8000:8000 \
  -e DATABASE_URL=postgresql://doc_intel:doc_intel@host.docker.internal:5433/doc_intel \
  -e API_KEYS=change-me -e ANTHROPIC_API_KEY=... \
  doc-intel:0.1.0
```

or the whole stack locally, including Ollama:

```bash
API_KEYS=change-me docker compose --profile full up -d
docker compose exec ollama ollama pull qwen2.5vl:3b
docker compose exec ollama ollama pull nomic-embed-text
```

## Probes

- `GET /health` — liveness: the process is up.
- `GET /ready` — readiness: Postgres answers and the model host is reachable; also reports
  whether auth is `api-key` or `open`.

## Schema

`db/schema.sql` is applied idempotently on startup. It creates tables and indexes if missing;
it never alters existing columns. When the schema needs to change under deployed data, add a
migration tool (Alembic or plain numbered SQL) before changing it.

## Sizing

CPU-only is fine for Tesseract and the API. Model calls dominate latency: local Ollama needs
a GPU or Apple Silicon for acceptable speed (a 3B vision model answers in seconds on a
laptop, tens of seconds for a full extraction); hosted providers need no local compute.
Postgres with pgvector's HNSW index handles hundreds of thousands of chunks on a small
instance.

## Not included

Terraform for AWS (ECS Fargate, RDS, S3, CloudWatch) was planned and deferred; the container
and environment above are all it would need.
