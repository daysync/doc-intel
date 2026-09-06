# Two stages: build the virtualenv with uv, then copy it into a slim runtime with Tesseract.
# Runs as a non-root user; the API listens on 8000; /health is the liveness probe.

FROM python:3.12-slim AS build
COPY --from=ghcr.io/astral-sh/uv:0.12 /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy UV_PYTHON_DOWNLOADS=never
WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY src ./src
COPY README.md ./
RUN uv sync --frozen --no-dev

FROM python:3.12-slim AS runtime
RUN apt-get update -q \
 && apt-get install -y -q --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng tesseract-ocr-rus tesseract-ocr-ukr tesseract-ocr-kat \
      libglib2.0-0 curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 1000 app
WORKDIR /app
COPY --from=build --chown=app:app /app /app
COPY --chown=app:app configs ./configs
COPY --chown=app:app data/fonts ./data/fonts
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1 PIPELINE_CONFIG=configs/default.yaml
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD curl -fsS http://localhost:8000/health || exit 1
CMD ["uvicorn", "doc_intel.api.app:app", "--factory", "--host", "0.0.0.0", "--port", "8000"]
