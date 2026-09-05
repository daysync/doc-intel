.PHONY: api test lint typecheck fmt dataset eval llm-smoke

api:        ## run the API with autoreload on :8000
	uv run uvicorn doc_intel.api.app:app --factory --reload --port 8000

test:       ## run the test suite
	uv run pytest

lint:       ## ruff lint + format check
	uv run ruff check . && uv run ruff format --check .

fmt:        ## auto-format and auto-fix
	uv run ruff format . && uv run ruff check --fix .

typecheck:  ## mypy strict
	uv run mypy

dataset:    ## generate labeled synthetic documents (Stage 2)
	@echo "make dataset: arrives in Stage 2"

eval:       ## run the golden set, log to MLflow (Stage 4)
	@echo "make eval: arrives in Stage 4"

llm-smoke:  ## same prompt across providers, compare cost and latency
	uv run python -m doc_intel.llm.smoke
