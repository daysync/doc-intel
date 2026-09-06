.PHONY: api test lint typecheck fmt dataset eval eval-retrieval eval-answers llm-smoke

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

dataset:    ## generate labeled synthetic documents into data/samples
	uv run python -m doc_intel.dataset.generate --out data/samples --n 24 --seed 7

eval:       ## run the pipeline over data/samples and report field accuracy
	uv run python -m doc_intel.eval.run --samples data/samples --config configs/default.yaml

llm-smoke:  ## same prompt across providers, compare cost and latency
	uv run python -m doc_intel.llm.smoke

eval-retrieval:  ## recall@5 and MRR of hybrid retrieval over ground-truth questions (needs Postgres)
	uv run python -m doc_intel.eval.retrieval --samples data/samples --config configs/default.yaml

eval-answers:  ## end-to-end QA accuracy and correct declines over the ground-truth index (Postgres + LLM)
	uv run python -m doc_intel.eval.answers --samples data/samples --config configs/default.yaml
