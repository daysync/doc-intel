"""MLflow tracking for every eval run: params, metrics, and the JSON report as an artifact.

Local SQLite store (``mlflow.db``, git-ignored; MLflow 3 retired the file store) and
``mlartifacts/`` for reports; ``make mlflow-ui`` opens the comparison UI. Runs are named by
eval kind and config so two configurations sit side by side.
"""

import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"


def _mlflow() -> Any:
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    import mlflow

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT", "doc-intel"))
    return mlflow


def log_run(
    kind: str,
    config_name: str,
    params: Mapping[str, object],
    metrics: Mapping[str, float],
    report_path: Path | None = None,
    tags: Mapping[str, str] | None = None,
) -> str:
    """Record one eval run. Returns the MLflow run id."""
    mlflow = _mlflow()
    with mlflow.start_run(run_name=f"{kind}/{config_name}") as run:
        mlflow.set_tags({"kind": kind, "config": config_name, **(tags or {})})
        mlflow.log_params({k: str(v)[:250] for k, v in params.items()})
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        if report_path is not None and report_path.exists():
            mlflow.log_artifact(str(report_path))
        return str(run.info.run_id)


def flatten_config(config: Mapping[str, Any], prefix: str = "") -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in config.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            out.update(flatten_config(value, f"{name}."))
        else:
            out[name] = value
    return out


def previous_metrics(kind: str, config_name: str) -> dict[str, float] | None:
    """Metrics of the latest earlier run of the same kind and config, for a diff."""
    mlflow = _mlflow()
    runs = mlflow.search_runs(
        filter_string=f"tags.kind = '{kind}' and tags.config = '{config_name}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
        output_format="list",
    )
    if not runs:
        return None
    return {k: float(v) for k, v in runs[0].data.metrics.items()}


def format_diff(current: Mapping[str, float], previous: Mapping[str, float] | None) -> str:
    lines = []
    for key in sorted(current):
        now = current[key]
        if previous is None or key not in previous:
            lines.append(f"  {key:32} {now:8.3f}")
        else:
            delta = now - previous[key]
            arrow = "▲" if delta > 0.0005 else "▼" if delta < -0.0005 else "="
            lines.append(f"  {key:32} {now:8.3f}  {arrow} {delta:+.3f} vs previous")
    return "\n".join(lines)


def dump(obj: Any) -> str:
    return json.dumps(obj, default=str)
