from pathlib import Path

import pytest

from doc_intel.eval.tracking import flatten_config, format_diff, log_run, previous_metrics


def test_flatten_config_uses_dotted_keys() -> None:
    assert flatten_config({"llm": {"model": "m", "x": {"y": 1}}, "k": 5}) == {
        "llm.model": "m",
        "llm.x.y": 1,
        "k": 5,
    }


def test_format_diff_marks_direction() -> None:
    text = format_diff({"acc": 0.7, "new": 1.0}, {"acc": 0.65})
    assert "▲ +0.050" in text and "new" in text
    assert "vs previous" not in format_diff({"acc": 0.7}, None)


def test_log_run_writes_to_a_local_store(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MLFLOW_TRACKING_URI", f"sqlite:///{tmp_path / 'mlflow.db'}")
    monkeypatch.chdir(tmp_path)  # artifacts land under ./mlartifacts
    monkeypatch.setenv("MLFLOW_EXPERIMENT", "test")
    report = tmp_path / "report.json"
    report.write_text("{}")
    assert previous_metrics("answers", "default") is None
    run_id = log_run("answers", "default", {"llm.model": "m"}, {"answer_accuracy": 0.89}, report)
    assert run_id
    assert previous_metrics("answers", "default") == {"answer_accuracy": 0.89}
