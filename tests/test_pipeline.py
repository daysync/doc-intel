from datetime import date
from pathlib import Path

import numpy as np
import pytest

from doc_intel.dataset.content import InvoiceContent
from doc_intel.llm.errors import FixtureMissingError
from doc_intel.ocr.image import to_jpeg
from doc_intel.pipeline import Pipeline, PipelineConfig
from tests.extract.conftest import recorded_llm


def test_default_config_loads() -> None:
    config = PipelineConfig.from_yaml(Path("configs/default.yaml"))
    assert config.name == "default"
    assert config.llm.provider == "ollama"
    assert config.extraction.strategy == "both"


async def test_pipeline_end_to_end_on_the_receipt(
    en_receipt: tuple[InvoiceContent, np.ndarray],
) -> None:
    content, image = en_receipt
    pipeline = Pipeline(PipelineConfig(), recorded_llm(), today=date(2026, 9, 5))
    try:
        result = await pipeline.process(to_jpeg(image), "image/jpeg", document_id="doc-1")
    except FixtureMissingError as error:
        pytest.skip(f"no fixture yet: {error}")
    assert result.document_id == "doc-1"
    assert result.invoice.number.value == content.invoice.number.value
    assert result.invoice.totals.grand_total.value == content.invoice.totals.grand_total.value
    assert result.ocr_engine == "tesseract"
    assert result.model == "qwen2.5vl:3b"
    assert result.cost_usd == 0
    assert result.timings.ocr_ms is not None and result.timings.total_ms >= result.timings.ocr_ms
    codes = {issue.code for issue in result.issues}
    assert "date_implausible" in codes  # the local model's year-308 date, caught by rules
    assert result.confidence["number"] > 0.5


async def test_cross_check_flags_the_same_document_twice(
    en_receipt: tuple[InvoiceContent, np.ndarray],
) -> None:
    _, image = en_receipt
    pipeline = Pipeline(PipelineConfig(), recorded_llm(), today=date(2026, 9, 5))
    try:
        a = await pipeline.process(to_jpeg(image), "image/jpeg", document_id="a")
        b = await pipeline.process(to_jpeg(image), "image/jpeg", document_id="b")
    except FixtureMissingError as error:
        pytest.skip(f"no fixture yet: {error}")
    cross = pipeline.cross_check({"a": a, "b": b})
    assert [i.code for i in cross["a"]] == ["duplicate_number"]
    assert cross["a"][0].related_document_ids == ["b"]
