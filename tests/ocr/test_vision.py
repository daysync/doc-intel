"""Vision transcription through the recorded adapter: replays a fixture, no network in CI.

Record locally with the free Ollama model:
    LLM_RECORD=1 uv run pytest tests/ocr/test_vision.py
"""

import os
import random
from pathlib import Path

import numpy as np
import pytest

from doc_intel.api.settings import Settings
from doc_intel.dataset.content import generate_content
from doc_intel.dataset.layouts import render_pdf
from doc_intel.dataset.render import pdf_to_image
from doc_intel.llm.errors import FixtureMissingError
from doc_intel.llm.factory import build_live_llm
from doc_intel.llm.recorded import RecordedLLM
from doc_intel.ocr.engine import Ocr
from doc_intel.ocr.vision import VisionTranscriber

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm" / "vision"
MODEL = "qwen2.5vl:3b"


def _recorded_llm() -> RecordedLLM:
    if os.environ.get("LLM_RECORD") == "1":
        return RecordedLLM(FIXTURES, inner=build_live_llm(Settings(_env_file=None), "ollama"))
    return RecordedLLM(FIXTURES, inner=None, priced_as="ollama")


def _small_page() -> np.ndarray:
    """A deterministic, low-resolution page so the fixture stays keyed to identical bytes."""
    rng = random.Random(21)
    return pdf_to_image(render_pdf(generate_content(rng, "en"), "receipt", rng), dpi=110)


async def test_vision_transcribes_the_receipt() -> None:
    rng = random.Random(21)
    truth = generate_content(rng, "en").invoice
    transcriber = VisionTranscriber(_recorded_llm(), MODEL)
    try:
        page = await transcriber.transcribe(_small_page())
    except FixtureMissingError as error:
        pytest.skip(f"no fixture yet: {error}")
    assert truth.number.quote and truth.number.quote in page.text
    assert page.mean_confidence > 50
    assert page.languages.startswith(MODEL)


async def test_engine_uses_tesseract_when_confident() -> None:
    result = await Ocr(vision=None).run([_small_page()])
    assert result.engine == "tesseract"
    assert result.cost_usd == 0


async def test_engine_falls_back_to_vision_when_threshold_is_high() -> None:
    transcriber = VisionTranscriber(_recorded_llm(), MODEL)
    try:
        result = await Ocr(vision=transcriber, fallback_below=101.0).run([_small_page()])
    except FixtureMissingError as error:
        pytest.skip(f"no fixture yet: {error}")
    assert result.engine == "vision"
    assert "INV-" in result.text or "-2026-" in result.text
