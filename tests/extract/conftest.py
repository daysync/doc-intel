"""Shared helpers: a deterministic synthetic document and the recorded Ollama adapter."""

import os
import random
from pathlib import Path

import numpy as np
import pytest

from doc_intel.api.settings import Settings
from doc_intel.dataset.content import InvoiceContent, generate_content
from doc_intel.dataset.layouts import render_pdf
from doc_intel.dataset.render import pdf_to_image
from doc_intel.llm.factory import build_live_llm
from doc_intel.llm.recorded import RecordedLLM

FIXTURES = Path(__file__).parent.parent / "fixtures" / "llm" / "extract"
MODEL = "qwen2.5vl:3b"


def recorded_llm() -> RecordedLLM:
    if os.environ.get("LLM_RECORD") == "1":
        return RecordedLLM(FIXTURES, inner=build_live_llm(Settings(_env_file=None), "ollama"))
    return RecordedLLM(FIXTURES, inner=None, priced_as="ollama")


@pytest.fixture(scope="session")
def en_receipt() -> tuple[InvoiceContent, np.ndarray]:
    rng = random.Random(21)
    content = generate_content(rng, "en")
    image = pdf_to_image(render_pdf(content, "receipt", rng), dpi=110)
    return content, image
