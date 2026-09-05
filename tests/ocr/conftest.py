import random

import numpy as np
import pytest

from doc_intel.dataset.content import generate_content
from doc_intel.dataset.layouts import render_pdf
from doc_intel.dataset.render import pdf_to_image


@pytest.fixture(scope="session")
def clean_en_page() -> np.ndarray:
    """One deterministic clean English invoice rendered at 150 dpi."""
    rng = random.Random(21)
    return pdf_to_image(render_pdf(generate_content(rng, "en"), "classic", rng), dpi=150)


@pytest.fixture(scope="session")
def clean_en_pdf() -> bytes:
    rng = random.Random(21)
    return render_pdf(generate_content(rng, "en"), "classic", rng)
