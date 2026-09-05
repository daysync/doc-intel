import random

import pytest

from doc_intel.dataset.content import LANGUAGES, generate_content
from doc_intel.dataset.layouts import LAYOUTS, render_pdf
from doc_intel.dataset.render import pdf_to_image


@pytest.mark.parametrize("layout", LAYOUTS)
@pytest.mark.parametrize("language", LANGUAGES)
def test_every_language_renders_in_every_layout(language: str, layout: str) -> None:
    content = generate_content(random.Random(11), language)
    pdf = render_pdf(content, layout, random.Random(11))
    assert pdf.startswith(b"%PDF")
    image = pdf_to_image(pdf, dpi=100)
    assert image.ndim == 3 and image.shape[2] == 3
    assert image.shape[0] > 300
    # a rendered invoice is mostly white with some ink
    dark = (image < 128).all(axis=2).mean()
    assert 0.002 < dark < 0.3


def test_rendering_is_deterministic() -> None:
    content = generate_content(random.Random(5), "en")
    assert render_pdf(content, "classic", random.Random(5)) == render_pdf(
        content, "classic", random.Random(5)
    )
