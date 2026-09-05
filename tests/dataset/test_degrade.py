import random

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from doc_intel.dataset.degrade import LEVELS, degrade


def _page() -> np.ndarray:
    image = Image.new("RGB", (600, 800), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype("data/fonts/DejaVuSans.ttf", 24)
    for i in range(20):
        draw.text(
            (40, 40 + i * 36), f"Line {i} Invoice No INV-{i:04d} 123.45", font=font, fill="black"
        )
    return np.asarray(image, dtype=np.uint8)


@pytest.mark.parametrize("level", LEVELS)
def test_levels_keep_shape_and_type(level: str) -> None:
    result = degrade(_page(), level, random.Random(0))
    assert result.image.dtype == np.uint8
    assert result.image.ndim == 3
    assert result.level == level


def test_hard_is_visibly_different_and_records_parameters() -> None:
    page = _page()
    clean = degrade(page, "clean", random.Random(0))
    hard = degrade(page, "hard", random.Random(0))
    assert abs(float(clean.image.mean()) - float(page.mean())) < 3
    assert hard.image.shape != page.shape  # padded
    assert hard.angle_deg != 0 and hard.blur_px in (3, 5) and hard.jpeg_quality < 70


def test_same_seed_same_photo() -> None:
    page = _page()
    a = degrade(page, "mild", random.Random(9)).image
    b = degrade(page, "mild", random.Random(9)).image
    assert np.array_equal(a, b)
