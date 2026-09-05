from io import BytesIO

import numpy as np
import pytest
from PIL import Image

from doc_intel.ocr.image import load_pages, to_jpeg


def _png(width: int = 40, height: int = 30) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (width, height), (10, 200, 30)).save(buffer, format="PNG")
    return buffer.getvalue()


def test_png_loads_as_one_rgb_page() -> None:
    pages = load_pages(_png(), "image/png")
    assert len(pages) == 1
    assert pages[0].shape == (30, 40, 3)
    assert tuple(pages[0][0, 0]) == (10, 200, 30)


def test_jpeg_exif_orientation_is_applied() -> None:
    image = Image.new("RGB", (40, 30), "white")
    exif = image.getexif()
    exif[0x0112] = 6  # rotate 90° clockwise
    buffer = BytesIO()
    image.save(buffer, format="JPEG", exif=exif.tobytes())
    pages = load_pages(buffer.getvalue(), "image/jpeg")
    assert pages[0].shape[:2] == (40, 30)


def test_pdf_pages_are_rendered(clean_en_pdf: bytes) -> None:
    pages = load_pages(clean_en_pdf, "application/pdf", dpi=72)
    assert len(pages) == 1
    assert pages[0].shape[2] == 3 and pages[0].shape[0] > 700


def test_unknown_mime_is_rejected() -> None:
    with pytest.raises(ValueError, match="unsupported"):
        load_pages(b"", "image/gif")


def test_to_jpeg_round_trips() -> None:
    page = np.full((20, 20, 3), 128, np.uint8)
    assert load_pages(to_jpeg(page), "image/jpeg")[0].shape == (20, 20, 3)
