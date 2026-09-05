import random

import numpy as np

from doc_intel.dataset.content import generate_content
from doc_intel.dataset.layouts import render_pdf
from doc_intel.dataset.render import pdf_to_image
from doc_intel.ocr.preprocess import preprocess
from doc_intel.ocr.tesseract import detect_languages, ocr_page


def test_clean_english_invoice_is_read(clean_en_page: np.ndarray) -> None:
    rng = random.Random(21)
    truth = generate_content(rng, "en").invoice
    page = ocr_page(preprocess(clean_en_page).binary, languages="eng")
    assert truth.number.quote and truth.number.quote in page.text
    assert truth.totals.grand_total.quote and truth.totals.grand_total.quote in page.text
    assert page.mean_confidence > 70
    assert page.lines and all(line.bbox[2] > 0 for line in page.lines)


def test_script_detection_picks_language_packs(clean_en_page: np.ndarray) -> None:
    assert detect_languages(clean_en_page) == "eng"
    rng = random.Random(8)
    uk = pdf_to_image(render_pdf(generate_content(rng, "uk"), "classic", rng), dpi=150)
    assert detect_languages(uk) == "rus+ukr+eng"


def test_ukrainian_invoice_number_and_total_are_read() -> None:
    rng = random.Random(8)
    content = generate_content(rng, "uk")
    image = pdf_to_image(render_pdf(content, "classic", rng), dpi=200)
    page = ocr_page(preprocess(image).binary, languages="rus+ukr+eng")
    number, total = content.invoice.number.quote, content.invoice.totals.grand_total.quote
    assert number and total
    # Tesseract tends to read the Latin prefix of a number as a Cyrillic look-alike (F -> Е);
    # the digits survive. Fixing that is the extraction stage's job, not OCR's.
    assert number.split("-", 1)[1] in page.text, page.text
    assert total in page.text, page.text
    assert page.mean_confidence > 85
