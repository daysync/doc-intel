"""Tesseract backend: free, fast, gives a confidence per word.

Language matters: running every language at once is slow and hurts accuracy, so the
script is detected first with Tesseract's orientation-and-script module and mapped to the
right language pack(s). Cyrillic gets Russian and Ukrainian together because the two
share most glyphs and the document could be either. English rides along with every
non-Latin script because invoice numbers, currency codes and product names are Latin.
"""

import re
from collections import defaultdict

import numpy as np
import pytesseract
from numpy.typing import NDArray

from doc_intel.ocr.types import OcrLine, OcrPage

SCRIPT_TO_LANGS = {"Latin": "eng", "Cyrillic": "rus+ukr+eng", "Georgian": "kat+eng"}
ALL_LANGS = "eng+rus+ukr+kat"


def detect_languages(image: NDArray[np.uint8]) -> str:
    """Map the dominant script to Tesseract language packs; fall back to all four."""
    try:
        osd = pytesseract.image_to_osd(image)
    except pytesseract.TesseractError:
        return ALL_LANGS
    match = re.search(r"Script: (\w+)", osd)
    if not match:
        return ALL_LANGS
    return SCRIPT_TO_LANGS.get(match.group(1), ALL_LANGS)


def ocr_page(image: NDArray[np.uint8], languages: str | None = None, psm: int = 3) -> OcrPage:
    """Words with boxes and confidences, grouped into lines in reading order."""
    languages = languages or detect_languages(image)
    data = pytesseract.image_to_data(
        image, lang=languages, config=f"--psm {psm}", output_type=pytesseract.Output.DICT
    )
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, word in enumerate(data["text"]):
        if not str(word).strip() or float(data["conf"][index]) < 0:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        grouped[key].append(index)

    lines: list[OcrLine] = []
    for key in sorted(grouped):
        indices = grouped[key]
        left = min(int(data["left"][i]) for i in indices)
        top = min(int(data["top"][i]) for i in indices)
        right = max(int(data["left"][i]) + int(data["width"][i]) for i in indices)
        bottom = max(int(data["top"][i]) + int(data["height"][i]) for i in indices)
        confidences = [float(data["conf"][i]) for i in indices]
        lines.append(
            OcrLine(
                text=" ".join(str(data["text"][i]) for i in indices),
                bbox=(left, top, right - left, bottom - top),
                confidence=sum(confidences) / len(confidences),
            )
        )

    words = [float(data["conf"][i]) for indices in grouped.values() for i in indices]
    mean = sum(words) / len(words) if words else 0.0
    return OcrPage(
        text="\n".join(line.text for line in lines),
        lines=lines,
        mean_confidence=mean,
        languages=languages,
    )
