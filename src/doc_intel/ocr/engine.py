"""Run Tesseract; fall back to the vision model when confidence is low."""

from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.typing import NDArray

from doc_intel.ocr.preprocess import preprocess
from doc_intel.ocr.tesseract import ocr_page
from doc_intel.ocr.types import OcrPage, OcrResult
from doc_intel.ocr.vision import VisionTranscriber


@dataclass
class Ocr:
    vision: VisionTranscriber | None = None
    fallback_below: float = 60.0
    """Mean Tesseract confidence (0-100) under which the vision model takes over."""
    psm: int = 3

    async def run(self, pages: list[NDArray[np.uint8]]) -> OcrResult:
        tesseract_pages: list[OcrPage] = []
        for page in pages:
            prepared = preprocess(page)
            tesseract_pages.append(ocr_page(prepared.binary, psm=self.psm))
        result = OcrResult(pages=tesseract_pages, engine="tesseract")

        if self.vision is None or result.mean_confidence >= self.fallback_below:
            return result

        cost_before = self.vision.llm.log.total_cost()
        vision_pages = [await self.vision.transcribe(page) for page in pages]
        cost = self.vision.llm.log.total_cost() - cost_before
        return OcrResult(pages=vision_pages, engine="vision", cost_usd=Decimal(cost))
