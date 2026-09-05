"""Run Tesseract; fall back to the vision model when confidence is low.

A failed vision call (garbage output, provider error) is not fatal: the Tesseract result is
returned with ``fallback_error`` set, so a weak model degrades a document instead of losing it.
"""

import logging
from dataclasses import dataclass
from decimal import Decimal

import numpy as np
from numpy.typing import NDArray

from doc_intel.llm.errors import LLMError
from doc_intel.ocr.preprocess import preprocess
from doc_intel.ocr.tesseract import ocr_page
from doc_intel.ocr.types import OcrPage, OcrResult
from doc_intel.ocr.vision import VisionTranscriber

logger = logging.getLogger("doc_intel.ocr")


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
        try:
            vision_pages = [await self.vision.transcribe(page) for page in pages]
        except LLMError as error:
            logger.warning("vision fallback failed, keeping Tesseract result: %s", error)
            cost = self.vision.llm.log.total_cost() - cost_before
            return result.model_copy(
                update={"cost_usd": Decimal(cost), "fallback_error": str(error)[:200]}
            )
        cost = self.vision.llm.log.total_cost() - cost_before
        return OcrResult(pages=vision_pages, engine="vision", cost_usd=Decimal(cost))
