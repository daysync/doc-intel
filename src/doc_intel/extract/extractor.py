"""Turn a page (image and/or OCR text) into an Invoice through the LLM interface.

Three strategies, chosen by config and compared by eval:

* ``text``: OCR text only. Cheapest; works with text-only models. Inherits OCR errors.
* ``image``: the page image only. Needs a vision model; ignores OCR entirely.
* ``both``: image plus OCR text as a hint. Usually best; costs the most tokens.

When the answer fails schema validation, one repair round shows the model its own answer
and the validation errors. Small models need this often; large ones rarely.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import NDArray

from doc_intel.extract.prompts import (
    BOTH_INSTRUCTION,
    IMAGE_INSTRUCTION,
    REPAIR_INSTRUCTION,
    SYSTEM,
    TEXT_INSTRUCTION,
)
from doc_intel.llm.base import LLM
from doc_intel.llm.errors import StructuredOutputError
from doc_intel.llm.types import CallRecord, ImagePart, LLMRequest, Message, Part, TextPart
from doc_intel.models import Invoice
from doc_intel.ocr.image import to_jpeg

Strategy = Literal["text", "image", "both"]


@dataclass
class Extraction:
    invoice: Invoice
    strategy: Strategy
    records: list[CallRecord] = field(default_factory=list)
    repairs: int = 0

    @property
    def record(self) -> CallRecord:
        return self.records[-1]


class Extractor:
    def __init__(
        self,
        llm: LLM,
        model: str,
        strategy: Strategy = "both",
        max_tokens: int = 4096,
        max_repairs: int = 1,
    ) -> None:
        self.llm = llm
        self.model = model
        self.strategy = strategy
        self.max_tokens = max_tokens
        self.max_repairs = max_repairs

    def build_request(self, image: NDArray[np.uint8] | None, ocr_text: str | None) -> LLMRequest:
        parts: list[Part]
        if self.strategy == "text":
            if ocr_text is None:
                raise ValueError("text strategy needs OCR text")
            parts = [TextPart(text=TEXT_INSTRUCTION + ocr_text)]
        elif self.strategy == "image":
            if image is None:
                raise ValueError("image strategy needs an image")
            parts = [
                ImagePart(data=to_jpeg(image), mime="image/jpeg"),
                TextPart(text=IMAGE_INSTRUCTION),
            ]
        else:
            if image is None or ocr_text is None:
                raise ValueError("both strategy needs an image and OCR text")
            parts = [
                ImagePart(data=to_jpeg(image), mime="image/jpeg"),
                TextPart(text=BOTH_INSTRUCTION + ocr_text),
            ]
        return LLMRequest(
            model=self.model,
            system=SYSTEM,
            messages=[Message(role="user", parts=parts)],
            max_tokens=self.max_tokens,
        )

    async def extract(self, image: NDArray[np.uint8] | None, ocr_text: str | None) -> Extraction:
        """One attempt, then up to ``max_repairs`` rounds that show the model its validation errors.

        The repair turn keeps the original request and appends the failed answer as an
        assistant message plus the errors as a user message, so the model corrects rather
        than restarts. Every round is a separate call with its own CallRecord.
        """
        request = self.build_request(image, ocr_text)
        for attempt in range(self.max_repairs + 1):
            try:
                response = await self.llm.complete(request, Invoice)
            except StructuredOutputError as error:
                if attempt >= self.max_repairs:
                    raise
                request = self._repair_request(request, error)
                continue
            records = self.llm.log.records[-(attempt + 1) :]
            return Extraction(
                invoice=response.output,
                strategy=self.strategy,
                records=list(records),
                repairs=attempt,
            )
        raise AssertionError("unreachable")

    @staticmethod
    def _repair_request(request: LLMRequest, error: StructuredOutputError) -> LLMRequest:
        problems = "\n".join(
            f"- {'.'.join(str(part) for part in item['loc'])}: {item['msg']} "
            f"(got {item.get('input')!r})"
            for item in error.validation_error.errors()[:20]
        )
        return request.model_copy(
            update={
                "messages": [
                    *request.messages,
                    Message(role="assistant", parts=[TextPart(text=error.raw_text)]),
                    Message(role="user", parts=[TextPart(text=REPAIR_INSTRUCTION + problems)]),
                ]
            }
        )
