"""Vision-model transcription: the fallback when Tesseract is not confident.

Any ``LLM`` adapter that accepts images works, including the local Ollama model, so the
fallback costs nothing during development and is measured against Tesseract on the same
dataset. The model reports its own confidence; unlike Tesseract's per-word numbers it is a
self-assessment, which is why the two are never averaged together.
"""

from typing import ClassVar

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field

from doc_intel.llm.base import LLM
from doc_intel.llm.types import ImagePart, LLMRequest, Message, TextPart
from doc_intel.ocr.image import to_jpeg
from doc_intel.ocr.types import OcrLine, OcrPage

SYSTEM = (
    "You are a transcription engine for photographed business documents. Transcribe every "
    "piece of text you can read, one line of the document per line of output, in reading "
    "order, preserving the original language, digits and punctuation exactly. Do not translate, "
    "summarise or add anything. Then give the document's language as an ISO 639-1 code and "
    "your confidence in the transcription between 0 and 1."
)


class Transcript(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    language: str
    confidence: float = Field(ge=0.0, le=1.0)


class VisionTranscriber:
    engine: ClassVar[str] = "vision"

    def __init__(self, llm: LLM, model: str, max_tokens: int = 4096) -> None:
        self.llm = llm
        self.model = model
        self.max_tokens = max_tokens

    async def transcribe(self, image: NDArray[np.uint8]) -> OcrPage:
        request = LLMRequest(
            model=self.model,
            system=SYSTEM,
            messages=[
                Message(
                    role="user",
                    parts=[
                        ImagePart(data=to_jpeg(image), mime="image/jpeg"),
                        TextPart(text="Transcribe this document."),
                    ],
                )
            ],
            max_tokens=self.max_tokens,
        )
        response = await self.llm.complete(request, Transcript)
        transcript = response.output
        height, width = image.shape[:2]
        lines = [
            OcrLine(text=line, bbox=(0, 0, width, height), confidence=transcript.confidence * 100)
            for line in transcript.text.splitlines()
            if line.strip()
        ]
        return OcrPage(
            text="\n".join(line.text for line in lines),
            lines=lines,
            mean_confidence=transcript.confidence * 100,
            languages=f"{self.model}:{transcript.language}",
        )
