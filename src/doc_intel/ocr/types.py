"""What every OCR engine returns, whichever engine it is."""

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class OcrLine(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    bbox: tuple[int, int, int, int] = Field(description="left, top, width, height in pixels")
    confidence: float = Field(ge=0.0, le=100.0)


class OcrPage(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: str
    lines: list[OcrLine]
    mean_confidence: float = Field(ge=0.0, le=100.0)
    languages: str = Field(
        description="Tesseract language string used, e.g. 'rus+ukr', or the vision model"
    )


class OcrResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    pages: list[OcrPage]
    engine: Literal["tesseract", "vision"]
    cost_usd: Decimal = Decimal(0)

    @property
    def text(self) -> str:
        return "\n\n".join(page.text for page in self.pages)

    @property
    def mean_confidence(self) -> float:
        if not self.pages:
            return 0.0
        return sum(page.mean_confidence for page in self.pages) / len(self.pages)
