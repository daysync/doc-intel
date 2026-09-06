"""The whole pipeline behind one method: bytes in, ProcessResult out.

``Pipeline.from_config("configs/default.yaml")`` wires the stages from a YAML file so that
evals can compare configurations without code changes. ``process()`` runs preprocess → OCR
(with vision fallback) → extraction (with repair) → validation and returns the invoice,
the issues, the cost and per-stage timings. ``cross_check()`` runs the batch rules.
"""

import time
from collections.abc import Mapping
from datetime import date
from decimal import Decimal
from pathlib import Path
from typing import Literal
from uuid import uuid4

import yaml
from pydantic import BaseModel, ConfigDict

from doc_intel.api.settings import Settings, get_settings
from doc_intel.extract.extractor import Extractor
from doc_intel.extract.rules import cross_document_issues, validate_invoice
from doc_intel.llm.base import LLM
from doc_intel.llm.factory import build_llm
from doc_intel.models import ProcessResult, Timings, ValidationIssue
from doc_intel.ocr.engine import Ocr
from doc_intel.ocr.image import load_pages
from doc_intel.ocr.vision import VisionTranscriber


class LLMConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = "ollama"
    model: str = "qwen2.5vl:3b"


class OcrConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    fallback_below: float = 60.0
    psm: int = 3
    vision_model: str | None = None


class ExtractionConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    strategy: Literal["text", "image", "both"] = "both"
    max_tokens: int = 4096
    max_repairs: int = 1


class EmbeddingsConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    provider: str = "ollama"
    model: str = "nomic-embed-text"
    dimensions: int = 768


class PipelineConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str = "default"
    llm: LLMConfig = LLMConfig()
    ocr: OcrConfig = OcrConfig()
    extraction: ExtractionConfig = ExtractionConfig()
    embeddings: EmbeddingsConfig = EmbeddingsConfig()

    @classmethod
    def from_yaml(cls, path: str | Path) -> "PipelineConfig":
        return cls.model_validate(yaml.safe_load(Path(path).read_text()) or {})


class Pipeline:
    def __init__(self, config: PipelineConfig, llm: LLM, today: date | None = None) -> None:
        self.config = config
        self.llm = llm
        self.today = today
        vision = VisionTranscriber(llm, config.ocr.vision_model or config.llm.model)
        self.ocr = Ocr(vision=vision, fallback_below=config.ocr.fallback_below, psm=config.ocr.psm)
        self.extractor = Extractor(
            llm,
            config.llm.model,
            config.extraction.strategy,
            max_tokens=config.extraction.max_tokens,
            max_repairs=config.extraction.max_repairs,
        )

    @classmethod
    def from_config(
        cls, path: str | Path, settings: Settings | None = None, llm: LLM | None = None
    ) -> "Pipeline":
        config = PipelineConfig.from_yaml(path)
        settings = settings or get_settings()
        return cls(config, llm or build_llm(settings, config.llm.provider))

    async def process(
        self, data: bytes, mime: str, document_id: str | None = None
    ) -> ProcessResult:
        document_id = document_id or uuid4().hex
        clock = time.perf_counter()
        pages = load_pages(data, mime)
        cost_before = self.llm.log.total_cost()

        ocr_started = time.perf_counter()
        ocr = await self.ocr.run(pages)
        ocr_ms = _ms(ocr_started)

        extract_started = time.perf_counter()
        extraction = await self.extractor.extract(pages[0], ocr.text)
        extract_ms = _ms(extract_started)

        validate_started = time.perf_counter()
        issues = validate_invoice(extraction.invoice, today=self.today)
        validate_ms = _ms(validate_started)

        return ProcessResult(
            document_id=document_id,
            invoice=extraction.invoice,
            issues=issues,
            cost_usd=Decimal(self.llm.log.total_cost() - cost_before),
            timings=Timings(
                ocr_ms=ocr_ms, extract_ms=extract_ms, validate_ms=validate_ms, total_ms=_ms(clock)
            ),
            ocr_pages=[page.text for page in ocr.pages],
            ocr_engine=ocr.engine,
            model=self.config.llm.model,
            repairs=extraction.repairs,
        )

    def cross_check(self, results: Mapping[str, ProcessResult]) -> dict[str, list[ValidationIssue]]:
        return cross_document_issues({doc_id: result.invoice for doc_id, result in results.items()})


def _ms(since: float) -> int:
    return int((time.perf_counter() - since) * 1000)
