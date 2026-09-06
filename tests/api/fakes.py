"""A processor that never touches OCR or a model: returns a canned ProcessResult, or fails."""

from collections.abc import Mapping
from decimal import Decimal

from doc_intel.extract.rules import cross_document_issues
from doc_intel.models import ProcessResult, Timings, ValidationIssue
from tests.models.factories import invoice


class FakeProcessor:
    def __init__(self, fail_on: bytes | None = b"BOOM") -> None:
        self.fail_on = fail_on
        self.calls: list[tuple[str, str | None]] = []

    async def process(
        self, data: bytes, mime: str, document_id: str | None = None
    ) -> ProcessResult:
        self.calls.append((mime, document_id))
        if self.fail_on is not None and data == self.fail_on:
            raise ValueError("unreadable document")
        return ProcessResult(
            document_id=document_id or "doc",
            invoice=invoice(),
            issues=[],
            cost_usd=Decimal("0.0042"),
            timings=Timings(ocr_ms=1, extract_ms=2, validate_ms=0, total_ms=3),
            ocr_engine="tesseract",
            model="fake",
        )

    def cross_check(self, results: Mapping[str, ProcessResult]) -> dict[str, list[ValidationIssue]]:
        return cross_document_issues({doc_id: result.invoice for doc_id, result in results.items()})


class FakeIndexer:
    def __init__(self) -> None:
        self.indexed: list[str] = []

    async def index(self, result: ProcessResult) -> None:
        self.indexed.append(result.document_id)
