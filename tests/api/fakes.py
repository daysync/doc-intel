"""A processor that never touches OCR or a model: returns a canned ProcessResult, or fails."""

from collections.abc import Mapping
from decimal import Decimal

from doc_intel.extract.rules import cross_document_issues
from doc_intel.models import ProcessResult, Timings, ValidationIssue
from doc_intel.rag.qa import QaResult
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


class FakeQa:
    """Answers one canned question with a citation; everything else is not in the documents."""

    async def ask(self, question: str, k: int | None = None) -> QaResult:
        from doc_intel.rag.answer import NOT_IN_DOCUMENTS, VerifiedCitation

        if "INV-1042" in question:
            citation = VerifiedCitation(
                chunk_id=7,
                document_id="doc-a",
                page=1,
                kind="totals",
                snippet="grand total 78.60 EUR",
            )
            return QaResult(
                "The grand total is 78.60 EUR.",
                [citation],
                False,
                True,
                0.9,
                ["doc-a"],
                [],
                Decimal("0.001"),
            )
        return QaResult(NOT_IN_DOCUMENTS, [], True, True, 0.95, None, [], Decimal(0))
