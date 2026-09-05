"""What the pipeline returns for one document, and how problems are reported."""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from doc_intel.models.fields import iter_extracted
from doc_intel.models.invoice import Invoice


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class ValidationIssue(BaseModel):
    """One inconsistency found by a field rule or a cross-document check."""

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="Stable machine code, e.g. totals_mismatch, duplicate_number")
    severity: Severity
    message: str = Field(description="One plain sentence for a reviewer")
    field: str | None = Field(default=None, description="Dotted path such as totals.grand_total")
    expected: str | None = None
    actual: str | None = None
    related_document_ids: list[str] = Field(default_factory=list)


class Timings(BaseModel):
    """Milliseconds per stage. ``None`` means the stage was skipped."""

    model_config = ConfigDict(extra="forbid")

    preprocess_ms: int | None = None
    ocr_ms: int | None = None
    extract_ms: int | None = None
    validate_ms: int | None = None
    total_ms: int


class ProcessResult(BaseModel):
    """The validated invoice plus everything a host needs to decide what to do with it."""

    model_config = ConfigDict(extra="forbid")

    document_id: str
    invoice: Invoice
    issues: list[ValidationIssue]
    cost_usd: Decimal
    timings: Timings

    @property
    def confidence(self) -> dict[str, float]:
        """Flat ``{"number": 0.9, "line_items[0].quantity": 0.6, ...}`` for a review screen."""
        return {path: field.confidence for path, field in iter_extracted(self.invoice)}

    @property
    def has_errors(self) -> bool:
        return any(issue.severity is Severity.ERROR for issue in self.issues)
