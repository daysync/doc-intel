"""Request and response bodies for the HTTP API.

These are separate from the core models on purpose: the wire format can add ids, timestamps
and pagination without leaking into ``Invoice`` or ``ProcessResult``.
"""

from datetime import datetime
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field

from doc_intel.api.jobs import JobStatus
from doc_intel.models import Invoice, ValidationIssue


class IngestResponse(BaseModel):
    job_id: str
    status: JobStatus


class DocumentOut(BaseModel):
    id: str
    filename: str
    status: JobStatus
    created_at: datetime
    invoice: Invoice | None
    issues: list[ValidationIssue]
    cost_usd: Decimal | None
    error: str | None


class DocumentsResponse(BaseModel):
    documents: list[DocumentOut]


class IssueOut(BaseModel):
    document_id: str
    issue: ValidationIssue


class IssuesResponse(BaseModel):
    issues: list[IssueOut]


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=2000)


class Citation(BaseModel):
    document_id: str
    page: int
    snippet: str


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation]
    cost_usd: Decimal


class MetricsResponse(BaseModel):
    """Mirrors the results table in the README. All ``None`` until the first eval run."""

    run_id: str | None
    field_accuracy: float | None
    retrieval_recall_at_5: float | None
    faithfulness: float | None
    cost_per_doc_usd: Decimal | None
    p95_latency_ms: int | None
