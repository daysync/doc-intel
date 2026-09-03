from fastapi import APIRouter

from doc_intel.api.schemas import MetricsResponse

router = APIRouter()


@router.get("/metrics")
def metrics() -> MetricsResponse:
    """Quality metrics from the latest eval run. Populated from Stage 4."""
    return MetricsResponse(
        run_id=None,
        field_accuracy=None,
        retrieval_recall_at_5=None,
        faithfulness=None,
        cost_per_doc_usd=None,
        p95_latency_ms=None,
    )
