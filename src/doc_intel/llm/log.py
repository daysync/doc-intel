"""The call ledger: every LLM call appends one CallRecord here."""

import logging
from decimal import Decimal

from doc_intel.llm.types import CallRecord

logger = logging.getLogger("doc_intel.llm")


class CallLog:
    def __init__(self) -> None:
        self.records: list[CallRecord] = []

    def append(self, record: CallRecord) -> None:
        self.records.append(record)
        logger.info(
            "llm.call provider=%s model=%s in=%d cached=%d out=%d cost_usd=%s latency_ms=%d",
            record.provider,
            record.model,
            record.input_tokens,
            record.cached_input_tokens,
            record.output_tokens,
            record.cost_usd,
            record.latency_ms,
        )

    def total_cost(self) -> Decimal:
        return sum((record.cost_usd for record in self.records), Decimal(0))

    def clear(self) -> None:
        self.records.clear()
