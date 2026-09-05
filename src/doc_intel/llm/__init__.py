"""Provider adapters, structured output, cost and latency logging.

Public surface: ``LLM`` (the interface), ``LLMRequest``/``LLMResponse``/``CallRecord``
(the types), ``CallLog`` (the ledger), ``Pricing`` (the price table), and the errors.
Adapters live in their own modules and are wired by ``factory.build_llm``.
"""

from doc_intel.llm.base import LLM
from doc_intel.llm.errors import (
    FixtureMissingError,
    LLMError,
    ProviderError,
    StructuredOutputError,
    UnknownModelPriceError,
)
from doc_intel.llm.log import CallLog
from doc_intel.llm.pricing import Pricing
from doc_intel.llm.types import (
    CallRecord,
    ImagePart,
    LLMRequest,
    LLMResponse,
    Message,
    RawCompletion,
    TextPart,
)

__all__ = [
    "LLM",
    "CallLog",
    "CallRecord",
    "FixtureMissingError",
    "ImagePart",
    "LLMError",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "Pricing",
    "ProviderError",
    "RawCompletion",
    "StructuredOutputError",
    "TextPart",
    "UnknownModelPriceError",
]
