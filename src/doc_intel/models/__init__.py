"""Core data models shared by the API, the extraction pipeline, and the evals."""

from doc_intel.models.fields import Extracted, iter_extracted

__all__ = ["Extracted", "iter_extracted"]
