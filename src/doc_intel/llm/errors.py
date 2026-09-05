"""Errors raised by the LLM layer. Provider SDK errors are wrapped, never leaked."""

from pathlib import Path

from pydantic import ValidationError


class LLMError(Exception):
    """Base class for everything raised by this package."""


class ProviderError(LLMError):
    """The provider refused or failed the request (auth, rate limit, 5xx, network)."""

    def __init__(self, provider: str, message: str, status: int | None = None) -> None:
        super().__init__(f"{provider}: {message}" + (f" (HTTP {status})" if status else ""))
        self.provider = provider
        self.status = status


class StructuredOutputError(LLMError):
    """The model answered, but the JSON did not validate against the output model."""

    def __init__(self, raw_text: str, validation_error: ValidationError) -> None:
        super().__init__(f"model output failed validation: {validation_error}")
        self.raw_text = raw_text
        self.validation_error = validation_error


class UnknownModelPriceError(LLMError):
    """A paid provider returned usage for a model with no entry in the price table."""

    def __init__(self, provider: str, model: str) -> None:
        super().__init__(f"no price configured for {provider}/{model}; add it to llm/pricing.py")


class FixtureMissingError(LLMError):
    """Recorded adapter has no fixture and no inner adapter to record from."""

    def __init__(self, key: str, path: Path) -> None:
        super().__init__(f"no fixture {path.name} for request {key[:12]}…; run with LLM_RECORD=1")
        self.key = key
        self.path = path
