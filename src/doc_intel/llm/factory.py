"""Build the configured adapter. The only module that imports every provider."""

from pathlib import Path

from doc_intel.api.settings import Settings
from doc_intel.llm.anthropic import AnthropicLLM
from doc_intel.llm.base import LLM
from doc_intel.llm.errors import LLMError
from doc_intel.llm.log import CallLog
from doc_intel.llm.ollama import OllamaLLM
from doc_intel.llm.openai import OpenAILLM
from doc_intel.llm.recorded import RecordedLLM

PROVIDERS = ("anthropic", "openai", "ollama")


def build_live_llm(settings: Settings, provider: str, log: CallLog | None = None) -> LLM:
    match provider:
        case "anthropic":
            if settings.anthropic_api_key is None:
                raise LLMError("ANTHROPIC_API_KEY is not set")
            return AnthropicLLM(api_key=settings.anthropic_api_key.get_secret_value(), log=log)
        case "openai":
            if settings.openai_api_key is None:
                raise LLMError("OPENAI_API_KEY is not set")
            return OpenAILLM(api_key=settings.openai_api_key.get_secret_value(), log=log)
        case "ollama":
            return OllamaLLM(host=settings.ollama_host, log=log)
    raise LLMError(f"unknown LLM_PROVIDER {provider!r}; expected one of {PROVIDERS}")


def build_llm(settings: Settings, provider: str | None = None, log: CallLog | None = None) -> LLM:
    """Live adapter, or a RecordedLLM around it when fixtures are in play.

    ``LLM_RECORD=1`` records misses through the live adapter. Otherwise, when a fixtures
    directory exists, replay only: a miss raises so CI never spends money by accident.
    """
    provider = provider or settings.llm_provider
    fixtures = Path(settings.llm_fixtures_dir)
    if settings.llm_record:
        return RecordedLLM(fixtures, inner=build_live_llm(settings, provider, log), log=log)
    if settings.llm_replay:
        return RecordedLLM(fixtures, inner=None, priced_as=provider, log=log)
    return build_live_llm(settings, provider, log)
