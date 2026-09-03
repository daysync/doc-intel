"""Runtime configuration, read from environment variables and an optional ``.env`` file."""

from functools import lru_cache

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    anthropic_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None
    ollama_host: str = "http://localhost:11434"
    llm_record: bool = False
    database_url: str = "postgresql+psycopg://doc_intel:doc_intel@localhost:5433/doc_intel"


@lru_cache
def get_settings() -> Settings:
    return Settings()
