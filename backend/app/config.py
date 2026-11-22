import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    database_url: str
    openai_base_url: str | None
    openai_api_key: str
    chat_model: str
    embedding_model: str
    chat_input_usd_per_million: float | None
    chat_output_usd_per_million: float | None
    embedding_usd_per_million: float | None


def _optional_float(name: str) -> float | None:
    value = os.getenv(name)
    return float(value) if value else None


def get_settings() -> Settings:
    return Settings(
        database_url=os.getenv("DATABASE_URL", "postgresql+psycopg://rag@db:5432/rag"),
        openai_base_url=os.getenv("OPENAI_BASE_URL") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY", ""),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        embedding_model=os.getenv("EMBEDDING_MODEL", "text-embedding-3-small"),
        chat_input_usd_per_million=_optional_float("CHAT_INPUT_USD_PER_MILLION"),
        chat_output_usd_per_million=_optional_float("CHAT_OUTPUT_USD_PER_MILLION"),
        embedding_usd_per_million=_optional_float("EMBEDDING_USD_PER_MILLION"),
    )
