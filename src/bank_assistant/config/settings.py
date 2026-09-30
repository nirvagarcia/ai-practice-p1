"""Typed runtime configuration loaded from the environment and `.env`."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, PositiveFloat, PositiveInt, SecretStr, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class ConfigurationError(RuntimeError):
    """Required configuration is missing or invalid."""


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_ignore_empty=True,
        extra="ignore",
        frozen=True,
    )

    openai_api_key: SecretStr | None = None

    router_model: str = "gpt-4.1-mini"
    generation_model: str = "gpt-4.1-mini"
    grader_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    evaluator_model: str = "gpt-4.1-mini"
    router_min_confidence: float = Field(default=0.6, ge=0, le=1)

    knowledge_base_path: Path = Path("data/knowledge_base")
    vectorstore_path: Path = Path("data/vectorstore")
    mocks_path: Path = Path("data/mocks")
    retrieval_top_k: PositiveInt = 4
    rag_strategy: Literal["rag-v1", "crag"] = "rag-v1"

    capture_content: bool = False
    max_input_chars: PositiveInt = 4000
    max_output_chars: PositiveInt = 4000
    enable_moderation: bool = False
    moderation_fail_closed: bool = True
    max_graph_steps: PositiveInt = 12
    max_retrieval_attempts: PositiveInt = 2
    max_tool_retries: int = Field(default=2, ge=0)
    llm_timeout_seconds: PositiveFloat = 30
    tool_timeout_seconds: PositiveFloat = 5
    max_tokens_per_request: PositiveInt = 4000
    max_total_tokens_per_request: PositiveInt = 64_000
    max_estimated_cost_usd: PositiveFloat | None = None
    input_cost_usd_per_million_tokens: PositiveFloat | None = None
    output_cost_usd_per_million_tokens: PositiveFloat | None = None
    pricing_version: str | None = None

    @model_validator(mode="after")
    def validate_pricing(self) -> "Settings":
        has_input_price = self.input_cost_usd_per_million_tokens is not None
        has_output_price = self.output_cost_usd_per_million_tokens is not None
        has_all_prices = has_input_price and has_output_price
        if has_input_price != has_output_price:
            raise ValueError("Configure both input and output token prices")
        if has_all_prices and not self.pricing_version:
            raise ValueError("PRICING_VERSION is required when token prices are configured")
        if self.max_estimated_cost_usd is not None and not has_all_prices:
            raise ValueError("Token prices are required to enforce MAX_ESTIMATED_COST_USD")
        return self

    def require_openai_api_key(self) -> SecretStr:
        if self.openai_api_key is None:
            raise ConfigurationError("OPENAI_API_KEY is not set. Add it to .env.")
        return self.openai_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
