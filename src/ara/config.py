"""Typed settings loaded from env vars + `.env`.

Prefix: `ARA_`. Example: `ARA_LLM_MODEL`, `ARA_GEMINI_API_KEY`.

Settings is constructed ONCE in cli.py and passed to the orchestrator.
No module-level caching (plays badly with pytest's monkeypatch); callers
use `load_settings()` every time a fresh read is needed.

Design notes:
    - Flat schema (10 fields, no nested submodels) — simpler for CLI/orchestrator.
    - `SecretStr` for API keys — repr-redacted so structlog / PipelineState
      dumps never leak credentials (pitfall P14).
    - `extra="ignore"` — unrelated env vars (e.g. `HOME`, `PATH`) do not raise.
    - `load_settings()` is a factory; orchestrator/CLI call it on entry.
    - `fake_llm_fixture` is the test-mode escape hatch — when set, the
      orchestrator constructs a `FakeLLM` reading that JSON file instead
      of a `LiteLLMClient`. Used by the Phase 1 smoke test (plan 09).
"""

from __future__ import annotations

from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Flat settings object — 10 fields, no nested submodels.

    All fields are overridable via `ARA_<UPPER_SNAKE>` env vars or a
    `.env` file in the CWD. See `.env.example` (plan 07) for the
    full surface and an annotated walkthrough.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="ARA_",
        extra="ignore",
    )

    llm_model: str = "gemini/gemini-2.5-flash"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    gemini_api_key: SecretStr | None = None
    openai_api_key: SecretStr | None = None

    runs_dir: str = "runs"
    prompts_dir: str = "prompts"

    log_level: str = "INFO"
    cached_only: bool = False

    retrieval_year_since: int | None = None
    retrieval_min_citations: int | None = None

    fake_llm_fixture: str | None = None


def load_settings() -> Settings:
    """Factory — `Settings()` but with room for future pre-processing."""
    return Settings()
