"""Settings / pydantic-settings tests (FND-04 env-only swap guarantee)."""

from __future__ import annotations

import os

import pytest
from pydantic import SecretStr


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Scrub ARA_* env vars and chdir to an empty dir so no stray .env is read."""
    for k in list(os.environ):
        if k.startswith("ARA_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)


def test_default_settings_values():
    from ara.config import Settings

    s = Settings()
    assert s.llm_model == "gemini/gemini-2.5-flash"
    assert s.llm_temperature == 0.0
    assert s.llm_max_tokens == 2048
    assert s.log_level == "INFO"
    assert s.cached_only is False
    assert s.runs_dir == "runs"
    assert s.prompts_dir == "prompts"
    assert s.gemini_api_key is None
    assert s.openai_api_key is None
    assert s.fake_llm_fixture is None


def test_env_override_llm_model(monkeypatch):
    from ara.config import Settings

    monkeypatch.setenv("ARA_LLM_MODEL", "openai/gpt-4o-mini")
    assert Settings().llm_model == "openai/gpt-4o-mini"


def test_env_override_numeric_coerced(monkeypatch):
    from ara.config import Settings

    monkeypatch.setenv("ARA_LLM_TEMPERATURE", "0.7")
    assert Settings().llm_temperature == 0.7


def test_api_keys_use_secret_str(monkeypatch):
    from ara.config import Settings

    monkeypatch.setenv("ARA_GEMINI_API_KEY", "AIzaTESTVALUE1234567890")
    s = Settings()
    assert isinstance(s.gemini_api_key, SecretStr)
    assert "AIzaTESTVALUE1234567890" not in repr(s)
    assert s.gemini_api_key.get_secret_value() == "AIzaTESTVALUE1234567890"


def test_api_keys_value_accessible_via_get_secret_value(monkeypatch):
    from ara.config import Settings

    monkeypatch.setenv("ARA_OPENAI_API_KEY", "sk-test-open-ai-value")
    s = Settings()
    assert isinstance(s.openai_api_key, SecretStr)
    assert s.openai_api_key.get_secret_value() == "sk-test-open-ai-value"


def test_extra_env_vars_ignored(monkeypatch):
    from ara.config import Settings

    monkeypatch.setenv("TOTALLY_UNRELATED", "xyz")
    Settings()


def test_load_settings_helper_works(monkeypatch):
    from ara.config import load_settings

    monkeypatch.setenv("ARA_LOG_LEVEL", "DEBUG")
    s = load_settings()
    assert s.log_level == "DEBUG"
