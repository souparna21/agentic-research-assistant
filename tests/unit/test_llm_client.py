"""Tests for LLMClient / LiteLLMClient / FakeLLM (FND-04, FND-05).

litellm.completion is always monkey-patched — pytest-socket blocks real HTTP anyway
(belt and suspenders). No network calls in this test file.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch, tmp_path):
    """Scrub ARA_* env vars so tests are isolated from host env."""
    for k in list(os.environ):
        if k.startswith("ARA_"):
            monkeypatch.delenv(k, raising=False)
    monkeypatch.chdir(tmp_path)


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _make_litellm_response(text: str = "hi", pt: int = 3, ct: int = 2):
    """Build a SimpleNamespace mirroring litellm.completion's return shape."""
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))],
        usage=SimpleNamespace(prompt_tokens=pt, completion_tokens=ct),
    )



def test_fake_llm_lookup(tmp_path: Path):
    from ara.services.llm import FakeLLM

    prompt = "Expand: What is RAG?"
    fixture = tmp_path / "fake.json"
    fixture.write_text(json.dumps({_hash(prompt): "RAG; retrieval-augmented generation"}))
    fake = FakeLLM(fixture)
    resp = fake.complete(prompt)
    assert "RAG" in resp.text
    assert resp.model == "fake"
    assert resp.prompt_tokens == 0
    assert fake.calls == [(prompt, 0.0)]


def test_fake_llm_raises_on_unknown(tmp_path: Path):
    from ara.errors import UnknownPromptError
    from ara.services.llm import FakeLLM

    fixture = tmp_path / "fake.json"
    fixture.write_text("{}")
    fake = FakeLLM(fixture)
    with pytest.raises(UnknownPromptError) as excinfo:
        fake.complete("a totally unknown prompt with many words " * 3)
    msg = str(excinfo.value)
    assert "prompt hash" in msg
    assert "totally unknown prompt" in msg


def test_fake_llm_records_temperature(tmp_path: Path):
    from ara.services.llm import FakeLLM

    prompt = "p"
    fixture = tmp_path / "fake.json"
    fixture.write_text(json.dumps({_hash(prompt): "ok"}))
    fake = FakeLLM(fixture)
    fake.complete(prompt, temperature=0.5)
    assert fake.calls[-1] == (prompt, 0.5)


def test_fake_llm_returns_llm_response_with_model_fake(tmp_path: Path):
    from ara.services.llm import FakeLLM, LLMResponse

    prompt = "ping"
    fixture = tmp_path / "fake.json"
    fixture.write_text(json.dumps({_hash(prompt): "pong"}))
    fake = FakeLLM(fixture)
    resp = fake.complete(prompt)
    assert isinstance(resp, LLMResponse)
    assert resp.text == "pong"
    assert resp.model == "fake"
    assert resp.prompt_tokens == 0
    assert resp.completion_tokens == 0



def test_litellm_default_model():
    from ara.services.llm import LiteLLMClient

    c = LiteLLMClient(model="gemini/gemini-2.5-flash")
    assert c.model == "gemini/gemini-2.5-flash"
    assert c.max_tokens == 2048


def test_litellm_complete_calls_litellm_completion():
    from ara.services.llm import LiteLLMClient

    with patch("ara.services.llm.litellm") as mock_litellm:
        mock_litellm.completion = MagicMock(return_value=_make_litellm_response("reply"))
        c = LiteLLMClient(model="gemini/gemini-2.5-flash", max_tokens=2048)
        resp = c.complete("hello")
        mock_litellm.completion.assert_called_once()
        call_kwargs = mock_litellm.completion.call_args.kwargs
        assert call_kwargs["model"] == "gemini/gemini-2.5-flash"
        assert call_kwargs["messages"] == [{"role": "user", "content": "hello"}]
        assert call_kwargs["temperature"] == 0.0
        assert call_kwargs["max_tokens"] == 2048
        assert call_kwargs.get("response_format") is None
        assert resp.text == "reply"
        assert resp.model == "gemini/gemini-2.5-flash"
        assert resp.prompt_tokens == 3
        assert resp.completion_tokens == 2


def test_litellm_schema_kwarg_triggers_json_response_format():
    from ara.services.llm import LiteLLMClient

    with patch("ara.services.llm.litellm") as mock_litellm:
        mock_litellm.completion = MagicMock(return_value=_make_litellm_response("{}"))
        c = LiteLLMClient(model="gemini/gemini-2.5-flash")
        c.complete("p", schema={"type": "object"})
        rf = mock_litellm.completion.call_args.kwargs["response_format"]
        assert rf == {"type": "json_object"}



def test_provider_swap_via_config(monkeypatch):
    """FND-04: changing ARA_LLM_MODEL env + rebuilding Settings is sufficient."""
    monkeypatch.setenv("ARA_LLM_MODEL", "openai/gpt-4o-mini")
    monkeypatch.setenv("ARA_OPENAI_API_KEY", "sk-test")
    from ara.config import load_settings
    from ara.services.llm import LiteLLMClient

    s = load_settings()
    assert s.llm_model == "openai/gpt-4o-mini"
    c = LiteLLMClient(model=s.llm_model)
    assert c.model == "openai/gpt-4o-mini"
