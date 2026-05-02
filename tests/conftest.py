"""Shared pytest fixtures for the ara test suite.

pytest-socket blocks all network access by default (configured via
pyproject.toml addopts `--disable-socket`). Tests that need sockets
can opt in via @pytest.mark.enable_socket (or `enable_socket` fixture
from pytest-socket).

Fixtures:
    tmp_runs_dir    — a throwaway runs/ directory for the test.
    fake_llm_factory — builds a FakeLLM from an in-memory dict of prompt hashes.
    sample_state    — a populated PipelineState for quick tests.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Callable

import pytest


@pytest.fixture
def tmp_runs_dir(tmp_path: Path) -> Path:
    """A throwaway runs/ directory scoped to one test."""
    d = tmp_path / "runs"
    d.mkdir(parents=True, exist_ok=True)
    return d


@pytest.fixture
def fake_llm_factory(tmp_path: Path) -> Callable[[dict[str, str]], "FakeLLMProto"]:
    """Return a factory that writes responses to a JSON fixture and builds a FakeLLM.

    The FakeLLM class itself is defined in src/ara/services/llm.py (plan 02).
    This factory is a convenience so every test doesn't re-write the fixture file.

    Usage (inside a test, once plan 02 lands FakeLLM):
        >>> from ara.services.llm import FakeLLM
        >>> fake = fake_llm_factory({"expand rag": "term1; term2"})
    """
    def _build(responses: dict[str, str]):  # pragma: no cover - wired in plan 02
        from ara.services.llm import FakeLLM  # type: ignore[import-not-found]

        fixture = tmp_path / "fake_llm.json"
        keyed = {
            hashlib.sha256(prompt.encode()).hexdigest()[:16]: resp
            for prompt, resp in responses.items()
        }
        fixture.write_text(json.dumps(keyed))
        return FakeLLM(fixture)

    return _build  # type: ignore[return-value]


class FakeLLMProto:  # pragma: no cover - protocol shim only
    """Forward-declared type alias; real FakeLLM in plan 02."""


@pytest.fixture
def sample_state():
    """A populated PipelineState for tests that just need 'any valid state'.

    PipelineState itself is defined in src/ara/state.py (plan 01). This fixture
    imports lazily so conftest can be collected before plan 01 lands.
    """
    from ara.state import PipelineState  # type: ignore[import-not-found]

    return PipelineState(
        run_id="20260417-120000-test",
        question="What is RAG?",
        config_snapshot={
            "model": "gemini/gemini-2.5-flash",
            "temperature": 0.0,
            "git_sha": "abc1234",
        },
    )
