"""LLM client layer: LiteLLMClient (real) + FakeLLM (test double).

Both conform to the `LLMClient` ABC. Selection happens in the orchestrator / CLI::

    client = (
        FakeLLM(Path(s.fake_llm_fixture))
        if s.fake_llm_fixture
        else LiteLLMClient(model=s.llm_model, max_tokens=s.llm_max_tokens)
    )

`FakeLLM` raises `UnknownPromptError` on a prompt whose SHA-256[:16] hex
prefix is not in the fixture — this is deliberate: silent empty responses
would hide prompt drift (pitfall P18). The fixture file is a JSON object
keyed by prompt-hash, value = canned response text.

`LiteLLMClient` wraps `litellm.completion(...)` so *any* LiteLLM-supported
provider is available through a single config string. Swapping Gemini →
OpenAI is a pure env change (FND-04): set `ARA_LLM_MODEL=openai/gpt-4o-mini`
and `ARA_OPENAI_API_KEY=...`.
"""

from __future__ import annotations

import hashlib
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import litellm
from pydantic import BaseModel

from ara.errors import UnknownPromptError


@dataclass(frozen=True)
class LLMResponse:
    """Unified return shape regardless of provider.

    Frozen to prevent accidental mutation by downstream consumers. Token
    counts are 0 for `FakeLLM` (it has no real token counting).
    """

    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int


class LLMClient(ABC):
    """Abstract base: a minimal one-shot completion interface.

    The orchestrator and agents depend only on this ABC — not on LiteLLM
    or any concrete provider. That's the single shim point (FND-04).
    """

    @abstractmethod
    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponse:
        """Run one completion.

        Args:
            prompt: The user-role message content.
            temperature: Sampling temperature (default 0.0 for determinism).
            schema: When non-None, forces JSON response mode.
                - ``type[BaseModel]`` subclass: Phase 3 structured-output path —
                  converts to ``response_format={"type": "json_schema", ...}``
                  via ``schema.model_json_schema()``. Gemini 2.0+ / OpenAI
                  honor ``responseJsonSchema`` via LiteLLM.
                - ``dict``: Phase 1 back-compat — routes to plain
                  ``response_format={"type": "json_object"}`` (no schema pinning).
                - ``None``: free-form text response.
        """


class LiteLLMClient(LLMClient):
    """Thin wrapper around `litellm.completion`. Model selection is pure string.

    The `model` parameter is a LiteLLM routing string: `"gemini/gemini-2.5-flash"`,
    `"openai/gpt-4o-mini"`, `"anthropic/claude-3-5-sonnet"`, etc. LiteLLM
    discovers the appropriate API key from env vars automatically.
    """

    def __init__(self, model: str, max_tokens: int = 2048) -> None:
        self.model = model
        self.max_tokens = max_tokens

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponse:
        response_format: dict[str, Any] | None = None
        if schema is not None:
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                json_schema = schema.model_json_schema()
                response_format = {
                    "type": "json_schema",
                    "json_schema": {
                        "name": schema.__name__,
                        "schema": json_schema,
                        "strict": True,
                    },
                }
            elif isinstance(schema, dict):
                response_format = {"type": "json_object"}

        import time as _time  # noqa: PLC0415

        last_exc: Exception | None = None
        for attempt in range(4):
            try:
                resp = litellm.completion(
                    model=self.model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=temperature,
                    max_tokens=self.max_tokens,
                    response_format=response_format,
                )
                return LLMResponse(
                    text=resp.choices[0].message.content,
                    model=self.model,
                    prompt_tokens=resp.usage.prompt_tokens,
                    completion_tokens=resp.usage.completion_tokens,
                )
            except (
                litellm.ServiceUnavailableError,
                litellm.RateLimitError,
                litellm.InternalServerError,
                litellm.APIConnectionError,
            ) as exc:
                last_exc = exc
                if attempt == 3:
                    raise
                _time.sleep(4 * (2**attempt))
        raise last_exc


@dataclass
class FakeLLM(LLMClient):
    """Scripted test double — prompt SHA-256[:16] → canned response mapping.

    Fixture format (JSON)::

        {
          "a1b2c3d4e5f60718": "canned response for the prompt that hashes here",
          "0011223344556677": "another canned response"
        }

    Unknown prompts raise `UnknownPromptError` (never returns empty text).
    This forces fixture updates whenever a prompt changes — catches prompt
    drift (pitfall P18) loudly at test time instead of producing degraded
    output silently.
    """

    fixture_path: Path
    _responses: dict[str, str] = field(init=False)
    calls: list[tuple[str, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._responses = json.loads(self.fixture_path.read_text(encoding="utf-8"))

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponse:
        self.calls.append((prompt, temperature))
        key = self._hash(prompt)
        if key not in self._responses:
            raise UnknownPromptError(
                f"FakeLLM fixture {self.fixture_path} has no response for "
                f"prompt hash {key}. First 120 chars of prompt: {prompt[:120]!r}"
            )
        return LLMResponse(
            text=self._responses[key],
            model="fake",
            prompt_tokens=0,
            completion_tokens=0,
        )
