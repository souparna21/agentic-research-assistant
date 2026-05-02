"""Typed exceptions for the Agentic Research Assistant.

Every module raises only typed exceptions from this file (or subclasses).
Bare `except:` and silent pass is banned (pitfall P8). `ruff` rule `E722`
enforces.
"""


class AraError(Exception):
    """Base exception for all ara-originated errors."""


class UnknownPromptError(AraError):
    """FakeLLM received a prompt whose hash is not in its fixture JSON."""


class FieldOwnershipViolation(AraError):
    """A stage wrote to a PipelineState field it does not own (pitfall P12)."""


class StageSkipped(AraError):
    """Internal signal: stage output artifact already exists; skip this stage."""


class MissingAPIKeyError(AraError):
    """The active LLM provider requires an API key that is not set."""


class PromptTemplateError(AraError):
    """Jinja2 StrictUndefined fired — a template variable was missing at render."""


class LatexCompileError(AraError):
    """latexmk exited non-zero, timed out, or the binary is missing (RPT-06)."""


class UnresolvedCitationError(AraError):
    """Compiled PDF contains [?] markers — BibTeX keys missing from references.bib (RPT-07)."""


class CachedDemoMissingError(AraError):
    """demo-cache.tar.gz is missing — cannot bootstrap --cached-only (PKG-02)."""
