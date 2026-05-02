"""Pydantic response schemas for Phase 3 analysis prompts (ANL-06) + verifier (ANL-08).

Shape-drift tripwire (P18): ``model_config = {"extra": "forbid"}`` on every schema.
An LLM response that omits a required field OR adds an unexpected field raises
``pydantic.ValidationError`` — AnalysisAgent's per-paper narrow try/except records
it as a per-paper failure without halting the stage (ANL-09).

Citation format (ANL-02): every ``Claim.citations`` entry MUST fullmatch
``[P<paper_id>-N]`` where N is a positive integer and ``<paper_id>`` is any
non-``]`` chars. Pinning the paper_id inside the citation marker ties the
citation to a single paper — resists attribution drift during cross-paper
synthesis (methodology matrix + gap analysis).

Structural choice: single flat module over a ``schemas/`` package. Rationale:
all 6 schemas are tightly coupled (``Claim`` is shared; ``MethodologyMatrix`` /
``GapAnalysis`` / ``VerifierVerdict`` each reference its structure), they fit in
~150 LOC, and a flat module removes ``__init__.py`` / re-export boilerplate.
Import site is one line::

    from ara.analysis_schemas import (
        Claim, PerPaperSummary, MethodologyExtraction, LimitationsExtraction,
        MethodologyMatrix, GapAnalysis, VerifierVerdict,
    )
"""
from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, Field, field_validator

_CITE_RE = re.compile(r"\[P[^\]]+-\d+\]")




class Claim(BaseModel):
    """Single claim sentence with at least one well-formed passage citation.

    Validation (ANL-02):
      - ``citations`` must be non-empty (every claim needs >= 1 citation)
      - each citation must fullmatch ``[P<paper_id>-N]`` where N is a positive int
    """

    model_config = {"extra": "forbid"}

    text: str
    citations: list[str] = Field(...)

    @field_validator("citations")
    @classmethod
    def citations_nonempty_and_wellformed(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("every claim needs >= 1 citation")
        for c in v:
            if not _CITE_RE.fullmatch(c):
                raise ValueError(f"citation '{c}' doesn't match [P<paper_id>-N]")
        return v




class PerPaperSummary(BaseModel):
    """Structured per-paper summary with typed Claim fields carrying citations."""

    model_config = {"extra": "forbid"}

    paper_id: str
    objective: Claim
    methodology: Claim
    findings: list[Claim] = Field(..., min_length=2, max_length=5)
    limitations: list[Claim] = Field(..., min_length=1, max_length=4)




class MethodologyExtraction(BaseModel):
    """Per-paper methodology attributes (prompts/methodology_extraction.j2).

    Feeds :class:`MethodologyMatrix` via the cross-paper synthesis call.
    """

    model_config = {"extra": "forbid"}

    paper_id: str
    datasets: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class MatrixRow(BaseModel):
    """One row of the cross-paper methodology comparison matrix."""

    model_config = {"extra": "forbid"}

    paper_id: str
    paper_title: str
    datasets: list[str] = Field(default_factory=list)
    techniques: list[str] = Field(default_factory=list)
    metrics: list[str] = Field(default_factory=list)


class MethodologyMatrix(BaseModel):
    """Cross-paper methodology matrix (prompts/methodology_matrix_synthesis.j2)."""

    model_config = {"extra": "forbid"}

    rows: list[MatrixRow] = Field(default_factory=list)




class LimitationsExtraction(BaseModel):
    """Per-paper limitations + future-work claim lists (prompts/limitations_extraction.j2).

    Feeds :class:`GapAnalysis` via the cross-paper gap synthesis call.
    """

    model_config = {"extra": "forbid"}

    paper_id: str
    limitations: list[Claim] = Field(default_factory=list)
    future_work: list[Claim] = Field(default_factory=list)


class GapAnalysis(BaseModel):
    """Cross-paper research-gap synthesis (prompts/gap_synthesis.j2)."""

    model_config = {"extra": "forbid"}

    gaps: list[Claim] = Field(default_factory=list)




class ClaimVerdict(BaseModel):
    """Per-claim verifier verdict (one entry per distinct claim in a summary)."""

    model_config = {"extra": "forbid"}

    claim_text: str
    citations: list[str]
    verdict: Literal["supported", "partial", "unsupported"]
    reason: str


class VerifierVerdict(BaseModel):
    """Verifier output aggregating per-claim verdicts for one summary (prompts/verifier.j2)."""

    model_config = {"extra": "forbid"}

    claims: list[ClaimVerdict] = Field(..., min_length=1)


__all__ = [
    "Claim",
    "ClaimVerdict",
    "GapAnalysis",
    "LimitationsExtraction",
    "MatrixRow",
    "MethodologyExtraction",
    "MethodologyMatrix",
    "PerPaperSummary",
    "VerifierVerdict",
]
