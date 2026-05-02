"""Unit tests for the 6 analysis-prompt pydantic schemas (ANL-02, ANL-06).

VALIDATION.md task IDs:
  3-05-01 -> test_all_schemas_validate_committed_fixtures  (ANL-06)
  3-05-02 -> test_claim_citations_are_well_formed          (ANL-02)

Additional negative tests lock extra='forbid' behavior (P18 shape-drift tripwire)
and VerifierVerdict's Literal verdict constraint.
"""
from __future__ import annotations

import json
from pathlib import Path

import pydantic
import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "schemas"


def _load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def test_all_schemas_validate_committed_fixtures() -> None:
    """Task 3-05-01 -- every valid.json parses cleanly against its schema (ANL-06)."""
    from ara.analysis_schemas import (
        GapAnalysis,
        LimitationsExtraction,
        MethodologyExtraction,
        MethodologyMatrix,
        PerPaperSummary,
        VerifierVerdict,
    )

    PerPaperSummary.model_validate(_load("per_paper_summary.valid.json"))
    MethodologyExtraction.model_validate(_load("methodology_extraction.valid.json"))
    LimitationsExtraction.model_validate(_load("limitations_extraction.valid.json"))
    MethodologyMatrix.model_validate(_load("methodology_matrix.valid.json"))
    GapAnalysis.model_validate(_load("gap_analysis.valid.json"))
    VerifierVerdict.model_validate(_load("verifier_verdict.valid.json"))


def test_claim_citations_are_well_formed() -> None:
    """Task 3-05-02 -- Claim rejects empty AND malformed citations.

    2026-05-02: bare ``P<id>-N`` is now accepted and normalized to
    ``[P<id>-N]``. Reason: LLMs repeatedly emit the bare form against
    long S2 paper-ids; the strict regex was crashing 10/26 papers per
    live run. Bracketed remains the canonical storage shape.
    """
    from ara.analysis_schemas import Claim

    Claim(text="RAG boosts QA.", citations=["[Ppaper1-1]"])

    with pytest.raises(pydantic.ValidationError, match="citation"):
        Claim(text="x", citations=[])

    with pytest.raises(pydantic.ValidationError, match="doesn't match"):
        Claim(text="x", citations=["[1]"])

    bare = Claim(text="x", citations=["Ppaper3-7"])
    assert bare.citations == ["[Ppaper3-7]"]

    bare_long = Claim(
        text="x",
        citations=["P659bf9ce7175e1ec266ff54359e2bd76e0b7ff31-2"],
    )
    assert bare_long.citations == [
        "[P659bf9ce7175e1ec266ff54359e2bd76e0b7ff31-2]"
    ]

    with pytest.raises(pydantic.ValidationError, match="doesn't match"):
        Claim(text="x", citations=["1"])

    Claim(text="x", citations=["[Ppaper2-4]", "[Ppaper7-12]"])


@pytest.mark.parametrize(
    "name,cls_name",
    [
        ("per_paper_summary", "PerPaperSummary"),
        ("methodology_extraction", "MethodologyExtraction"),
        ("limitations_extraction", "LimitationsExtraction"),
        ("methodology_matrix", "MethodologyMatrix"),
        ("gap_analysis", "GapAnalysis"),
        ("verifier_verdict", "VerifierVerdict"),
    ],
)
def test_every_schema_rejects_extra_fields(name: str, cls_name: str) -> None:
    """Negative coverage -- model_config extra='forbid' catches schema drift (P18)."""
    import ara.analysis_schemas as mod

    cls = getattr(mod, cls_name)
    data = _load(f"{name}.valid.json")
    data["unexpected_drift_field"] = "gotcha"
    with pytest.raises(pydantic.ValidationError, match="extra"):
        cls.model_validate(data)


def test_verifier_verdict_requires_literal_verdict_value() -> None:
    """VerifierVerdict rejects verdicts outside {supported, partial, unsupported}."""
    from ara.analysis_schemas import VerifierVerdict

    data = _load("verifier_verdict.valid.json")
    data["claims"][0]["verdict"] = "maybe"
    with pytest.raises(pydantic.ValidationError):
        VerifierVerdict.model_validate(data)
