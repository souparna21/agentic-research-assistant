"""Grep tests over report.tex — PKG-03 structural gate.

These tests verify that the final IE624 course report (``report.tex`` at the
repo root) preserves its pre-existing §1-§10 sections and gains the three new
sections added in plan 04-04 (Implementation Results, Evaluation, Lessons
Learned), plus module-path annotations on the TikZ pipeline diagram.

Tests 1-4 are the RED baseline for plan 04-04 Task 2 (the surgical ``report.tex``
edit); they must fail on the pre-edit file and pass after the edit lands.
Test 5 is the pre-existing-content safety net — it must PASS from RED onward
to prove the edit does not regress §1-§10.
"""

from __future__ import annotations

from pathlib import Path

REPORT = Path(__file__).resolve().parents[2] / "report.tex"


def _read() -> str:
    assert REPORT.exists(), f"report.tex missing at {REPORT}"
    return REPORT.read_text(encoding="utf-8")


def test_has_implementation_results() -> None:
    body = _read()
    assert r"\section{Implementation Results}" in body, (
        "report.tex missing \\section{Implementation Results}"
    )


def test_has_evaluation() -> None:
    body = _read()
    assert r"\section{Evaluation}" in body, (
        "report.tex missing \\section{Evaluation}"
    )


def test_has_lessons_learned() -> None:
    body = _read()
    assert r"\section{Lessons Learned}" in body, (
        "report.tex missing \\section{Lessons Learned}"
    )


def test_tikz_has_module_paths() -> None:
    """All 6 production module paths must appear in the TikZ pipeline diagram.

    The planner spec (04-04-PLAN.md §interfaces) requires each of the six
    pipeline-stage boxes to be annotated with ``ara.agents.<name>`` so the
    diagram reads as a module map rather than an abstract pipeline.
    """
    body = _read()
    for module in (
        "ara.agents.query",
        "ara.agents.retrieval",
        "ara.agents.extraction",
        "ara.agents.indexing",
        "ara.agents.analysis",
        "ara.agents.report",
    ):
        assert module in body, f"report.tex TikZ missing module path: {module!r}"


def test_preserves_existing_sections() -> None:
    """Safety net: all pre-existing ``\\section`` headings must remain present.

    This catches accidental deletion of §1-§10 content during the surgical
    edit. The titles listed here are the verbatim pre-existing section
    headings of the preliminary IE624 report as of plan 04-04 RED.
    """
    body = _read()
    required = [
        r"\section{Introduction}",
        r"\section{Sub-Problem Decomposition}",
        r"\section{System Architecture}",
        r"\section{Technology Stack}",
        r"\section{Multi-Agent Framework Design}",
        r"\section{RAG Pipeline Design}",
        r"\section{Feasibility Analysis}",
        r"\section{Technical Obstacles and Mitigations}",
        r"\section{Related Work}",
        r"\section{Conclusion}",
    ]
    for heading in required:
        assert heading in body, f"report.tex LOST pre-existing section: {heading}"


def test_conclusion_mentions_shipped() -> None:
    """Conclusion rewrite signal: narrative must use shipping language.

    The pre-edit Conclusion opens ``This report has presented the design and
    preliminary validation…''; the post-edit version must reflect that the
    v1.0 system shipped. We assert at least one shipping verb is present in
    the body (excluding the section heading itself).
    """
    body = _read()
    idx = body.find(r"\section{Conclusion}")
    assert idx >= 0, "report.tex missing \\section{Conclusion}"
    tail = body[idx:]
    terminators = [r"\bibliography{", r"\end{document}"]
    stops = [tail.find(t) for t in terminators if tail.find(t) >= 0]
    conclusion = tail[: min(stops)] if stops else tail

    shipping_verbs = ("shipped", "shipping", "delivered", "release", "v1.0")
    assert any(verb in conclusion.lower() for verb in shipping_verbs), (
        "Conclusion body does not mention any of: "
        f"{shipping_verbs!r} — rewrite may not have landed."
    )
