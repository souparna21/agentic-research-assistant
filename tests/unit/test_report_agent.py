"""Unit tests for ReportAgent (RPT-01/02/03/05/07/08/09).

VALIDATION.md task IDs:
  3-08-01  -> test_report_tex_rendered_to_run_dir                    (RPT-01)
  3-08-02  -> test_latex_env_uses_custom_delimiters                  (RPT-02)
  3-08-03  -> test_paper_title_with_specials_is_escaped              (RPT-03)
  3-08-04  -> test_adversarial_title_still_compiles          [latex] (RPT-03)
  3-08-05  -> test_bibtex_keys_are_unique_post_collision_fix         (RPT-05)
  3-08-06  -> test_twin_papers_produce_unique_keys_and_compile [latex](RPT-05)
  3-08-07  -> test_report_fails_on_unresolved_citation               (RPT-07)
  3-08-08  -> test_successful_compile_zero_unresolved         [latex] (RPT-07)
  3-08-09  -> test_stage_report_appendix_rendered                    (RPT-08)
  3-08-10  -> test_final_artifacts_persisted                        (RPT-09)
  3-08-11  -> test_report_writes_are_atomic                         (RPT-09)

Plus one implicit passthrough test:
  test_standard_jinja_delimiters_passthrough  (RPT-02 belt-and-suspenders)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


def _seed_analysis_artifacts(tmp_path: Path, papers, matrix_rows=None, gaps_rows=None):
    """Write runs/<rid>/summaries/<pid>.json + comparison.json + gaps.json under tmp_path."""
    run_dir = tmp_path / "rid"
    (run_dir / "summaries").mkdir(parents=True, exist_ok=True)
    for p in papers:
        summary = {
            "paper_id": p.paper_id,
            "summary": {
                "paper_id": p.paper_id,
                "objective": {
                    "text": f"Objective of {p.paper_id}",
                    "citations": [f"[P{p.paper_id}-1]"],
                },
                "methodology": {
                    "text": f"Methodology of {p.paper_id}",
                    "citations": [f"[P{p.paper_id}-2]"],
                },
                "findings": [
                    {
                        "text": f"Finding A for {p.paper_id}",
                        "citations": [f"[P{p.paper_id}-3]"],
                    },
                    {
                        "text": f"Finding B for {p.paper_id}",
                        "citations": [f"[P{p.paper_id}-4]"],
                    },
                ],
                "limitations": [
                    {
                        "text": f"Limitation of {p.paper_id}",
                        "citations": [f"[P{p.paper_id}-5]"],
                    },
                ],
            },
            "methodology": {
                "paper_id": p.paper_id,
                "datasets": ["D1"],
                "techniques": ["T1"],
                "metrics": ["M1"],
            },
            "limitations": {
                "paper_id": p.paper_id,
                "limitations": [
                    {"text": "lim", "citations": [f"[P{p.paper_id}-6]"]}
                ],
                "future_work": [
                    {"text": "fw", "citations": [f"[P{p.paper_id}-7]"]}
                ],
            },
            "verifier": {
                "claims": [
                    {
                        "claim_text": "Finding A",
                        "citations": [f"[P{p.paper_id}-3]"],
                        "verdict": "supported",
                        "reason": "ok",
                    }
                ]
            },
        }
        (run_dir / "summaries" / f"{p.paper_id}.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
    matrix = {
        "rows": matrix_rows
        or [
            {
                "paper_id": p.paper_id,
                "paper_title": p.title,
                "datasets": ["D1"],
                "techniques": ["T1"],
                "metrics": ["M1"],
            }
            for p in papers
        ]
    }
    (run_dir / "comparison.json").write_text(json.dumps(matrix), encoding="utf-8")
    gaps = {
        "gaps": gaps_rows
        or [
            {
                "text": "no corpus covers cross-domain.",
                "citations": [f"[P{papers[0].paper_id}-1]"],
            }
        ]
    }
    (run_dir / "gaps.json").write_text(json.dumps(gaps), encoding="utf-8")
    return run_dir


def _default_state(tmp_path, papers, run_dir):
    from ara.state import PipelineState

    return PipelineState(
        run_id="rid",
        question="How does retrieval-augmented generation reduce hallucination?",
        config_snapshot={"model": "gemini/gemini-2.5-flash"},
        runs_dir=str(tmp_path),
        papers=papers,
        summaries_dir="rid/summaries",
        comparison_path="rid/comparison.json",
        gaps_path="rid/gaps.json",
    )


def _mk_paper(paper_id: str, title: str, year: int = 2024, author: str = "Smith"):
    from ara.state import Paper

    return Paper(
        paper_id=paper_id,
        title=title,
        year=year,
        authors=[author],
        analysis_status="succeeded",
    )




def test_report_tex_rendered_to_run_dir(tmp_path: Path) -> None:
    """Task 3-08-01 -- report.tex under runs/<run_id>/ after ReportAgent.run()."""
    from ara.agents.report import ReportAgent
    from ara.services.prompts import LatexTemplateLoader

    papers = [
        _mk_paper("p1", "RAG"),
        _mk_paper("p2", "Attention", author="Vaswani", year=2017),
    ]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)

    agent = ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts")))
    state = agent.run(state)

    tex = run_dir / "report.tex"
    assert tex.exists()
    body = tex.read_text(encoding="utf-8")
    assert "\\documentclass" in body
    assert "\\section{" in body
    assert "<<" not in body
    assert ">>" not in body




def test_latex_env_uses_custom_delimiters() -> None:
    """Task 3-08-02 -- LatexTemplateLoader env uses <% %> / << >> delimiters."""
    from ara.services.prompts import LatexTemplateLoader

    tl = LatexTemplateLoader(Path("prompts"))
    assert tl.env.block_start_string == "<%"
    assert tl.env.block_end_string == "%>"
    assert tl.env.variable_start_string == "<<"
    assert tl.env.variable_end_string == ">>"
    assert tl.env.autoescape is False


def test_standard_jinja_delimiters_passthrough(tmp_path: Path) -> None:
    """Implicit RPT-02 belt-and-suspenders -- ``{% for x %}`` / ``{{ y }}``
    in the LaTeX template render as LITERAL text because LatexTemplateLoader
    uses custom delimiters."""
    from ara.services.prompts import LatexTemplateLoader

    tmpl = tmp_path / "passthrough.tex.j2"
    tmpl.write_text(
        "literal {% for x %} and {{ y }} remain\n<< name >>", encoding="utf-8"
    )
    tl = LatexTemplateLoader(tmp_path)
    out = tl.render("passthrough.tex.j2", name="Alice")
    assert "{% for x %}" in out
    assert "{{ y }}" in out
    assert "Alice" in out




def test_paper_title_with_specials_is_escaped(tmp_path: Path) -> None:
    """Task 3-08-03 -- Paper title with & % $ _ # ^ ~ \\ { } is escaped in report.tex."""
    from ara.agents.report import ReportAgent
    from ara.services.prompts import LatexTemplateLoader

    adversarial = "R&D 100% $cost$ _undersc #tag ^caret ~tilde \\back {brace}"
    papers = [_mk_paper("p1", adversarial)]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)

    ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)
    body = (run_dir / "report.tex").read_text(encoding="utf-8")
    assert "R\\&D" in body
    assert "100\\%" in body
    assert "\\$cost\\$" in body
    assert "\\_undersc" in body
    assert "\\#tag" in body
    assert "\\textbackslash{}back" in body
    assert "\\{brace\\}" in body


@pytest.mark.latex
def test_adversarial_title_still_compiles(tmp_path: Path) -> None:
    """Task 3-08-04 -- Adversarial title compiles cleanly via latexmk."""
    from ara.agents.report import ReportAgent
    from ara.services.latex import count_unresolved_citations, latexmk_available
    from ara.services.prompts import LatexTemplateLoader

    if not latexmk_available():
        pytest.skip("latexmk not installed")

    papers = [_mk_paper("p1", "& % $_ #^~\\{}")]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)
    ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)

    pdf = run_dir / "report.pdf"
    assert pdf.exists()
    assert count_unresolved_citations(pdf) == 0




def test_bibtex_keys_are_unique_post_collision_fix(tmp_path: Path) -> None:
    """Task 3-08-05 -- Colliding pairs produce unique bibtex keys after resolution."""
    from ara.agents.report import ReportAgent
    from ara.services.prompts import LatexTemplateLoader

    p1 = _mk_paper("p1", "Attention Networks", year=2024, author="Smith")
    p2 = _mk_paper("p2", "Attention Mechanisms", year=2024, author="Smith")
    papers = [p1, p2]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)
    ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)

    keys = [p.bibtex_key for p in state.papers if p.bibtex_key]
    assert len(keys) == 2
    assert len(set(keys)) == 2
    assert state.report_stats.bibtex_collision_fixes >= 1


@pytest.mark.latex
def test_twin_papers_produce_unique_keys_and_compile(tmp_path: Path) -> None:
    """Task 3-08-06 -- Twin-paper fixture resolves collision AND compiles with 0 [?]."""
    from ara.agents.report import ReportAgent
    from ara.services.latex import count_unresolved_citations, latexmk_available
    from ara.services.prompts import LatexTemplateLoader

    if not latexmk_available():
        pytest.skip("latexmk not installed")

    p1 = _mk_paper("p1", "Attention Networks", 2024, "Smith")
    p2 = _mk_paper("p2", "Attention Mechanisms", 2024, "Smith")
    papers = [p1, p2]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)
    ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)

    pdf = run_dir / "report.pdf"
    assert pdf.exists()
    assert count_unresolved_citations(pdf) == 0




def test_report_fails_on_unresolved_citation(tmp_path: Path, monkeypatch) -> None:
    """Task 3-08-07 -- If count_unresolved_citations > 0, ReportAgent raises UnresolvedCitationError."""
    from ara.agents.report import ReportAgent
    from ara.services.latex import UnresolvedCitationError
    from ara.services.prompts import LatexTemplateLoader

    papers = [_mk_paper("p1", "RAG")]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)

    def fake_compile(tex_path, out_dir, *, timeout_s=120):
        pdf = out_dir / (tex_path.stem + ".pdf")
        pdf.write_bytes(b"%PDF-1.4 dummy")
        return pdf

    import ara.agents.report as rpt_mod

    monkeypatch.setattr(rpt_mod, "compile_latex", fake_compile)
    monkeypatch.setattr(rpt_mod, "count_unresolved_citations", lambda p: 3)
    monkeypatch.setattr(rpt_mod, "latexmk_available", lambda: True)

    with pytest.raises(UnresolvedCitationError, match="3"):
        ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)


@pytest.mark.latex
def test_successful_compile_zero_unresolved(tmp_path: Path) -> None:
    """Task 3-08-08 -- Clean compile -> count_unresolved_citations == 0."""
    from ara.agents.report import ReportAgent
    from ara.services.latex import latexmk_available
    from ara.services.prompts import LatexTemplateLoader

    if not latexmk_available():
        pytest.skip("latexmk not installed")

    papers = [
        _mk_paper("p1", "RAG", 2020, "Lewis"),
        _mk_paper("p2", "DPR", 2020, "Karpukhin"),
    ]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)
    ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)
    assert state.report_stats.unresolved_citations == 0




def test_stage_report_appendix_rendered(tmp_path: Path) -> None:
    """Task 3-08-09 -- report.tex contains retrieval/extraction/indexing/analysis/report stats sections."""
    from ara.agents.report import ReportAgent
    from ara.services.prompts import LatexTemplateLoader

    papers = [_mk_paper("p1", "RAG")]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)
    ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)

    body = (run_dir / "report.tex").read_text(encoding="utf-8")
    for section in ["Retrieval", "Extraction", "Indexing", "Analysis", "Report"]:
        assert section in body, f"appendix missing {section!r}"




def test_final_artifacts_persisted(tmp_path: Path) -> None:
    """Task 3-08-10 -- report.tex + references.bib exist under runs/<run_id>/."""
    from ara.agents.report import ReportAgent
    from ara.services.prompts import LatexTemplateLoader

    papers = [_mk_paper("p1", "RAG"), _mk_paper("p2", "BART")]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)
    ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)

    assert (run_dir / "report.tex").exists()
    assert (run_dir / "references.bib").exists()


def test_report_writes_are_atomic(tmp_path: Path, monkeypatch) -> None:
    """Task 3-08-11 -- report.tex + references.bib go through atomic_write_text."""
    import ara.persistence as pers_mod
    from ara.agents.report import ReportAgent
    from ara.services.prompts import LatexTemplateLoader

    papers = [_mk_paper("p1", "RAG")]
    run_dir = _seed_analysis_artifacts(tmp_path, papers)
    state = _default_state(tmp_path, papers, run_dir)

    calls: list[str] = []
    orig = pers_mod.atomic_write_text

    def spy(text, path):
        calls.append(str(path))
        orig(text, path)

    import ara.agents.report as rpt_mod

    monkeypatch.setattr(rpt_mod, "atomic_write_text", spy)
    ReportAgent(latex_templates=LatexTemplateLoader(Path("prompts"))).run(state)

    paths = set(calls)
    assert any(p.endswith("report.tex") for p in paths)
    assert any(p.endswith("references.bib") for p in paths)
