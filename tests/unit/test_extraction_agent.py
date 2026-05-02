"""Unit tests for ExtractionAgent (EXT-05, EXT-06).

Strategy: inject a fake PdfExtractor whose responses are scripted per paper_id.
The real PdfExtractor is exercised in test_pdf_extractor.py — this file is
about orchestration: persistence path, failure isolation, stats population.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import pytest  # noqa: F401 — kept for fixture access in future growth

from ara.agents.extraction import ExtractionAgent
from ara.services.pdf import ExtractionResult, ExtractionTooShort
from ara.state import ExtractionStats, Paper, PipelineState


@dataclass
class _FakeExtractor:
    responses: dict[str, "ExtractionResult | Exception"] = field(default_factory=dict)
    calls: list[Path] = field(default_factory=list)

    def extract(self, pdf_path: Path) -> ExtractionResult:
        self.calls.append(pdf_path)
        key = pdf_path.stem
        resp = self.responses.get(key)
        if isinstance(resp, Exception):
            raise resp
        if resp is None:
            raise ExtractionTooShort(f"no fake response for {key}")
        return resp


def _paper(paper_id: str, pdf_path: str | None = None) -> Paper:
    return Paper(paper_id=paper_id, title=f"T-{paper_id}", pdf_path=pdf_path)


def _state_with_papers(tmp_path: Path, papers: list[Paper]) -> PipelineState:
    runs_dir = tmp_path / "runs"
    s = PipelineState(
        run_id="R-EXT",
        question="q",
        config_snapshot={},
        runs_dir=str(runs_dir),
        papers=papers,
    )
    for p in papers:
        if p.pdf_path:
            Path(p.pdf_path).parent.mkdir(parents=True, exist_ok=True)
            Path(p.pdf_path).write_bytes(b"%PDF-fake")
    return s


def _ok_result(md: str = "# Title\n\nBody text long enough to be real.") -> ExtractionResult:
    return ExtractionResult(
        markdown=md,
        num_detected_sections=1,
        hidden_text_spans_stripped=0,
        header_footer_lines_stripped=0,
        pages=3,
    )




def test_per_paper_failure_isolation(tmp_path):
    """EXT-05: one malformed PDF does not halt the stage; other papers succeed."""
    import pymupdf
    p_ok = _paper("ok", pdf_path=str(tmp_path / "pdfs" / "ok.pdf"))
    p_bad = _paper("bad", pdf_path=str(tmp_path / "pdfs" / "bad.pdf"))
    state = _state_with_papers(tmp_path, [p_ok, p_bad])
    extractor = _FakeExtractor(responses={
        "ok": _ok_result(),
        "bad": pymupdf.FileDataError("malformed"),
    })
    agent = ExtractionAgent(pdf_extractor=extractor)
    out = agent.run(state)

    ok_paper = next(p for p in out.papers if p.paper_id == "ok")
    bad_paper = next(p for p in out.papers if p.paper_id == "bad")
    assert ok_paper.extraction_ok is True
    assert bad_paper.extraction_ok is False
    assert bad_paper.fallback_to_abstract is True
    assert any(pid == "bad" for pid, _ in out.extraction_stats.extraction_failures)


def test_too_short_extraction_falls_back(tmp_path):
    """EXT-05: ExtractionTooShort → fallback_to_abstract=True, not a raise."""
    p = _paper("short", pdf_path=str(tmp_path / "pdfs" / "short.pdf"))
    state = _state_with_papers(tmp_path, [p])
    extractor = _FakeExtractor(responses={"short": ExtractionTooShort("only 50 chars")})
    out = ExtractionAgent(pdf_extractor=extractor).run(state)
    short_paper = out.papers[0]
    assert short_paper.extraction_ok is False
    assert short_paper.fallback_to_abstract is True


def test_all_papers_fail_stage_still_completes(tmp_path):
    """EXT-05: when every paper fails, the stage returns normally (no raise)."""
    papers = [_paper(f"p{i}", pdf_path=str(tmp_path / "pdfs" / f"p{i}.pdf")) for i in range(3)]
    state = _state_with_papers(tmp_path, papers)
    extractor = _FakeExtractor(responses={
        f"p{i}": ExtractionTooShort("too short") for i in range(3)
    })
    out = ExtractionAgent(pdf_extractor=extractor).run(state)
    assert all(not p.extraction_ok for p in out.papers)
    assert all(p.fallback_to_abstract for p in out.papers)
    assert out.extraction_stats.pdfs_extracted_ok == 0
    assert len(out.extraction_stats.extraction_failures) == 3


def test_abstract_only_papers_are_skipped(tmp_path):
    """Papers without pdf_path are left alone (retrieval already flagged them)."""
    p_abs = _paper("abs", pdf_path=None)
    state = _state_with_papers(tmp_path, [p_abs])
    state.papers[0].fallback_to_abstract = True
    state.papers[0].pdf_source = "abstract-only"
    extractor = _FakeExtractor()
    out = ExtractionAgent(pdf_extractor=extractor).run(state)
    assert extractor.calls == []
    assert out.extraction_stats.pdfs_attempted == 0




def test_extracted_md_path_layout(tmp_path):
    """EXT-06: extracted.md is written to runs/<run_id>/papers/<paper_id>/extracted.md."""
    p = _paper("good", pdf_path=str(tmp_path / "pdfs" / "good.pdf"))
    state = _state_with_papers(tmp_path, [p])
    extractor = _FakeExtractor(responses={"good": _ok_result(md="# Hello\n\nBody here long enough.")})
    out = ExtractionAgent(pdf_extractor=extractor).run(state)

    expected = Path(state.runs_dir) / state.run_id / "papers" / "good" / "extracted.md"
    assert expected.exists(), f"missing extracted.md at {expected}"
    assert expected.read_text() == "# Hello\n\nBody here long enough."

    good = out.papers[0]
    assert good.extracted_md_path is not None
    assert good.extracted_md_path.endswith("papers/good/extracted.md")
    assert good.extraction_meta is not None
    assert good.extraction_meta.num_detected_sections == 1
    assert good.extraction_meta.pages == 3


def test_atomic_write_text_crash_safe(tmp_path, monkeypatch):
    """EXT-06: ExtractionAgent uses atomic_write_text (verifies path consistency).

    Indirect assertion: patch ara.agents.extraction.atomic_write_text
    (the alias ExtractionAgent imports) so a spy records calls.
    """
    calls: list[tuple[str, Path]] = []

    def spy(text: str, path: Path) -> None:
        calls.append((text, path))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)

    import ara.agents.extraction as ext_mod
    monkeypatch.setattr(ext_mod, "atomic_write_text", spy)

    p = _paper("atomic", pdf_path=str(tmp_path / "pdfs" / "atomic.pdf"))
    state = _state_with_papers(tmp_path, [p])
    extractor = _FakeExtractor(responses={"atomic": _ok_result(md="# Atomic\n\nLong body.")})
    ExtractionAgent(pdf_extractor=extractor).run(state)
    assert len(calls) == 1
    _, written_path = calls[0]
    assert written_path.name == "extracted.md"




def test_extraction_stats_populated(tmp_path):
    """state.extraction_stats is an ExtractionStats BaseModel with sane counters."""
    papers = [
        _paper("a", pdf_path=str(tmp_path / "pdfs" / "a.pdf")),
        _paper("b", pdf_path=str(tmp_path / "pdfs" / "b.pdf")),
    ]
    state = _state_with_papers(tmp_path, papers)
    extractor = _FakeExtractor(responses={
        "a": ExtractionResult(markdown="# A\n\nBody", num_detected_sections=1,
                              hidden_text_spans_stripped=2, header_footer_lines_stripped=3, pages=5),
        "b": ExtractionTooShort("too short"),
    })
    out = ExtractionAgent(pdf_extractor=extractor).run(state)
    s = out.extraction_stats
    assert isinstance(s, ExtractionStats)
    assert s.pdfs_attempted == 2
    assert s.pdfs_extracted_ok == 1
    assert s.hidden_text_spans_stripped == 2
    assert s.header_footer_lines_stripped == 3
    assert len(s.extraction_failures) == 1


def test_stage_report_appended(tmp_path):
    """One StageReport(stage='extraction') appended per run."""
    p = _paper("x", pdf_path=str(tmp_path / "pdfs" / "x.pdf"))
    state = _state_with_papers(tmp_path, [p])
    extractor = _FakeExtractor(responses={"x": _ok_result()})
    before = len(state.stage_reports)
    out = ExtractionAgent(pdf_extractor=extractor).run(state)
    assert len(out.stage_reports) == before + 1
    rep = out.stage_reports[-1]
    assert rep.stage == "extraction"
    assert rep.inputs_count == 1
    assert rep.outputs_count == 1
