"""ReportAgent — renders report.tex + references.bib, compiles via latexmk, scans for [?].

RPT-01..09 orchestration. Composition:
  - LatexTemplateLoader from services.prompts (plan 03-08 task 08-01)
  - latex_escape filter from services.latex (plan 03-07 RPT-03)
  - make_bibtex_key + resolve_key_collisions from services.latex (RPT-04/05)
  - compile_latex + count_unresolved_citations from services.latex (RPT-06/07)
  - atomic_write_text from persistence (RPT-09)

Narrow exceptions: only ``(LatexCompileError, UnresolvedCitationError, OSError)``
are caught. No bare ``except Exception`` — pitfall P8 discipline.
"""
from __future__ import annotations

import json
import time
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

from ara.persistence import atomic_write_text
from ara.services.latex import (
    LatexCompileError,
    UnresolvedCitationError,
    compile_latex,
    count_unresolved_citations,
    latexmk_available,
    make_bibtex_key,
    resolve_key_collisions,
)
from ara.services.prompts import LatexTemplateLoader
from ara.state import PipelineState, ReportStats, StageReport

log = structlog.get_logger(__name__)


@dataclass
class ReportAgent:
    """Stage Protocol conformant. Renders + compiles the final literature-review LaTeX."""

    latex_templates: LatexTemplateLoader
    name: str = "report"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "report.tex"

    def run(self, state: PipelineState) -> PipelineState:
        """Render report.tex + references.bib, compile via latexmk, verify [?] count == 0.

        Orchestration steps:
          1. Resolve BibTeX keys on all citable papers (analysis_status in
             {"succeeded", None}); collision-resolve; write papers[i].bibtex_key.
          2. Write references.bib atomically (@article entries keyed by the
             collision-resolved keys).
          3. Load per-paper summaries + comparison + gaps from AnalysisAgent
             artifacts; wrap dicts in SimpleNamespace for dot-access in template.
          4. Render report_skeleton.tex.j2 via LatexTemplateLoader; write atomically.
          5. Compile via latexmk when available; count [?] markers in the PDF;
             raise UnresolvedCitationError if > 0.
          6. Populate state.report_path (relative) + state.report_stats + append
             StageReport.
        """
        t0 = time.perf_counter()
        run_dir = Path(state.runs_dir) / state.run_id
        run_dir.mkdir(parents=True, exist_ok=True)

        candidate_papers = [
            p for p in state.papers if p.analysis_status in {"succeeded", None}
        ]
        base_keys = [
            make_bibtex_key(
                (p.authors[0].split()[-1] if p.authors else "unknown"),
                p.year,
                p.title,
            )
            for p in candidate_papers
        ]
        resolved = resolve_key_collisions(base_keys)
        collision_fixes = sum(
            1 for b, r in zip(base_keys, resolved, strict=True) if b != r
        )
        for p, key in zip(candidate_papers, resolved, strict=True):
            p.bibtex_key = key

        bib_entries: list[str] = []
        for p, key in zip(candidate_papers, resolved, strict=True):
            author_names = " and ".join(p.authors) if p.authors else "Unknown"
            bib_entries.append(
                f"@article{{{key},\n"
                f"  title={{{{{p.title}}}}},\n"
                f"  author={{{author_names}}},\n"
                f"  year={{{p.year if p.year else 'n.d.'}}},\n"
                f"  journal={{{p.venue if p.venue else 'Preprint'}}},\n"
                f"}}"
            )
        bib_text = "\n\n".join(bib_entries) + ("\n" if bib_entries else "")
        bib_path = run_dir / "references.bib"
        atomic_write_text(bib_text, bib_path)

        summaries_list: list[dict[str, Any]] = []
        summaries_dir = run_dir / "summaries"
        if summaries_dir.exists():
            for p in candidate_papers:
                summary_path = summaries_dir / f"{p.paper_id}.json"
                if not summary_path.exists():
                    continue
                data = json.loads(summary_path.read_text(encoding="utf-8"))
                summaries_list.append(
                    {
                        "paper": p,
                        "summary": data["summary"],
                        "methodology": data.get("methodology"),
                        "limitations": data.get("limitations"),
                    }
                )

        matrix_data: dict[str, Any] | None = None
        gaps_data: dict[str, Any] | None = None
        if state.comparison_path:
            cp = Path(state.comparison_path)
            if not cp.is_absolute():
                cp = Path(state.runs_dir) / state.comparison_path
            if cp.exists():
                matrix_data = json.loads(cp.read_text(encoding="utf-8"))
        if state.gaps_path:
            gp = Path(state.gaps_path)
            if not gp.is_absolute():
                gp = Path(state.runs_dir) / state.gaps_path
            if gp.exists():
                gaps_data = json.loads(gp.read_text(encoding="utf-8"))

        matrix_ns = (
            types.SimpleNamespace(rows=matrix_data["rows"]) if matrix_data else None
        )
        gaps_ns = (
            types.SimpleNamespace(gaps=gaps_data["gaps"]) if gaps_data else None
        )

        wrapped_summaries = []
        for entry in summaries_list:
            summary_dict = entry["summary"]
            wrapped_summaries.append(
                types.SimpleNamespace(
                    paper=entry["paper"],
                    summary=types.SimpleNamespace(
                        objective=types.SimpleNamespace(**summary_dict["objective"]),
                        methodology=types.SimpleNamespace(
                            **summary_dict["methodology"]
                        ),
                        findings=[
                            types.SimpleNamespace(**f)
                            for f in summary_dict["findings"]
                        ],
                        limitations=[
                            types.SimpleNamespace(**lim)
                            for lim in summary_dict["limitations"]
                        ],
                    ),
                )
            )

        rendered = self.latex_templates.render(
            "report_skeleton.tex.j2",
            state=state,
            summaries=wrapped_summaries,
            matrix=matrix_ns,
            gaps=gaps_ns,
        )
        tex_path = run_dir / "report.tex"
        atomic_write_text(rendered, tex_path)

        unresolved = 0
        pdf_bytes = 0
        latexmk_duration_ms = 0
        if latexmk_available():
            compile_t0 = time.perf_counter()
            try:
                pdf_path = compile_latex(tex_path, out_dir=run_dir, timeout_s=120)
            except LatexCompileError:
                latexmk_duration_ms = int((time.perf_counter() - compile_t0) * 1000)
                raise
            latexmk_duration_ms = int((time.perf_counter() - compile_t0) * 1000)
            unresolved = count_unresolved_citations(pdf_path)
            if unresolved > 0:
                raise UnresolvedCitationError(
                    f"Compiled PDF {pdf_path.name} has {unresolved} "
                    f"unresolved-citation markers ([?])"
                )
            try:
                pdf_bytes = pdf_path.stat().st_size
            except OSError:
                pdf_bytes = 0

        try:
            state.report_path = str(tex_path.relative_to(Path(state.runs_dir)))
        except ValueError:
            state.report_path = str(tex_path)

        state.report_stats = ReportStats(
            latex_bytes=len(rendered.encode("utf-8")),
            pdf_bytes=pdf_bytes,
            bibtex_keys=len(resolved),
            bibtex_collision_fixes=collision_fixes,
            latexmk_duration_ms=latexmk_duration_ms,
            unresolved_citations=unresolved,
        )

        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=len(candidate_papers),
                outputs_count=1,
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )
        return state


__all__ = ["ReportAgent"]
