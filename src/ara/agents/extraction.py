"""ExtractionAgent — per-paper PDF → Markdown with graceful degradation (EXT-05, EXT-06).

For every paper in state.papers with pdf_path populated:
    1. Call PdfExtractor.extract(Path(pdf_path))
    2. On success: atomic_write_text(md, runs/<run_id>/papers/<paper_id>/extracted.md),
       set paper.extracted_md_path (relative), paper.extraction_ok=True,
       paper.extraction_meta = ExtractionMeta(...).
    3. On narrow failure (ExtractionTooShort, pymupdf.FileDataError, OSError,
       IOError, ValueError): set paper.extraction_ok=False, paper.fallback_to_abstract=True,
       append (paper_id, reason) to extraction_stats.extraction_failures. CONTINUE.
    4. Papers with pdf_path=None are skipped (retrieval already tagged them).

Writes TOP-LEVEL state.extraction_stats only. Nested Paper mutations are
part of the shared papers list — the orchestrator's snapshot-diff check
will flag this as a cross-ownership write unless FIELD_OWNERS is extended
in plan 02-07 OR the orchestrator is taught about nested-field ownership.
This plan's unit tests call .run() directly (not via orchestrator), so
the invariant check is deferred to the plan 02-07 smoke-test gate.

Narrow exception list rationale:
    - ExtractionTooShort: raised by PdfExtractor when < min_output_chars extracted
    - pymupdf.FileDataError: raised by pymupdf.open on malformed/truncated PDFs
    - OSError / IOError: filesystem failures reading the PDF or writing markdown
    - ValueError: pymupdf may raise on broken byte streams or invalid coordinates
We never catch bare Exception — unexpected bugs MUST fail loud rather than
silently masking a paper as "fallback_to_abstract".
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import structlog

from ara.persistence import atomic_write_text
from ara.services.pdf import ExtractionTooShort, PdfExtractor
from ara.state import ExtractionMeta, ExtractionStats, PipelineState, StageReport

log = structlog.get_logger(__name__)


@dataclass
class ExtractionAgent:
    """Implements the Stage protocol. FIELD_OWNERS-owned: extraction_stats.

    Nested Paper mutations (extraction_meta, extraction_ok, fallback_to_abstract,
    extracted_md_path) are intentional — retrieval owns list shape + Paper
    identity fields; extraction owns per-Paper extraction_* fields. Plan 02-07
    will formalize this in FIELD_OWNERS or the orchestrator snapshot-diff.
    """

    pdf_extractor: PdfExtractor
    name: str = "extraction"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "extraction.marker"

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.perf_counter()
        pdfs_attempted = 0
        pdfs_extracted_ok = 0
        total_hidden_stripped = 0
        total_hdrftr_stripped = 0
        failures: list[tuple[str, str]] = []

        runs_dir = Path(state.runs_dir)
        for paper in state.papers:
            if not paper.pdf_path:
                continue
            pdfs_attempted += 1
            pdf_path = Path(paper.pdf_path)
            try:
                result = self.pdf_extractor.extract(pdf_path)
            except (
                ExtractionTooShort,
                pymupdf.FileDataError,
                OSError,
                IOError,
                ValueError,
            ) as exc:
                log.warning(
                    "extraction_per_paper_failed",
                    paper_id=paper.paper_id,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                failures.append((paper.paper_id, f"{type(exc).__name__}: {exc}"))
                paper.extraction_ok = False
                paper.fallback_to_abstract = True
                continue

            out_path = runs_dir / state.run_id / "papers" / paper.paper_id / "extracted.md"
            try:
                atomic_write_text(result.markdown, out_path)
            except OSError as exc:
                log.warning(
                    "extraction_write_failed",
                    paper_id=paper.paper_id,
                    error=str(exc),
                )
                failures.append((paper.paper_id, f"write-failed: {exc}"))
                paper.extraction_ok = False
                paper.fallback_to_abstract = True
                continue

            try:
                rel = out_path.relative_to(runs_dir)
                paper.extracted_md_path = str(rel)
            except ValueError:
                paper.extracted_md_path = str(out_path)
            paper.extraction_ok = True
            paper.fallback_to_abstract = False
            paper.extraction_meta = ExtractionMeta(
                num_detected_sections=result.num_detected_sections,
                header_footer_lines_stripped=result.header_footer_lines_stripped,
                pages=result.pages,
            )
            pdfs_extracted_ok += 1
            total_hidden_stripped += result.hidden_text_spans_stripped
            total_hdrftr_stripped += result.header_footer_lines_stripped

        duration_ms = int((time.perf_counter() - t0) * 1000)
        state.extraction_stats = ExtractionStats(
            pdfs_attempted=pdfs_attempted,
            pdfs_extracted_ok=pdfs_extracted_ok,
            extraction_failures=failures,
            hidden_text_spans_stripped=total_hidden_stripped,
            header_footer_lines_stripped=total_hdrftr_stripped,
            duration_ms=duration_ms,
        )
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=pdfs_attempted,
                outputs_count=pdfs_extracted_ok,
                errors_count=len(failures),
                duration_ms=float(duration_ms),
            )
        )

        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state
