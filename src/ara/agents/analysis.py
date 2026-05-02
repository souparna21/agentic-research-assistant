"""AnalysisAgent — hierarchical structured-output analysis with per-claim verifier.

ANL-01..09. Flow (per run):

    1. Load FAISS index from ``state.index_path`` via :class:`FaissVectorStore`.
    2. For each paper with ``extraction_ok=True`` and ``full_text_available=True``:
         a. Retrieve top-k passages via ``store.search(...)``; scope to this paper.
         b. 3 per-paper structured-output calls (schema=<BaseModel>, temperature=0.0):
              - ``per_paper_summary.j2``           -> :class:`PerPaperSummary`
              - ``methodology_extraction.j2``      -> :class:`MethodologyExtraction`
              - ``limitations_extraction.j2``      -> :class:`LimitationsExtraction`
         c. 1 verifier call (``verifier.j2`` -> :class:`VerifierVerdict`) scoring
            each claim in the summary against the cited passages.
         d. Persist ``runs/<run_id>/summaries/<paper_id>.json`` with
            ``[UNVERIFIED]`` prepended to any claim the verifier marked unsupported.
         e. On narrow exception (``ValidationError``, ``UnknownPromptError``,
            ``TimeoutError``, LiteLLM ``APIError``): set
            ``paper.analysis_status='failed'``, append ``(paper_id, reason)`` to
            ``AnalysisStats.analysis_failures``, and continue to the next paper.
    3. After the per-paper loop (over succeeded papers only):
         a. ``methodology_matrix_synthesis.j2`` -> :class:`MethodologyMatrix`
            (consumes the TYPED per-paper extractions — NEVER raw passages; this
            is the Belem-2025 / pitfall-P7 hierarchical-synthesis defense.)
         b. ``gap_synthesis.j2`` -> :class:`GapAnalysis`
    4. Persist ``runs/<run_id>/comparison.json`` + ``gaps.json``; populate
       ``state.summaries_dir`` / ``comparison_path`` / ``gaps_path`` /
       ``analysis_stats``; append a :class:`StageReport`.

Budget: 4N + 2 LLM calls per run (N = count of succeeded papers).
All calls use ``temperature=0.0`` — grep assertable.

Per-paper failure isolation (ANL-09): narrow try/except list
``(pydantic.ValidationError, UnknownPromptError, TimeoutError, litellm.APIError)``.
We NEVER catch bare ``Exception`` — pitfall P8 discipline, same as
Phase 2 :class:`ara.agents.extraction.ExtractionAgent`.

Downstream:
  * Plan 03-08 ReportAgent reads ``state.summaries_dir`` (per-paper JSON),
    ``state.comparison_path`` (MethodologyMatrix), and ``state.gaps_path``
    (GapAnalysis). The ``[UNVERIFIED]`` tags are already applied in-place
    inside the per-paper summary JSON.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import pydantic
import structlog

from ara.analysis_schemas import (
    Claim,
    GapAnalysis,
    LimitationsExtraction,
    MethodologyExtraction,
    MethodologyMatrix,
    PerPaperSummary,
    VerifierVerdict,
)
from ara.errors import UnknownPromptError
from ara.persistence import atomic_write_text
from ara.services.embed import MiniLMEmbedder
from ara.services.llm import LLMClient
from ara.services.prompts import PromptLoader
from ara.services.vector_store import FaissVectorStore
from ara.state import AnalysisStats, PipelineState, StageReport

log = structlog.get_logger(__name__)


try:
    from litellm.exceptions import APIError

    _PER_PAPER_EXC: tuple[type[Exception], ...] = (
        pydantic.ValidationError,
        UnknownPromptError,
        TimeoutError,
        APIError,
    )
except ImportError:
    _PER_PAPER_EXC = (
        pydantic.ValidationError,
        UnknownPromptError,
        TimeoutError,
    )


@dataclass
class AnalysisAgent:
    """Stage-Protocol conformant. Required deps: ``llm``, ``prompts``.

    FIELD_OWNERS["analysis"] = {papers, summaries_dir, comparison_path,
    gaps_path, analysis_stats}. Nested mutation of ``papers[i].analysis_status``
    is permitted — the ``papers`` top-level co-ownership in FIELD_OWNERS
    authorizes it.
    """

    llm: LLMClient
    prompts: PromptLoader
    name: str = "analysis"
    top_k: int = 8

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "summaries" / ".analysis.marker"

    def run(self, state: PipelineState) -> PipelineState:  # noqa: C901 — orchestration
        t0 = time.perf_counter()

        if not state.index_path:
            raise RuntimeError(
                "AnalysisAgent requires state.index_path from IndexingAgent (plan 03-04)"
            )
        idx_path = Path(state.index_path)
        if not idx_path.is_absolute():
            idx_path = Path(state.runs_dir) / state.index_path
        chunks_path = idx_path.parent / "chunks.json"
        store = FaissVectorStore(embedder=MiniLMEmbedder())
        store.load(idx_path, chunks_path)

        per_paper_calls = 0
        verifier_calls = 0
        unsupported_claims_total = 0
        analysis_failures: list[tuple[str, str]] = []

        summaries_dir = Path(state.runs_dir) / state.run_id / "summaries"
        summaries_dir.mkdir(parents=True, exist_ok=True)
        summaries_dir_rel = str(summaries_dir.relative_to(Path(state.runs_dir)))

        per_paper_extractions: list[dict[str, object]] = []
        per_paper_limitations: list[dict[str, object]] = []
        succeeded_papers: list[object] = []

        for paper in state.papers:
            if not paper.extraction_ok or not paper.full_text_available:
                paper.analysis_status = "skipped"
                continue
            try:
                retrieved_all = store.search(
                    f"What is the objective, methodology, findings, limitations of "
                    f"'{paper.title}'?",
                    k=self.top_k,
                )
                scoped = [c for c in retrieved_all if c.paper_id == paper.paper_id]
                effective = scoped or retrieved_all
                passages = [
                    {
                        "paper_id": c.paper_id,
                        "section": c.section,
                        "score": c.score,
                        "text": c.text,
                    }
                    for c in effective
                ]

                prompt = self.prompts.render(
                    "per_paper_summary.j2", paper=paper, passages=passages
                )
                resp = self.llm.complete(
                    prompt, schema=PerPaperSummary, temperature=0.0
                )
                per_paper_calls += 1
                summary = PerPaperSummary.model_validate_json(resp.text)

                prompt = self.prompts.render(
                    "methodology_extraction.j2", paper=paper, passages=passages
                )
                resp = self.llm.complete(
                    prompt, schema=MethodologyExtraction, temperature=0.0
                )
                per_paper_calls += 1
                methodology = MethodologyExtraction.model_validate_json(resp.text)

                prompt = self.prompts.render(
                    "limitations_extraction.j2", paper=paper, passages=passages
                )
                resp = self.llm.complete(
                    prompt, schema=LimitationsExtraction, temperature=0.0
                )
                per_paper_calls += 1
                limitations = LimitationsExtraction.model_validate_json(resp.text)

                verifier_passages = [
                    {"paper_id": p["paper_id"], "text": p["text"]} for p in passages
                ]
                prompt = self.prompts.render(
                    "verifier.j2",
                    summary=summary.model_dump(),
                    passages=verifier_passages,
                )
                resp = self.llm.complete(
                    prompt, schema=VerifierVerdict, temperature=0.0
                )
                verifier_calls += 1
                verdict = VerifierVerdict.model_validate_json(resp.text)

                unsupported_texts = {
                    cv.claim_text for cv in verdict.claims if cv.verdict == "unsupported"
                }
                unsupported_claims_total += len(unsupported_texts)
                tagged_summary = _tag_unsupported(summary, unsupported_texts)

                out = {
                    "paper_id": paper.paper_id,
                    "summary": json.loads(tagged_summary.model_dump_json()),
                    "methodology": json.loads(methodology.model_dump_json()),
                    "limitations": json.loads(limitations.model_dump_json()),
                    "verifier": json.loads(verdict.model_dump_json()),
                }
                atomic_write_text(
                    json.dumps(out, indent=2, sort_keys=True),
                    summaries_dir / f"{paper.paper_id}.json",
                )

                paper.analysis_status = "succeeded"
                per_paper_extractions.append(
                    {
                        "paper_id": paper.paper_id,
                        "paper_title": paper.title,
                        "datasets": methodology.datasets,
                        "techniques": methodology.techniques,
                        "metrics": methodology.metrics,
                    }
                )
                per_paper_limitations.append(
                    json.loads(limitations.model_dump_json())
                )
                succeeded_papers.append(paper)

            except _PER_PAPER_EXC as exc:
                paper.analysis_status = "failed"
                reason = f"{type(exc).__name__}: {str(exc)[:200]}"
                analysis_failures.append((paper.paper_id, reason))
                log.warning(
                    "analysis_paper_failed",
                    paper_id=paper.paper_id,
                    reason=type(exc).__name__,
                )
                continue

        cross_paper_calls = 0
        matrix: MethodologyMatrix | None = None
        gaps: GapAnalysis | None = None
        if succeeded_papers:
            prompt = self.prompts.render(
                "methodology_matrix_synthesis.j2",
                extractions=per_paper_extractions,
            )
            resp = self.llm.complete(
                prompt, schema=MethodologyMatrix, temperature=0.0
            )
            cross_paper_calls += 1
            matrix = MethodologyMatrix.model_validate_json(resp.text)

            prompt = self.prompts.render(
                "gap_synthesis.j2",
                matrix=json.loads(matrix.model_dump_json()),
                limitations=per_paper_limitations,
            )
            resp = self.llm.complete(prompt, schema=GapAnalysis, temperature=0.0)
            cross_paper_calls += 1
            gaps = GapAnalysis.model_validate_json(resp.text)

        run_dir = Path(state.runs_dir) / state.run_id
        if matrix is not None:
            comparison_path = run_dir / "comparison.json"
            atomic_write_text(matrix.model_dump_json(indent=2), comparison_path)
            state.comparison_path = str(
                comparison_path.relative_to(Path(state.runs_dir))
            )
        if gaps is not None:
            gaps_path = run_dir / "gaps.json"
            atomic_write_text(gaps.model_dump_json(indent=2), gaps_path)
            state.gaps_path = str(gaps_path.relative_to(Path(state.runs_dir)))

        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")

        state.summaries_dir = summaries_dir_rel
        state.analysis_stats = AnalysisStats(
            per_paper_calls=per_paper_calls,
            cross_paper_calls=cross_paper_calls,
            verifier_calls=verifier_calls,
            unsupported_claims=unsupported_claims_total,
            analysis_failures=analysis_failures,
            duration_ms=int((time.perf_counter() - t0) * 1000),
        )
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=len(state.papers),
                outputs_count=len(succeeded_papers),
                errors_count=len(analysis_failures),
                duration_ms=(time.perf_counter() - t0) * 1000.0,
            )
        )
        return state


def _tag_unsupported(
    summary: PerPaperSummary, unsupported_texts: set[str]
) -> PerPaperSummary:
    """Prepend ``[UNVERIFIED]`` to any claim whose text is in ``unsupported_texts``.

    ANL-08 implementation — called once per per-paper summary after the
    verifier call. The prepended tag is consumed verbatim by plan 03-08's
    ReportAgent LaTeX rendering.
    """

    def _tag(claim: Claim) -> Claim:
        if claim.text in unsupported_texts:
            return Claim(
                text="[UNVERIFIED] " + claim.text, citations=list(claim.citations)
            )
        return claim

    return PerPaperSummary(
        paper_id=summary.paper_id,
        objective=_tag(summary.objective),
        methodology=_tag(summary.methodology),
        findings=[_tag(c) for c in summary.findings],
        limitations=[_tag(c) for c in summary.limitations],
    )
