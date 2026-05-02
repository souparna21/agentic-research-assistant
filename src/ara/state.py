"""PipelineState — the single cross-stage communication channel.

Every field is owned by exactly ONE stage (see FIELD_OWNERS dict below).
The orchestrator (src/ara/orchestrator.py, plan 05) enforces this via
pre-stage / post-stage snapshot diff. Adding a field requires:
    1. Adding the attribute here
    2. Adding it to the appropriate FIELD_OWNERS entry
    3. Updating tests/fixtures/golden_state.json if smoke-test-visible
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StageReport(BaseModel):
    """Per-stage telemetry appended to PipelineState.stage_reports."""

    stage: str
    inputs_count: int = 0
    outputs_count: int = 0
    errors_count: int = 0
    skipped_count: int = 0
    duration_ms: float = 0.0
    warnings: list[str] = Field(default_factory=list)


class RetrievalStats(BaseModel):
    """Typed retrieval telemetry (replaces the previous ``dict`` shape).

    Writer: RetrievalAgent (plan 02-03 / 02-04). Phase 3 RPT-08 renders
    these fields into the final LaTeX report appendix.
    """

    model_config = {"extra": "forbid"}

    search_terms_submitted: int = 0
    s2_requests: int = 0
    s2_429_retries: int = 0
    s2_total_results: int = 0
    arxiv_fallback_hits: int = 0
    pdf_download_count: int = 0
    pdf_download_bytes: int = 0
    pdf_download_failures: list[tuple[str, str]] = Field(default_factory=list)
    abstract_only_count: int = 0
    duration_ms: int = 0
    empty_query_terms: list[str] = Field(default_factory=list)


class ExtractionStats(BaseModel):
    """Typed extraction telemetry (replaces the previous ``dict`` shape).

    Writer: ExtractionAgent (plan 02-05 / 02-06). Phase 3 RPT-08 renders
    these fields into the final LaTeX report appendix.
    """

    model_config = {"extra": "forbid"}

    pdfs_attempted: int = 0
    pdfs_extracted_ok: int = 0
    extraction_failures: list[tuple[str, str]] = Field(default_factory=list)
    hidden_text_spans_stripped: int = 0
    header_footer_lines_stripped: int = 0
    duration_ms: int = 0


class IndexingStats(BaseModel):
    """Typed indexing telemetry. Writer: IndexingAgent (plan 03-04).

    Populated as chunk/embed/build/persist sub-steps execute; surfaced in the
    StageReport appendix of the final LaTeX report (RPT-08).
    """

    model_config = {"extra": "forbid"}

    papers_chunked: int = 0
    total_chunks: int = 0
    unique_chunks: int = 0
    duplicate_chunks_dropped: int = 0
    embedding_duration_ms: int = 0
    index_build_duration_ms: int = 0
    index_bytes: int = 0


class AnalysisStats(BaseModel):
    """Typed analysis telemetry. Writer: AnalysisAgent (plan 03-06).

    Tracks the 4N+2 LLM-call budget plus verifier calls + unsupported-claim
    count; `analysis_failures` holds per-paper (paper_id, reason) tuples for
    papers whose analysis isolated-failed (ANL-09 narrow-catch path).
    """

    model_config = {"extra": "forbid"}

    per_paper_calls: int = 0
    cross_paper_calls: int = 0
    verifier_calls: int = 0
    unsupported_claims: int = 0
    analysis_failures: list[tuple[str, str]] = Field(default_factory=list)
    duration_ms: int = 0


class ReportStats(BaseModel):
    """Typed report telemetry. Writer: ReportAgent (plan 03-08).

    `unresolved_citations` MUST be 0 at the phase gate (RPT-07) — scanned
    post-compile via ``pymupdf.open(report.pdf).get_text().count('[?]')``.
    Any non-zero hit raises ReportStageError before the stage returns.
    """

    model_config = {"extra": "forbid"}

    latex_bytes: int = 0
    pdf_bytes: int = 0
    bibtex_keys: int = 0
    bibtex_collision_fixes: int = 0
    latexmk_duration_ms: int = 0
    unresolved_citations: int = 0


class ExtractionMeta(BaseModel):
    """Per-paper extraction diagnostics — consumed by Phase 3 chunker (EXT-04).

    Written by ExtractionAgent alongside the extracted.md artifact.
    ``num_detected_sections`` gates the Phase 3 chunker's fallback-to-
    paragraph-boundary behavior (IDX-02 / EXT-04 contract).
    """

    model_config = {"extra": "forbid"}

    num_detected_sections: int = 0
    header_footer_lines_stripped: int = 0
    pages: int = 0


class Paper(BaseModel):
    """Single paper record. Written by retrieval; extended by extraction."""

    paper_id: str
    title: str
    authors: list[str] = Field(default_factory=list)
    year: int | None = None
    venue: str | None = None
    citation_count: int | None = None
    abstract: str | None = None
    pdf_url: str | None = None
    pdf_path: str | None = None
    pdf_source: str | None = None
    extracted_md_path: str | None = None
    extraction_ok: bool = False
    fallback_to_abstract: bool = False
    full_text_available: bool = False
    analysis_status: str | None = None
    """'succeeded' | 'failed' | None (not-yet-run). Writer: AnalysisAgent (plan 03-06).

    Nested-field co-ownership: AnalysisAgent is permitted to mutate this on each
    Paper in state.papers; the list shape itself remains retrieval/extraction-owned.
    FIELD_OWNERS['analysis'] co-owns 'papers' at top level so the orchestrator's
    snapshot-diff tolerates the nested write.
    """

    bibtex_key: str | None = None
    """Resolved BibTeX key (unique after collision resolution). Writer: ReportAgent
    (plan 03-08). Format: ``<first_author_surname>_<year>_<first_title_word>``
    (lowercased, ASCII via unidecode); on collision suffix ``_2``, ``_3``.

    Nested-field co-ownership: ReportAgent mutates this per Paper; FIELD_OWNERS
    ['report'] co-owns 'papers' at top level so orchestrator snapshot-diff passes.
    """

    extraction_meta: ExtractionMeta | None = None
    """Per-paper extraction diagnostics. Writer: ExtractionAgent (plan 02-06)."""


Paper.model_rebuild()


class PipelineState(BaseModel):
    """The single source of truth for one pipeline run.

    Orchestrator writes identity fields at init; each stage writes only its
    owned fields (per FIELD_OWNERS). Persisted to runs/<run_id>/state.json.
    """

    model_config = {"extra": "forbid"}

    run_id: str
    """YYYYMMDD-HHMMSS-<slug>. Writer: Orchestrator (at init)."""

    question: str
    """Verbatim user question. Writer: Orchestrator (at init)."""

    config_snapshot: dict
    """{model, temperature, git_sha, ...}. Writer: Orchestrator (at init)."""

    runs_dir: str = "runs"
    """Root runs directory. Writer: Orchestrator (at init)."""

    search_terms: list[str] = Field(default_factory=list)
    """3–5 search terms. Writer: QueryAgent."""

    papers: list[Paper] = Field(default_factory=list)
    """Retrieved + deduplicated paper records. Writer: RetrievalAgent."""

    retrieval_stats: RetrievalStats = Field(default_factory=RetrievalStats)
    """{search_terms_submitted, s2_requests, s2_429_retries, ...}. Writer: RetrievalAgent."""

    extraction_stats: ExtractionStats = Field(default_factory=ExtractionStats)
    """{pdfs_attempted, pdfs_extracted_ok, ...}. Writer: ExtractionAgent."""

    chunks_count: int = 0
    """Total chunks indexed. Writer: IndexingAgent."""

    index_path: str | None = None
    """Path to runs/<run_id>/index/faiss.index. Writer: IndexingAgent."""

    index_meta: dict = Field(default_factory=dict)
    """{dim, model, built_at, cache_key}. Writer: IndexingAgent."""

    indexing_stats: IndexingStats = Field(default_factory=IndexingStats)
    """{papers_chunked, total_chunks, unique_chunks, duplicate_chunks_dropped,
    embedding_duration_ms, index_build_duration_ms, index_bytes}.
    Writer: IndexingAgent (plan 03-04)."""

    summaries_dir: str | None = None
    """Path to runs/<run_id>/summaries/. Writer: AnalysisAgent."""

    comparison_path: str | None = None
    """Path to comparison.json. Writer: AnalysisAgent."""

    gaps_path: str | None = None
    """Path to gaps.json. Writer: AnalysisAgent."""

    analysis_stats: AnalysisStats = Field(default_factory=AnalysisStats)
    """{per_paper_calls, cross_paper_calls, verifier_calls, unsupported_claims,
    analysis_failures, duration_ms}. Writer: AnalysisAgent (plan 03-06)."""

    report_path: str | None = None
    """Path to report.tex. Writer: ReportAgent."""

    report_stats: ReportStats = Field(default_factory=ReportStats)
    """{latex_bytes, pdf_bytes, bibtex_keys, bibtex_collision_fixes,
    latexmk_duration_ms, unresolved_citations}. Writer: ReportAgent (plan 03-08)."""

    stage_reports: list[StageReport] = Field(default_factory=list)
    """Per-stage telemetry. Any stage may append its own StageReport.
    Ownership enforcement treats stage_reports as a shared append-only log."""


FIELD_OWNERS: dict[str, frozenset[str]] = {
    "query": frozenset({"search_terms"}),
    "retrieval": frozenset({"papers", "retrieval_stats"}),
    "extraction": frozenset({"papers", "extraction_stats"}),
    "indexing": frozenset({"chunks_count", "index_path", "index_meta", "indexing_stats"}),
    "analysis": frozenset(
        {"papers", "summaries_dir", "comparison_path", "gaps_path", "analysis_stats"}
    ),
    "report": frozenset({"papers", "report_path", "report_stats"}),
}
"""Stage → fields that stage is allowed to write.

`papers` is CO-OWNED by FOUR stages at DIFFERENT nesting levels. The orchestrator's
snapshot-diff is TOP-LEVEL only; it cannot enforce the nested split — adding a
stage to the `papers` co-ownership list authorizes ONLY that stage's documented
nested writes. Convention (code-review + unit tests are the check):

  - retrieval: mutates state.papers at the LIST level (assigns a fresh list);
    writes S2 metadata, pdf_path, pdf_source, fallback_to_abstract
  - extraction: mutates nested extraction_* subset per paper —
    extracted_md_path, extraction_ok, fallback_to_abstract, extraction_meta
  - analysis (Phase 3): mutates nested analysis_status per paper
    (``'succeeded'`` | ``'failed'`` | ``'skipped'``) — same pattern as
    extraction's nested mutation; writer: AnalysisAgent (plan 03-06)
  - report (Phase 3): mutates nested bibtex_key per paper after BibTeX
    collision resolution; writer: ReportAgent (plan 03-08)

The four-way co-ownership lets the full pipeline (Phase-3 close) pass the
orchestrator's snapshot-diff invariant without widening enforcement to
nested fields. The post-Phase-3 smoke test (`test_smoke_produces_artifacts`)
exercises all six real agents end-to-end with `_run_stage` around each
invocation and green-gates this split.

Cross-violations (e.g., retrieval mutating extraction_meta, analysis
removing paper entries, report re-assigning `state.papers`) are NOT caught
by the orchestrator — code review + unit tests are the check. Each plan's
unit tests assert the narrow ownership boundaries.

Orchestrator (plan 01-05) snapshots non-owned fields before each stage
runs and asserts they are unchanged afterward. `stage_reports` is
intentionally NOT in any stage's owned set — it's append-only and
orchestrator handles the "append allowed, modify-prior-entries forbidden"
check separately.

Identity fields (run_id, question, config_snapshot, runs_dir) are owned
by the orchestrator itself at init time and immutable thereafter.
"""
