"""Tests for PipelineState schema, round-trip, and field ownership (FND-02, FND-03)."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError


def test_pipeline_state_imports():
    """Sanity: the module imports and exposes the expected names."""
    from ara.state import FIELD_OWNERS, Paper, PipelineState, StageReport

    assert PipelineState is not None
    assert Paper is not None
    assert StageReport is not None
    assert isinstance(FIELD_OWNERS, dict)


def test_extra_forbid_rejects_unknown_fields():
    """model_config extra='forbid' — typos in field names must error, not silently no-op."""
    from ara.state import PipelineState

    with pytest.raises(ValidationError):
        PipelineState(
            run_id="20260417-120000-test",
            question="q",
            config_snapshot={},
            totally_bogus_field=1,  # type: ignore[call-arg]
        )


def test_field_owners_complete():
    """FIELD_OWNERS covers all six stages and every owned field is a real attribute."""
    from ara.state import FIELD_OWNERS, PipelineState

    expected_stages = {"query", "retrieval", "extraction", "indexing", "analysis", "report"}
    assert set(FIELD_OWNERS.keys()) == expected_stages

    state_fields = set(PipelineState.model_fields.keys())
    for stage, owned in FIELD_OWNERS.items():
        assert isinstance(owned, frozenset), f"{stage} owned must be frozenset"
        missing = owned - state_fields
        assert not missing, f"stage {stage} claims fields not on PipelineState: {missing}"


def test_roundtrip(tmp_path: Path):
    """FND-03: PipelineState serializes to state.json and back losslessly."""
    from ara.persistence import atomic_write_json, load_state
    from ara.state import Paper, PipelineState, RetrievalStats, StageReport

    state = PipelineState(
        run_id="20260417-120000-roundtrip",
        question="What is RAG?",
        config_snapshot={
            "model": "gemini/gemini-2.5-flash",
            "temperature": 0.0,
            "git_sha": "abc1234",
        },
        search_terms=["retrieval-augmented generation", "RAG"],
        papers=[
            Paper(
                paper_id="s2:abc",
                title="A paper",
                authors=["Smith"],
                year=2024,
                pdf_source="arxiv",
            )
        ],
        retrieval_stats=RetrievalStats(s2_requests=5, s2_total_results=2),
        stage_reports=[
            StageReport(stage="query", outputs_count=2, duration_ms=12.3),
            StageReport(stage="retrieval", outputs_count=2, duration_ms=500.1),
        ],
    )
    path = tmp_path / "runs" / state.run_id / "state.json"
    atomic_write_json(state, path)
    reloaded = load_state(path)
    assert reloaded == state


def test_field_ownership_violation_detected(sample_state):
    """FND-02: a stage that writes a field it does not own triggers FieldOwnershipViolation.

    This test depends on orchestrator._run_stage (plan 05); skip until present.
    """
    run_stage = pytest.importorskip("ara.orchestrator")._run_stage
    from ara.errors import FieldOwnershipViolation
    from ara.state import Paper

    class BadQueryStage:
        name = "query"

        def run(self, state):
            state.search_terms = ["ok"]
            state.papers.append(Paper(paper_id="X", title="stolen"))
            return state

    with pytest.raises(FieldOwnershipViolation, match="papers"):
        run_stage(BadQueryStage(), sample_state)




def test_retrieval_stats_is_typed_basemodel():
    """02-00: retrieval_stats is a typed RetrievalStats BaseModel, not a dict."""
    from ara.state import PipelineState, RetrievalStats

    s = PipelineState(run_id="r", question="q", config_snapshot={}, runs_dir="runs")
    assert isinstance(s.retrieval_stats, RetrievalStats)
    assert s.retrieval_stats.s2_requests == 0
    assert s.retrieval_stats.empty_query_terms == []


def test_extraction_stats_is_typed_basemodel():
    """02-00: extraction_stats is a typed ExtractionStats BaseModel, not a dict."""
    from ara.state import ExtractionStats, PipelineState

    s = PipelineState(run_id="r", question="q", config_snapshot={}, runs_dir="runs")
    assert isinstance(s.extraction_stats, ExtractionStats)
    assert s.extraction_stats.pdfs_extracted_ok == 0


def test_paper_extraction_meta_optional():
    """02-00: Paper.extraction_meta is an optional ExtractionMeta BaseModel."""
    from ara.state import ExtractionMeta, Paper

    p = Paper(paper_id="p1", title="T")
    assert p.extraction_meta is None
    p.extraction_meta = ExtractionMeta(num_detected_sections=3, pages=12)
    assert p.extraction_meta.num_detected_sections == 3


def test_typed_stats_roundtrip_via_json():
    """02-00: typed stats survive model_dump_json -> model_validate_json."""
    from ara.state import ExtractionStats, PipelineState, RetrievalStats

    s = PipelineState(
        run_id="r",
        question="q",
        config_snapshot={},
        runs_dir="runs",
        retrieval_stats=RetrievalStats(s2_requests=5, empty_query_terms=["foo"]),
        extraction_stats=ExtractionStats(pdfs_extracted_ok=2),
    )
    dumped = s.model_dump_json()
    loaded = PipelineState.model_validate_json(dumped)
    assert loaded.retrieval_stats.s2_requests == 5
    assert loaded.retrieval_stats.empty_query_terms == ["foo"]
    assert loaded.extraction_stats.pdfs_extracted_ok == 2


def test_field_owners_still_covers_extended_stats():
    """02-00: FIELD_OWNERS ownership unchanged — retrieval owns retrieval_stats,
    extraction owns extraction_stats, stage_reports remains append-only (no owner).
    """
    from ara.state import FIELD_OWNERS

    assert "retrieval_stats" in FIELD_OWNERS["retrieval"]
    assert "extraction_stats" in FIELD_OWNERS["extraction"]
    assert "search_terms" in FIELD_OWNERS["query"]
    for stage_name, owned in FIELD_OWNERS.items():
        assert (
            "stage_reports" not in owned
        ), f"stage_reports must not be owned by any stage; found in {stage_name}"


def test_extraction_owns_papers_for_nested_mutation() -> None:
    """After plan 02-07: extraction co-owns `papers` at top level so nested Paper mutations
    don't trip the orchestrator's snapshot-diff. See FIELD_OWNERS docstring for the
    semantic split between retrieval (list shape) and extraction (nested extraction_* fields)."""
    from ara.state import FIELD_OWNERS

    assert "papers" in FIELD_OWNERS["extraction"]
    assert "papers" in FIELD_OWNERS["retrieval"]




def test_indexing_stats_fields_typed_and_defaults() -> None:
    """03-00: IndexingStats is a typed BaseModel with extra='forbid'."""
    import pydantic

    from ara.state import IndexingStats

    s = IndexingStats()
    assert s.papers_chunked == 0
    assert s.total_chunks == 0
    assert s.unique_chunks == 0
    assert s.duplicate_chunks_dropped == 0
    assert s.embedding_duration_ms == 0
    assert s.index_build_duration_ms == 0
    assert s.index_bytes == 0
    with pytest.raises(pydantic.ValidationError):
        IndexingStats(unexpected_field=1)  # type: ignore[call-arg]


def test_analysis_stats_has_failure_tuple_list() -> None:
    """03-00: AnalysisStats.analysis_failures is a list of (paper_id, reason) tuples."""
    from ara.state import AnalysisStats

    s = AnalysisStats(analysis_failures=[("P1", "ValidationError: bad json")])
    assert s.analysis_failures == [("P1", "ValidationError: bad json")]
    assert s.per_paper_calls == 0
    assert s.cross_paper_calls == 0
    assert s.verifier_calls == 0
    assert s.unsupported_claims == 0
    assert s.duration_ms == 0


def test_report_stats_tracks_unresolved_citations() -> None:
    """03-00: ReportStats.unresolved_citations MUST be 0 at phase gate (RPT-07)."""
    from ara.state import ReportStats

    s = ReportStats(unresolved_citations=0)
    assert s.unresolved_citations == 0
    assert s.latex_bytes == 0
    assert s.pdf_bytes == 0
    assert s.bibtex_keys == 0
    assert s.bibtex_collision_fixes == 0
    assert s.latexmk_duration_ms == 0


def test_paper_has_analysis_status_and_bibtex_key() -> None:
    """03-00: Paper has new analysis_status (AnalysisAgent-written) and bibtex_key
    (ReportAgent-written) fields — both Optional[str] defaulting to None."""
    from ara.state import Paper

    p = Paper(paper_id="P1", title="Attention Is All You Need")
    assert p.analysis_status is None
    assert p.bibtex_key is None
    p.analysis_status = "succeeded"
    p.bibtex_key = "vaswani_2017_attention"
    assert p.analysis_status == "succeeded"
    assert p.bibtex_key == "vaswani_2017_attention"


def test_field_owners_phase3_extensions() -> None:
    """03-00: FIELD_OWNERS extended for indexing / analysis / report.

    - indexing: adds 'indexing_stats'
    - analysis: adds 'analysis_stats' + co-owns 'papers' (nested analysis_status)
    - report: adds 'report_stats' + co-owns 'papers' (nested bibtex_key)
    """
    from ara.state import FIELD_OWNERS

    assert "indexing_stats" in FIELD_OWNERS["indexing"]
    assert "analysis_stats" in FIELD_OWNERS["analysis"]
    assert "papers" in FIELD_OWNERS["analysis"]
    assert "report_stats" in FIELD_OWNERS["report"]
    assert "papers" in FIELD_OWNERS["report"]


def test_pipeline_state_roundtrip_with_phase3_stats() -> None:
    """03-00: PipelineState with the new typed stats fields round-trips through
    model_dump_json / model_validate_json losslessly."""
    from ara.state import AnalysisStats, IndexingStats, PipelineState, ReportStats

    s = PipelineState(
        run_id="rt-test",
        question="q?",
        config_snapshot={"model": "gemini/gemini-2.5-flash"},
        indexing_stats=IndexingStats(papers_chunked=2, total_chunks=17),
        analysis_stats=AnalysisStats(per_paper_calls=6),
        report_stats=ReportStats(unresolved_citations=0),
    )
    dumped = s.model_dump_json()
    loaded = PipelineState.model_validate_json(dumped)
    assert loaded.indexing_stats.papers_chunked == 2
    assert loaded.indexing_stats.total_chunks == 17
    assert loaded.analysis_stats.per_paper_calls == 6
    assert loaded.report_stats.unresolved_citations == 0
