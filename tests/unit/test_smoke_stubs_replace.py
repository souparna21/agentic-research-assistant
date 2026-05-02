"""Assert that ``build_stub_stages`` returns REAL agents for all six stages
(plan 03-09 close): query / retrieval / extraction (Phase 2) plus indexing /
analysis / report (Phase 3).

Legacy ``build_pure_stub_stages`` is also covered to document the stub-only
fallback path used by Phase-1-style orchestrator unit tests that want
purely deterministic, zero-LLM, zero-disk-IO stage behavior.
"""

from __future__ import annotations

from ara.agents.analysis import AnalysisAgent
from ara.agents.extraction import ExtractionAgent
from ara.agents.indexing import IndexingAgent
from ara.agents.query import QueryAgent
from ara.agents.report import ReportAgent
from ara.agents.retrieval import RetrievalAgent
from ara.agents.stubs import build_pure_stub_stages, build_stub_stages


def test_build_stub_stages_wires_all_six_real_agents() -> None:
    """Plan 03-09: all six stages are real agents; no Stub* classes in the chain."""
    stages = build_stub_stages()
    by_name = {s.name: s for s in stages}

    assert isinstance(by_name["query"], QueryAgent)
    assert isinstance(by_name["retrieval"], RetrievalAgent)
    assert isinstance(by_name["extraction"], ExtractionAgent)
    assert isinstance(by_name["indexing"], IndexingAgent)
    assert isinstance(by_name["analysis"], AnalysisAgent)
    assert isinstance(by_name["report"], ReportAgent)

    for name, stage in by_name.items():
        assert not stage.__class__.__name__.startswith("Stub"), (
            f"stage {name!r} is still a stub: {stage.__class__.__name__}"
        )


def test_build_pure_stub_stages_is_all_stubs() -> None:
    """Legacy all-stub factory remains available for tests that want deterministic stubs."""
    stages = build_pure_stub_stages()
    for s in stages:
        assert s.__class__.__name__.startswith("Stub"), s
