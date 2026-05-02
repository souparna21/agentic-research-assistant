"""Orchestrator tests (FND-08 stage order + skip + persist; FND-02 invariant).

Covers:
    - Fixed stage order (query → retrieval → extraction → indexing → analysis → report)
    - Skip-on-artifact-exists (delete-to-invalidate idempotency)
    - Atomic persist of PipelineState after each stage
    - Single-writer invariant enforcement (pitfall P12 defense)
    - stage_reports append-only semantics
    - run_single_stage for `ara stage` CLI subcommand
    - Resume from existing state.json
    - _run_stage helper exposed at module level (plan 01's test re-uses it)
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ara.config import Settings
from ara.errors import FieldOwnershipViolation
from ara.state import Paper, PipelineState, StageReport


class DummyStage:
    """In-memory stub that records invocation + optionally mutates owned fields.

    The orchestrator only needs three things from a stage: `.name`,
    `.output_artifact_path(state)`, and `.run(state) -> PipelineState`.
    """

    def __init__(self, name: str, artifact_filename: str, mutate=None):
        self.name = name
        self.artifact_filename = artifact_filename
        self.mutate = mutate
        self.ran = False

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / self.artifact_filename

    def run(self, state: PipelineState) -> PipelineState:
        self.ran = True
        if self.mutate:
            self.mutate(state)
        self.output_artifact_path(state).parent.mkdir(parents=True, exist_ok=True)
        self.output_artifact_path(state).write_text("ok")
        return state




def _make_orch(tmp_path, stages, run_id="20260417-000000-test", question="hi"):
    from ara.orchestrator import Orchestrator

    s = Settings(runs_dir=str(tmp_path / "runs"))
    return Orchestrator(s, run_id, git_sha="abc1234", question=question, stages=stages)


def _filler_stages(skip_name: str) -> list[DummyStage]:
    """Build DummyStages for all 6 stage names EXCEPT the one given."""
    all_names = ("query", "retrieval", "extraction", "indexing", "analysis", "report")
    return [DummyStage(n, f"{n}.marker") for n in all_names if n != skip_name]




def test_stage_order(tmp_path):
    """Orchestrator invokes stages in the canonical 6-stage order."""
    order = []

    def _record(name):
        def fn(state):
            order.append(name)

        return fn

    stages = [
        DummyStage("query", "query.marker", _record("query")),
        DummyStage("retrieval", "retrieval.marker", _record("retrieval")),
        DummyStage("extraction", "extraction.marker", _record("extraction")),
        DummyStage("indexing", "indexing.marker", _record("indexing")),
        DummyStage("analysis", "analysis.marker", _record("analysis")),
        DummyStage("report", "report.marker", _record("report")),
    ]
    orch = _make_orch(tmp_path, stages)
    orch.run()
    assert order == ["query", "retrieval", "extraction", "indexing", "analysis", "report"]


def test_stage_skips_when_artifact_exists(tmp_path):
    """If stage.output_artifact_path exists, the orchestrator does NOT call .run()."""
    stage = DummyStage("query", "query.marker")
    stages = [stage, *_filler_stages("query")]
    orch = _make_orch(tmp_path, stages)
    marker = tmp_path / "runs" / "20260417-000000-test" / "query.marker"
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text("pre-existing")
    orch.run()
    assert stage.ran is False


def test_persists_state_after_each_stage(tmp_path):
    """After a stage runs, its owned-field mutations are reflected in state.json."""

    def set_terms(state):
        state.search_terms = ["term-a", "term-b"]

    stages = [
        DummyStage("query", "query.marker", set_terms),
        *_filler_stages("query"),
    ]
    orch = _make_orch(tmp_path, stages)
    orch.run()
    state_json = tmp_path / "runs" / "20260417-000000-test" / "state.json"
    data = json.loads(state_json.read_text())
    assert data["search_terms"] == ["term-a", "term-b"]


def test_single_writer_invariant_enforced(tmp_path):
    """A query-stage that writes state.papers (owned by retrieval) → FieldOwnershipViolation."""

    def steal_papers(state):
        state.papers.append(Paper(paper_id="X", title="stolen"))

    bad_query = DummyStage("query", "query.marker", steal_papers)
    stages = [bad_query, *_filler_stages("query")]
    orch = _make_orch(tmp_path, stages)
    with pytest.raises(FieldOwnershipViolation, match="papers"):
        orch.run()


def test_stage_reports_appendable_without_violation(tmp_path):
    """stage_reports is an append-only shared log; any stage may append to it."""

    def append_report(state):
        state.search_terms = ["ok"]
        state.stage_reports.append(StageReport(stage="query", outputs_count=1))

    stages = [
        DummyStage("query", "query.marker", append_report),
        *_filler_stages("query"),
    ]
    orch = _make_orch(tmp_path, stages)
    orch.run()
    assert len(orch.state.stage_reports) == 1
    assert orch.state.stage_reports[0].stage == "query"


def test_run_single_stage_runs_only_that_stage(tmp_path):
    """`orch.run_single_stage('retrieval')` runs ONLY retrieval, not the other five."""
    called: list[str] = []

    stages = [
        DummyStage(name, f"{name}.marker", lambda s, n=name: called.append(n))
        for name in ("query", "retrieval", "extraction", "indexing", "analysis", "report")
    ]
    orch = _make_orch(tmp_path, stages)
    orch.run_single_stage("retrieval")
    assert called == ["retrieval"]


def test_orchestrator_resumes_from_existing_state(tmp_path):
    """When runs/<id>/state.json exists, Orchestrator loads it instead of rebuilding."""
    from ara.orchestrator import Orchestrator
    from ara.persistence import atomic_write_json

    pre = PipelineState(
        run_id="20260417-000000-resume",
        question="previously-saved",
        config_snapshot={"git_sha": "cafebabe"},
        runs_dir=str(tmp_path / "runs"),
    )
    atomic_write_json(pre, tmp_path / "runs" / pre.run_id / "state.json")

    s = Settings(runs_dir=str(tmp_path / "runs"))
    orch = Orchestrator(s, pre.run_id, git_sha="abc1234", stages=[])
    assert orch.state.question == "previously-saved"


def test_orchestrator_creates_new_state_when_no_existing(tmp_path):
    """With no state.json on disk, orchestrator builds a fresh PipelineState from question."""
    from ara.orchestrator import Orchestrator

    s = Settings(runs_dir=str(tmp_path / "runs"))
    orch = Orchestrator(s, "20260417-000000-fresh", git_sha="abc1234", question="fresh-question", stages=[])
    assert orch.state.question == "fresh-question"
    assert orch.state.run_id == "20260417-000000-fresh"
    assert orch.state.config_snapshot["git_sha"] == "abc1234"


def test_invariant_ignores_identity_fields_when_unchanged(tmp_path):
    """Identity fields (run_id, question, etc.) are treated as non-owned by every stage;
    as long as a stage doesn't touch them, it's fine. This is the default path."""

    def benign_mutation(state):
        state.search_terms = ["ok"]

    stages = [
        DummyStage("query", "query.marker", benign_mutation),
        *_filler_stages("query"),
    ]
    orch = _make_orch(tmp_path, stages)
    orch.run()
    assert orch.state.question == "hi"


def test_run_stage_helper_exposed_for_unit_tests():
    """Plan 01's test_field_ownership_violation_detected calls ara.orchestrator._run_stage."""
    from ara.orchestrator import _run_stage

    assert callable(_run_stage)


def test_stage_order_constant_is_canonical():
    """STAGE_ORDER must match the canonical 6-stage sequence."""
    from ara.orchestrator import STAGE_ORDER

    assert STAGE_ORDER == ("query", "retrieval", "extraction", "indexing", "analysis", "report")


def test_run_single_stage_raises_on_unknown_name(tmp_path):
    """Calling run_single_stage with an unknown name gives an actionable error."""
    stages = [DummyStage("query", "query.marker")]
    orch = _make_orch(tmp_path, stages)
    with pytest.raises(ValueError, match="No stage named"):
        orch.run_single_stage("bogus")
