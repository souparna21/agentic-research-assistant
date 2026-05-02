"""End-to-end stub-driven smoke test (FND-12).

All six stages are stubbed; zero network; must complete in <30s in CI.
pytest-socket (configured via pyproject.toml addopts `--disable-socket`)
blocks any socket use — an accidental network call fails loudly with
SocketBlockedError rather than silently delaying the run.

This file is the Phase 1 capstone: if these tests pass green, the twelve
FND-* requirements are integration-proved. Phase 2 and Phase 3 agents
replace stubs one by one; the same test file continues to gate every PR
through submission day.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import pytest

from ara.agents.stubs import build_stub_stages
from ara.config import Settings
from ara.orchestrator import STAGE_ORDER, Orchestrator
from ara.runids import make_run_id
from ara.state import FIELD_OWNERS, PipelineState

pytestmark = pytest.mark.smoke




@pytest.fixture(autouse=True)
def _hf_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force HuggingFace to serve the cached MiniLM snapshot (no network pings).

    build_stub_stages wires a real MiniLMEmbedder into IndexingAgent (plan 03-04)
    and AnalysisAgent (plan 03-06 — instantiates its own inside run()). Without
    this fixture, sentence-transformers would attempt a model-card fetch on
    first load and pytest-socket raises SocketBlockedError. Matches the
    per-file pattern used by test_embed / test_vector_store / test_indexing_agent
    / test_analysis_agent; if a sixth consumer appears, promote to conftest.py.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")




GOLDEN_RUN_ID = "GOLDEN-SMOKE-STATE"
GOLDEN_GIT_SHA = "GOLDEN"
GOLDEN_QUESTION = "What is retrieval-augmented generation?"


def _normalize_state_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Replace per-run values with stable placeholders for golden-state compare.

    The following fields vary per run and must be normalized before equality:
      - run_id (timestamped)
      - config_snapshot.git_sha (per-repo-HEAD)
      - index_meta.built_at (currently a placeholder but future-proofs)
      - stage_reports[*].duration_ms (wall-clock noise)
      - runs_dir (absolute tmp_path in tests; literal "runs" in golden)
      - any run_id-embedded paths (index_path / summaries_dir / comparison_path /
        gaps_path / report_path) are rewritten to use GOLDEN-SMOKE-STATE
    """
    d = dict(d)
    run_id = d["run_id"]

    def _swap_rid(s: str | None) -> str | None:
        if s is None:
            return None
        return s.replace(run_id, GOLDEN_RUN_ID) if run_id in s else s

    d["run_id"] = GOLDEN_RUN_ID
    d["runs_dir"] = "runs"

    if "config_snapshot" in d:
        cs = dict(d["config_snapshot"])
        cs["git_sha"] = GOLDEN_GIT_SHA
        d["config_snapshot"] = cs

    if d.get("index_meta"):
        im = dict(d["index_meta"])
        if "built_at" in im:
            im["built_at"] = "GOLDEN-TIMESTAMP"
        d["index_meta"] = im

    for key in (
        "index_path",
        "summaries_dir",
        "comparison_path",
        "gaps_path",
        "report_path",
    ):
        if d.get(key):
            d[key] = _swap_rid(d[key])

    reports = []
    for rep in d.get("stage_reports", []):
        rep = dict(rep)
        rep["duration_ms"] = 0.0
        reports.append(rep)
    d["stage_reports"] = reports

    if isinstance(d.get("retrieval_stats"), dict):
        rs = dict(d["retrieval_stats"])
        rs["duration_ms"] = 0
        d["retrieval_stats"] = rs

    if isinstance(d.get("extraction_stats"), dict):
        es = dict(d["extraction_stats"])
        es["duration_ms"] = 0
        d["extraction_stats"] = es

    if isinstance(d.get("indexing_stats"), dict):
        is_ = dict(d["indexing_stats"])
        is_["embedding_duration_ms"] = 0
        is_["index_build_duration_ms"] = 0
        is_["index_bytes"] = 0
        d["indexing_stats"] = is_

    if isinstance(d.get("analysis_stats"), dict):
        as_ = dict(d["analysis_stats"])
        as_["duration_ms"] = 0
        d["analysis_stats"] = as_

    if isinstance(d.get("report_stats"), dict):
        rps = dict(d["report_stats"])
        rps["latexmk_duration_ms"] = 0
        rps["latex_bytes"] = 0
        rps["pdf_bytes"] = 0
        d["report_stats"] = rps

    if d.get("papers"):
        papers = []
        for p in d["papers"]:
            p = dict(p)
            if p.get("pdf_path"):
                swapped = _swap_rid(p["pdf_path"])
                assert swapped is not None
                marker = f"/{GOLDEN_RUN_ID}/"
                idx = swapped.find(marker)
                if idx >= 0:
                    p["pdf_path"] = f"<RUNS_DIR>/{GOLDEN_RUN_ID}{swapped[idx + len(marker) - 1:]}"
                else:
                    p["pdf_path"] = swapped
            if p.get("pdf_url"):
                url = p["pdf_url"]
                marker = "tests/fixtures/papers/"
                idx = url.find(marker)
                if idx >= 0:
                    p["pdf_url"] = f"<FIXTURES>/{url[idx:]}"
            if p.get("extracted_md_path"):
                p["extracted_md_path"] = _swap_rid(p["extracted_md_path"])
            papers.append(p)
        d["papers"] = papers

    return d


def _build_orch(tmp_path: Path, run_id: str, git_sha: str, question: str) -> Orchestrator:
    settings = Settings(
        runs_dir=str(tmp_path / "runs"),
        llm_model="gemini/gemini-2.5-flash-lite",
    )
    return Orchestrator(
        settings,
        run_id,
        git_sha=git_sha,
        question=question,
        stages=build_stub_stages(),
    )




def test_smoke_under_30s(tmp_path: Path) -> None:
    """The full stub pipeline must complete in <30 seconds."""
    run_id = make_run_id("What is RAG?")
    orch = _build_orch(tmp_path, run_id, git_sha="smokegit", question="What is RAG?")

    t0 = time.perf_counter()
    orch.run()
    elapsed = time.perf_counter() - t0

    assert elapsed < 30.0, f"Smoke took {elapsed:.2f}s (budget: 30s)"

    state_json = tmp_path / "runs" / run_id / "state.json"
    assert state_json.exists(), f"state.json missing at {state_json}"
    PipelineState.model_validate_json(state_json.read_text(encoding="utf-8"))


def test_smoke_produces_artifacts(tmp_path: Path) -> None:
    """state.json + run.log exist; every log line carries run_id + git_sha bindings."""
    run_id = make_run_id("artifacts-test")
    _build_orch(tmp_path, run_id, git_sha="artgit", question="artifacts-test").run()

    run_dir = tmp_path / "runs" / run_id
    assert (run_dir / "state.json").exists(), "state.json missing"

    log_path = run_dir / "run.log"
    assert log_path.exists(), "run.log missing"

    lines = [
        json.loads(ln) for ln in log_path.read_text(encoding="utf-8").splitlines() if ln.strip()
    ]
    assert lines, "run.log is empty"

    for rec in lines:
        assert rec.get("run_id") == run_id, f"Log line missing run_id binding: {rec}"
        assert rec.get("git_sha") == "artgit", f"Log line missing git_sha binding: {rec}"

    for stage_name in STAGE_ORDER:
        assert any(
            r.get("event") == "stage_start" and r.get("stage") == stage_name for r in lines
        ), f"no stage_start for {stage_name}"
        assert any(
            r.get("event") == "stage_end" and r.get("stage") == stage_name for r in lines
        ), f"no stage_end for {stage_name}"


def test_smoke_is_idempotent(tmp_path: Path) -> None:
    """Second run with same run_id: zero stub .run() invocations — all stages skip."""
    settings = Settings(runs_dir=str(tmp_path / "runs"))
    run_id = make_run_id("idempotent-test")

    invocation_counter = {"n": 0}

    def _wrap(stage: Any) -> Any:
        orig = stage.run

        def wrapped(state: Any) -> Any:
            invocation_counter["n"] += 1
            return orig(state)

        stage.run = wrapped  # noqa: E501  (monkey-patch the stub's run method)
        return stage

    stages1 = [_wrap(s) for s in build_stub_stages()]
    Orchestrator(
        settings,
        run_id,
        git_sha="idemp",
        question="idempotent-test",
        stages=stages1,
    ).run()
    assert (
        invocation_counter["n"] == 6
    ), f"first run should invoke 6 stubs, got {invocation_counter['n']}"

    invocation_counter["n"] = 0
    stages2 = [_wrap(s) for s in build_stub_stages()]
    Orchestrator(
        settings,
        run_id,
        git_sha="idemp",
        question="idempotent-test",
        stages=stages2,
    ).run()
    assert invocation_counter["n"] == 0, (
        f"second run invoked {invocation_counter['n']} stubs; "
        "expected 0 (all should skip due to existing artifact markers)"
    )


def test_smoke_final_state_matches_golden(tmp_path: Path) -> None:
    """Final state must match tests/fixtures/golden_state.json (modulo per-run values)."""
    run_id = make_run_id("golden-test")
    _build_orch(tmp_path, run_id, git_sha=GOLDEN_GIT_SHA, question=GOLDEN_QUESTION).run()

    produced_path = tmp_path / "runs" / run_id / "state.json"
    produced = json.loads(produced_path.read_text(encoding="utf-8"))
    produced_norm = _normalize_state_dict(produced)

    golden_path = Path(__file__).resolve().parents[1] / "fixtures" / "golden_state.json"
    golden = json.loads(golden_path.read_text(encoding="utf-8"))
    golden_norm = _normalize_state_dict(golden)

    assert produced_norm == golden_norm, (
        "Smoke final state diverged from golden_state.json.\n"
        f"Produced (normalized):\n"
        f"{json.dumps(produced_norm, indent=2, sort_keys=True)[:3000]}\n"
        f"Golden (normalized):\n"
        f"{json.dumps(golden_norm, indent=2, sort_keys=True)[:3000]}"
    )


def test_smoke_no_field_ownership_violation(tmp_path: Path) -> None:
    """Pipeline runs without raising FieldOwnershipViolation — invariant holds."""
    run_id = make_run_id("ownership-test")
    _build_orch(tmp_path, run_id, git_sha="own", question="ownership-test").run()

    assert set(FIELD_OWNERS.keys()) == set(STAGE_ORDER)




def test_smoke_produces_compilable_report_pdf(tmp_path: Path) -> None:
    """Phase 3 closure — full pipeline produces a parseable report.tex + references.bib.

    No-latex mode (always green): asserts report.tex is structurally valid LaTeX
    via substring checks for \\documentclass, \\begin{document}, \\end{document},
    \\bibliography{references}; asserts references.bib exists.

    Latex mode (auto-gated by latexmk_available()): additionally asserts
    report.pdf exists + count_unresolved_citations(pdf) == 0.  The latex path
    runs only when latexmk is on PATH — no pytest.skip needed because
    ReportAgent handles the latexmk-missing case internally (writes .tex +
    .bib, skips compile).
    """
    from ara.services.latex import count_unresolved_citations, latexmk_available

    run_id = make_run_id("compilable-report")
    orch = _build_orch(tmp_path, run_id, git_sha="compgit", question=GOLDEN_QUESTION)
    orch.run()

    run_dir = tmp_path / "runs" / run_id
    tex = run_dir / "report.tex"
    bib = run_dir / "references.bib"

    assert tex.exists(), f"report.tex missing at {tex}"
    assert bib.exists(), f"references.bib missing at {bib}"

    body = tex.read_text(encoding="utf-8")
    for marker in (
        r"\documentclass",
        r"\begin{document}",
        r"\end{document}",
        r"\bibliography{references}",
    ):
        assert marker in body, f"report.tex missing structural marker: {marker!r}"

    if latexmk_available():
        pdf = run_dir / "report.pdf"
        assert pdf.exists(), f"report.pdf missing after latexmk compile"
        assert pdf.stat().st_size > 0, "report.pdf is empty"
        assert (
            count_unresolved_citations(pdf) == 0
        ), "report.pdf has [?] unresolved citations"


@pytest.mark.latex
def test_smoke_report_pdf_compiles_cleanly(tmp_path: Path) -> None:
    """Latex-marked companion of test_smoke_produces_compilable_report_pdf.

    Runs only when the ``latex`` marker is selected (``-m latex``) AND
    latexmk is on PATH.  Provides an explicit latex-gated node in the
    test catalog so ``pytest -m latex`` picks it up for the Phase-3
    manual-verification gate per 03-VALIDATION.md.
    """
    from ara.services.latex import count_unresolved_citations, latexmk_available

    if not latexmk_available():
        pytest.skip("latexmk not installed")

    run_id = make_run_id("compilable-report-latex")
    _build_orch(tmp_path, run_id, git_sha="lxgit", question=GOLDEN_QUESTION).run()

    run_dir = tmp_path / "runs" / run_id
    pdf = run_dir / "report.pdf"
    assert pdf.exists(), f"report.pdf missing: {pdf}"
    assert pdf.stat().st_size > 0, "report.pdf is empty"
    assert count_unresolved_citations(pdf) == 0


def test_smoke_is_idempotent_across_all_stages(tmp_path: Path) -> None:
    """Second full-pipeline run with same run_id: all 6 stages skip (0 run invocations).

    Parallel to test_smoke_is_idempotent but exercises the full Phase-3-real
    agent set — IndexingAgent's faiss.index, AnalysisAgent's
    summaries/.analysis.marker, and ReportAgent's report.tex all act as
    delete-to-invalidate idempotency markers.  If any Phase-3 agent's
    output_artifact_path() regresses, this test catches the second-run
    re-execution.
    """
    settings = Settings(runs_dir=str(tmp_path / "runs"))
    run_id = make_run_id("full-idempotency")

    invocation_counter = {"n": 0}

    def _wrap(stage: Any) -> Any:
        orig = stage.run

        def wrapped(state: Any) -> Any:
            invocation_counter["n"] += 1
            return orig(state)

        stage.run = wrapped
        return stage

    stages1 = [_wrap(s) for s in build_stub_stages()]
    Orchestrator(
        settings,
        run_id,
        git_sha="idemp-all",
        question=GOLDEN_QUESTION,
        stages=stages1,
    ).run()
    assert (
        invocation_counter["n"] == 6
    ), f"first run should invoke 6 stages, got {invocation_counter['n']}"

    invocation_counter["n"] = 0
    stages2 = [_wrap(s) for s in build_stub_stages()]
    Orchestrator(
        settings,
        run_id,
        git_sha="idemp-all",
        question=GOLDEN_QUESTION,
        stages=stages2,
    ).run()
    assert invocation_counter["n"] == 0, (
        f"second run invoked {invocation_counter['n']} stages; "
        "expected 0 (delete-to-invalidate idempotency)"
    )




def test_cached_only_mode_works_offline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """`ara run --cached-only` extracts demo-cache.tar.gz and produces report.tex offline.

    Zero network: pytest-socket is globally active (--disable-socket in pyproject.toml);
    any accidental socket use raises SocketBlockedError and fails this test loudly.

    The autouse ``_hf_offline`` fixture (already applied per-file) covers
    ``HF_HUB_OFFLINE=1`` + ``TRANSFORMERS_OFFLINE=1``.

    Flow:
      1. Chdir to repo root so ``demo-cache.tar.gz`` (relative default) resolves.
      2. Redirect ``runs_dir`` to ``tmp_path`` via ``ARA_RUNS_DIR`` so no real
         runs/ is polluted.
      3. Invoke ``ara --cached-only run "demo question"``; assert exit_code == 0.
      4. Assert the extracted run dir is ``runs/demo-v1/`` containing
         ``state.json`` and ``report.tex``.
      5. If ``report.pdf`` happens to be in the tarball (latexmk was on PATH
         when the tarball was generated), verify the %PDF- magic bytes.
    """
    from typer.testing import CliRunner

    from ara.cli import app

    repo_root = Path(__file__).resolve().parents[2]
    assert (repo_root / "demo-cache.tar.gz").exists(), (
        "demo-cache.tar.gz must be committed at repo root "
        "(run scripts/create_demo_cache.py)"
    )
    monkeypatch.chdir(repo_root)

    runs_dir = tmp_path / "runs"
    monkeypatch.setenv("ARA_RUNS_DIR", str(runs_dir))

    runner = CliRunner()
    result = runner.invoke(app, ["--cached-only", "run", "demo question"])

    assert result.exit_code == 0, (
        f"CLI failed (exit {result.exit_code}):\nSTDOUT: {result.stdout}\n"
        f"EXCEPTION: {result.exception!r}"
    )

    run_dirs = [p for p in runs_dir.iterdir() if p.is_dir()]
    assert len(run_dirs) == 1, f"Expected 1 run dir, got {len(run_dirs)}: {run_dirs}"
    run_dir = run_dirs[0]
    assert run_dir.name == "demo-v1", (
        f"Expected run dir 'demo-v1', got {run_dir.name!r}"
    )

    assert (run_dir / "state.json").exists(), "state.json missing in extracted run"
    assert (run_dir / "report.tex").exists(), "report.tex missing in extracted run"

    pdf = run_dir / "report.pdf"
    if pdf.exists() and pdf.stat().st_size > 0:
        with pdf.open("rb") as f:
            magic = f.read(5)
        assert magic == b"%PDF-", f"report.pdf has bad magic bytes: {magic!r}"
