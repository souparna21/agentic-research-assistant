"""Typer CLI tests (FND-07: subcommands, fast help, lazy imports).

These tests guard three contracts:
1. All four subcommands (run / stage / show / clean) are registered and reachable
   via `ara --help`.
2. `cli.py` has no heavy module-top imports (torch, sentence_transformers, faiss,
   pymupdf, litellm, ara.orchestrator). AST-scanned — instant red flag if a lazy
   import accidentally gets hoisted.
3. Subcommand behaviors: run creates/resumes run_ids, stage requires --run-id,
   clean requires confirmation unless --yes, show reads state.json.
"""

from __future__ import annotations

import ast
import subprocess
import sys
import time
import types
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner


def _install_orchestrator_stub(monkeypatch, fake_cls):
    """Install a stub `ara.orchestrator` module so `ara.cli.run()`'s lazy import resolves.

    Plan 05 will create the real `ara.orchestrator`; this stub lets plan 04's tests
    patch `Orchestrator` without requiring plan 05 to land first.
    """
    stub = types.ModuleType("ara.orchestrator")
    stub.Orchestrator = fake_cls  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ara.orchestrator", stub)




def test_subcommands_registered():
    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    for sub in ("run", "stage", "show", "clean"):
        assert sub in result.stdout, f"subcommand {sub!r} missing from --help output"


def test_cli_has_no_heavy_top_level_imports():
    """Pitfall: ara --help slow because torch/faiss/sentence_transformers import at top."""
    src = Path("src/ara/cli.py").read_text()
    tree = ast.parse(src)
    forbidden = {"torch", "sentence_transformers", "faiss", "pymupdf", "pymupdf4llm"}
    forbidden_module_top = forbidden | {"litellm", "ara.orchestrator"}
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                assert alias.name not in forbidden_module_top, (
                    f"{alias.name} imported at module top — MUST be lazy-loaded inside a function"
                )
        elif isinstance(node, ast.ImportFrom) and node.module:
            base = node.module.split(".")[0]
            assert base not in forbidden, (
                f"from {node.module} import ... at module top — MUST be lazy-loaded"
            )
            assert node.module != "ara.orchestrator", (
                "from ara.orchestrator ... at module top — MUST be lazy-loaded inside run()/stage()"
            )




@pytest.mark.smoke
def test_help_is_fast():
    """ara --help < 1.5s wall-clock on warm run (uv wrapper adds ~200-400ms overhead)."""
    subprocess.run(
        ["uv", "run", "--no-sync", "ara", "--help"],
        capture_output=True,
        timeout=10,
    )
    t0 = time.perf_counter()
    result = subprocess.run(
        ["uv", "run", "--no-sync", "ara", "--help"],
        capture_output=True,
        timeout=5,
    )
    elapsed = time.perf_counter() - t0
    assert result.returncode == 0
    assert elapsed < 1.5, f"ara --help took {elapsed:.3f}s (budget: 1.5s wall-clock incl. uv overhead)"




def test_run_creates_run_id(tmp_path: Path, monkeypatch):
    """ara run <q> builds an Orchestrator with a freshly-minted run_id."""
    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    class FakeOrchestrator:
        def __init__(self, settings, run_id, git_sha=None, question=None, stages=None):
            captured["run_id"] = run_id
            captured["question"] = question
            captured["stages"] = stages

        def run(self):
            captured["ran"] = True

    _install_orchestrator_stub(monkeypatch, FakeOrchestrator)

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["run", "What is RAG?"])
    assert result.exit_code == 0, result.stdout

    assert captured.get("ran") is True
    import re

    assert re.match(r"^\d{8}-\d{6}-[a-z0-9\-]+$", captured["run_id"]), captured["run_id"]


def test_run_with_existing_run_id_resumes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    class FakeOrchestrator:
        def __init__(self, settings, run_id, git_sha=None, question=None, stages=None):
            captured["run_id"] = run_id

        def run(self):
            pass

    _install_orchestrator_stub(monkeypatch, FakeOrchestrator)

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app, ["run", "--run-id", "20260417-000000-existing", "q"]
    )
    assert result.exit_code == 0

    assert captured["run_id"] == "20260417-000000-existing"




def test_stage_requires_run_id():
    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["stage", "query"])
    assert result.exit_code != 0
    combined = (result.stdout + (result.output or "")).lower()
    assert "run-id" in combined or "run_id" in combined




def test_clean_all_without_yes_prompts(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20260417-000000-x").mkdir()

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["clean", "--all"], input="n\n")
    assert result.exit_code == 0
    assert (runs / "20260417-000000-x").exists(), "Should NOT have deleted without confirmation"


def test_clean_all_with_yes_deletes(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "20260417-000000-x").mkdir()

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["clean", "--all", "--yes"])
    assert result.exit_code == 0
    assert not (runs / "20260417-000000-x").exists()




def test_show_reads_state_json(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    from ara.persistence import atomic_write_json
    from ara.state import PipelineState

    state = PipelineState(
        run_id="20260417-010203-show", question="hello", config_snapshot={}
    )
    atomic_write_json(state, tmp_path / "runs" / state.run_id / "state.json")

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["show", state.run_id])
    assert result.exit_code == 0
    assert "hello" in result.stdout
    assert state.run_id in result.stdout




def test_retrieval_filters_passthrough(monkeypatch, tmp_path: Path):
    """QRY-04: --year-since / --min-citations flags propagate into Settings.

    Exercises the CLI's binding of Typer options to Settings — the actual
    consumption by RetrievalAgent is tested in plan 02-04's unit tests.
    """
    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    class FakeOrchestrator:
        def __init__(self, settings, run_id, *, git_sha, question=None, stages=None):
            captured["settings"] = settings
            captured["question"] = question

        def run(self):
            captured["ran"] = True

        def run_single_stage(self, name):
            pass

    _install_orchestrator_stub(monkeypatch, FakeOrchestrator)

    monkeypatch.delenv("ARA_RETRIEVAL_YEAR_SINCE", raising=False)
    monkeypatch.delenv("ARA_RETRIEVAL_MIN_CITATIONS", raising=False)

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app,
        ["run", "--year-since", "2020", "--min-citations", "10", "--run-id", "test-qry4", "What is RAG?"],
    )
    assert result.exit_code == 0, result.stdout

    s = captured["settings"]
    assert s.retrieval_year_since == 2020
    assert s.retrieval_min_citations == 10


def test_retrieval_filters_default_none(monkeypatch, tmp_path: Path):
    """QRY-04: without flags or env vars, filters default to None (off)."""
    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    class FakeOrchestrator:
        def __init__(self, settings, run_id, *, git_sha, question=None, stages=None):
            captured["settings"] = settings

        def run(self):
            pass

        def run_single_stage(self, name):
            pass

    _install_orchestrator_stub(monkeypatch, FakeOrchestrator)

    monkeypatch.delenv("ARA_RETRIEVAL_YEAR_SINCE", raising=False)
    monkeypatch.delenv("ARA_RETRIEVAL_MIN_CITATIONS", raising=False)

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--run-id", "test-qry4-defaults", "q"])
    assert result.exit_code == 0, result.stdout
    assert captured["settings"].retrieval_year_since is None
    assert captured["settings"].retrieval_min_citations is None


def test_retrieval_filters_env_var_honored(monkeypatch, tmp_path: Path):
    """QRY-04: ARA_RETRIEVAL_YEAR_SINCE / ARA_RETRIEVAL_MIN_CITATIONS env vars honored when CLI flags absent."""
    monkeypatch.chdir(tmp_path)
    captured: dict = {}

    class FakeOrchestrator:
        def __init__(self, settings, run_id, *, git_sha, question=None, stages=None):
            captured["settings"] = settings

        def run(self):
            pass

        def run_single_stage(self, name):
            pass

    _install_orchestrator_stub(monkeypatch, FakeOrchestrator)

    monkeypatch.setenv("ARA_RETRIEVAL_YEAR_SINCE", "2019")
    monkeypatch.setenv("ARA_RETRIEVAL_MIN_CITATIONS", "5")

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(app, ["run", "--run-id", "test-qry4-env", "q"])
    assert result.exit_code == 0, result.stdout
    assert captured["settings"].retrieval_year_since == 2019
    assert captured["settings"].retrieval_min_citations == 5




def test_cached_only_wired(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """`--cached-only` invokes extract_demo_cache BEFORE Orchestrator construction.

    Asserts the full wiring contract:
      1. `extract_demo_cache` is called exactly once when `--cached-only` is set.
      2. The returned run_id becomes the Orchestrator's `rid` argument
         (not a freshly-minted timestamped run_id).
      3. `--cached-only` + `--run-id X` -> cached wins (WARN on stderr).

    Uses fakes for both extract_demo_cache and Orchestrator so no real
    tarball / pipeline is needed. `demo_cache` is imported lazily inside
    `cli.run`, so patching the module attribute `ara.demo_cache.extract_demo_cache`
    is sufficient — the lazy `from ... import ...` inside run() resolves
    through the patched module.
    """
    monkeypatch.chdir(tmp_path)
    call_log: dict[str, Any] = {"extract_calls": 0, "orch_rid": None}

    def _fake_extract(target_runs_dir: Path, tarball_path: Path | None = None) -> str:
        call_log["extract_calls"] += 1
        call_log["target_runs_dir"] = target_runs_dir
        rid = "demo-v1"
        (target_runs_dir / rid).mkdir(parents=True, exist_ok=True)
        (target_runs_dir / rid / "state.json").write_text(
            '{"run_id": "demo-v1"}', encoding="utf-8"
        )
        return rid

    class _FakeOrch:
        def __init__(
            self,
            settings: Any,
            rid: str,
            *,
            git_sha: str,
            question: str | None = None,
            stages: Any = None,
        ) -> None:
            call_log["orch_rid"] = rid
            call_log["orch_question"] = question

        def run(self) -> None:
            call_log["orch_ran"] = True

    monkeypatch.setattr("ara.demo_cache.extract_demo_cache", _fake_extract)
    _install_orchestrator_stub(monkeypatch, _FakeOrch)

    from ara.cli import app

    runner = CliRunner()
    result = runner.invoke(
        app, ["--cached-only", "run", "demo question"]
    )

    assert result.exit_code == 0, (
        f"CLI failed: stdout={result.stdout!r} exc={result.exception!r}"
    )
    assert call_log["extract_calls"] == 1, "extract_demo_cache not called"
    assert call_log["orch_rid"] == "demo-v1", (
        f"Orchestrator rid should be demo-v1 (from extract_demo_cache), "
        f"got {call_log['orch_rid']!r}"
    )
    assert call_log.get("orch_ran") is True


def test_cached_only_ignores_explicit_run_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """When both --cached-only and --run-id given, cached-only wins + WARN emitted."""
    monkeypatch.chdir(tmp_path)
    call_log: dict[str, Any] = {"orch_rid": None}

    def _fake_extract(target_runs_dir: Path, tarball_path: Path | None = None) -> str:
        (target_runs_dir / "demo-v1").mkdir(parents=True, exist_ok=True)
        return "demo-v1"

    class _FakeOrch:
        def __init__(
            self,
            settings: Any,
            rid: str,
            *,
            git_sha: str,
            question: str | None = None,
            stages: Any = None,
        ) -> None:
            call_log["orch_rid"] = rid

        def run(self) -> None:
            pass

    monkeypatch.setattr("ara.demo_cache.extract_demo_cache", _fake_extract)
    _install_orchestrator_stub(monkeypatch, _FakeOrch)

    from ara.cli import app

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(
        app,
        ["--cached-only", "run", "--run-id", "some-user-rid", "demo question"],
    )

    assert result.exit_code == 0, (
        f"CLI failed: stdout={result.stdout!r} stderr={result.stderr!r} "
        f"exc={result.exception!r}"
    )
    assert call_log["orch_rid"] == "demo-v1"
    combined = (result.stderr or "") + (result.stdout or "")
    assert "some-user-rid" in combined, (
        f"expected WARN about ignored --run-id; got stdout={result.stdout!r} "
        f"stderr={result.stderr!r}"
    )
