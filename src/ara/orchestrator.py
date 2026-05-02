"""Pipeline orchestrator.

Runs stages in fixed order, enforces the single-writer-per-field invariant
(pitfall P12 defense — the project's highest-probability catastrophic
failure), persists PipelineState atomically after each stage, and skips
stages whose output artifact already exists (delete-to-invalidate
idempotency, FND-08).

Invariant scope:
    Identity fields (run_id, question, config_snapshot, runs_dir) are
    owned by the orchestrator at init and treated as non-owned by every
    stage. `stage_reports` is append-only: any stage may append, but
    shortening or re-ordering prior entries raises FieldOwnershipViolation.
    Every other field has exactly one owner per FIELD_OWNERS.

Structured logging is wired in plan 06 — this module uses structlog
and calls `configure_logging(...)` once per orchestrator construction,
binding `run_id` + `git_sha` via contextvars. The `stage_start` /
`stage_end` / `stage_skipped` events emitted here inherit those bindings
automatically, as do any stdlib log records from LiteLLM / httpx /
urllib3 (routed through the stdlib→structlog ProcessorFormatter bridge).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path

import structlog

from ara.agents.base import Stage
from ara.config import Settings
from ara.errors import FieldOwnershipViolation
from ara.persistence import atomic_write_json, load_state
from ara.state import FIELD_OWNERS, PipelineState

log = structlog.get_logger(__name__)


STAGE_ORDER: tuple[str, ...] = (
    "query",
    "retrieval",
    "extraction",
    "indexing",
    "analysis",
    "report",
)
"""Canonical stage execution order. Never change without updating
FIELD_OWNERS, all stage implementations, the smoke test, and the
three-developer-parallel contract in 01-RESEARCH.md."""



_APPEND_ONLY_FIELDS: frozenset[str] = frozenset({"stage_reports"})


def _non_owned_fields(stage_name: str) -> frozenset[str]:
    """Fields that stage `stage_name` MUST NOT modify.

    Identity fields (run_id, question, config_snapshot, runs_dir) are NOT
    in any stage's FIELD_OWNERS entry, so they land in the non-owned set
    automatically — no stage may write them.
    """
    owned = FIELD_OWNERS.get(stage_name, frozenset())
    all_fields = frozenset(PipelineState.model_fields.keys())
    return all_fields - owned - _APPEND_ONLY_FIELDS


def _run_stage(stage: Stage, state: PipelineState) -> PipelineState:
    """Execute one stage enforcing the single-writer invariant.

    Exposed at module level so plan 01's
    `test_field_ownership_violation_detected` can import and call directly
    via `pytest.importorskip("ara.orchestrator")._run_stage`.

    Raises:
        FieldOwnershipViolation: if the stage wrote to a field it does
            not own, or if it shortened / modified a prior stage_reports
            entry (append-only invariant).
    """
    non_owned = _non_owned_fields(stage.name)

    dump = state.model_dump(mode="python")
    before = {k: deepcopy(dump[k]) for k in non_owned}
    before_reports = deepcopy(
        [r.model_dump(mode="python") for r in state.stage_reports]
    )

    state = stage.run(state)

    after_dump = state.model_dump(mode="python")
    after = {k: after_dump[k] for k in non_owned}
    diff = [k for k in non_owned if before[k] != after[k]]
    if diff:
        raise FieldOwnershipViolation(
            f"Stage {stage.name!r} wrote to non-owned fields: {sorted(diff)}. "
            f"Allowed writes: {sorted(FIELD_OWNERS.get(stage.name, frozenset()))} "
            f"(plus append-only stage_reports)."
        )

    current_reports = [r.model_dump(mode="python") for r in state.stage_reports]
    if len(current_reports) < len(before_reports):
        raise FieldOwnershipViolation(
            f"Stage {stage.name!r} shortened stage_reports "
            f"({len(before_reports)} → {len(current_reports)}); "
            f"stage_reports is append-only."
        )
    for i, prior in enumerate(before_reports):
        if current_reports[i] != prior:
            raise FieldOwnershipViolation(
                f"Stage {stage.name!r} modified stage_reports[{i}]; "
                f"stage_reports is append-only — new entries may be "
                f"appended, but prior entries are immutable."
            )

    return state




class Orchestrator:
    """Runs stages in fixed order with invariant enforcement and idempotency.

    Args:
        settings: Loaded `ara.config.Settings`.
        run_id: `YYYYMMDD-HHMMSS-<slug>` run identifier.
        git_sha: Git short SHA for `config_snapshot` (reproducibility).
        question: User's research question. Used ONLY when creating a
            NEW state (no existing state.json). On resume, the saved
            state's `question` wins and this argument is ignored.
        stages: List of `Stage`-implementing objects in the order they
            must run. When `None`, the orchestrator starts empty and
            `run()` is a no-op — useful for tests that only need
            construction + state loading (resume tests).
    """

    def __init__(
        self,
        settings: Settings,
        run_id: str,
        *,
        git_sha: str,
        question: str | None = None,
        stages: Sequence[Stage] | None = None,
    ) -> None:
        self.settings = settings
        self.run_id = run_id
        self.git_sha = git_sha
        self.stages: list[Stage] = list(stages) if stages is not None else []

        self._runs_dir = Path(settings.runs_dir)
        self._state_path = self._runs_dir / run_id / "state.json"

        if self._state_path.exists():
            self.state = load_state(self._state_path)
        else:
            if question is None:
                raise ValueError(
                    f"No existing state at {self._state_path} and no "
                    f"question provided. Cannot create a fresh run "
                    f"without a question."
                )
            self.state = PipelineState(
                run_id=run_id,
                question=question,
                config_snapshot={
                    "model": settings.llm_model,
                    "temperature": settings.llm_temperature,
                    "git_sha": git_sha,
                    "cached_only": settings.cached_only,
                },
                runs_dir=settings.runs_dir,
            )
            atomic_write_json(self.state, self._state_path)

        from ara.logging import configure_logging

        configure_logging(
            log_path=self._runs_dir / run_id / "run.log",
            log_level=settings.log_level,
            run_id=run_id,
            git_sha=git_sha,
        )


    def run(self) -> None:
        """Execute all configured stages in order."""
        for stage in self.stages:
            self._maybe_run_stage(stage)

    def run_single_stage(self, name: str) -> None:
        """Run only the named stage (for `ara stage` CLI subcommand)."""
        for stage in self.stages:
            if stage.name == name:
                self._maybe_run_stage(stage)
                return
        raise ValueError(
            f"No stage named {name!r} is configured. "
            f"Available: {[s.name for s in self.stages]}"
        )


    def _maybe_run_stage(self, stage: Stage) -> None:
        """Check skip-on-artifact, then run with invariant enforcement."""
        artifact = stage.output_artifact_path(self.state)
        if artifact.exists():
            log.info(
                "stage_skipped",
                stage=stage.name,
                artifact_paths=[str(artifact)],
            )
            return

        log.info("stage_start", stage=stage.name)
        t0 = time.perf_counter()
        self.state = _run_stage(stage, self.state)
        duration_ms = (time.perf_counter() - t0) * 1000.0
        log.info(
            "stage_end",
            stage=stage.name,
            duration_ms=duration_ms,
            artifact_paths=[str(artifact)],
        )
        atomic_write_json(self.state, self._state_path)
