"""Stage protocol — the contract every agent must implement.

Every pipeline stage (QueryAgent, RetrievalAgent, ExtractionAgent,
IndexBuilder, AnalysisAgent, ReportGenerator, plus Phase 1 stubs) must
conform to the `Stage` protocol. The orchestrator (src/ara/orchestrator.py)
is `Stage`-generic: it invokes `.run(state)`, checks `.output_artifact_path`
for skip-on-exists, and enforces the single-writer invariant via
`.name` lookup in `FIELD_OWNERS`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, runtime_checkable

from ara.state import PipelineState


@runtime_checkable
class Stage(Protocol):
    """A pipeline stage.

    Attributes:
        name: one of "query" | "retrieval" | "extraction" | "indexing"
            | "analysis" | "report". Keys into `FIELD_OWNERS` for the
            single-writer invariant check.

    Methods:
        output_artifact_path(state):
            Path to the stage's completion marker / primary artifact.
            The orchestrator uses the existence of this path as the
            skip signal — delete the file to re-run the stage
            (delete-to-invalidate idempotency).

        run(state):
            Execute the stage. MUST only write to fields listed in
            `FIELD_OWNERS[self.name]` (plus append-only `stage_reports`).
            Any write outside that set triggers `FieldOwnershipViolation`
            via the orchestrator's snapshot-diff check.
    """

    name: str

    def output_artifact_path(self, state: PipelineState) -> Path: ...

    def run(self, state: PipelineState) -> PipelineState: ...
