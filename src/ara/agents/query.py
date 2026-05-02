"""QueryAgent — expand user question + LLM-as-judge filter (QRY-01..03).

Two LLM calls per run:
  1. Expansion (temperature=0.3) → 3–5 JSON terms
  2. Judge (temperature=0.0) → per-term on_topic verdict

Off-topic terms from the judge are dropped before state.search_terms is written.
Writes the marker file runs/<run_id>/query.marker for delete-to-invalidate
idempotency (same pattern as StubQueryStage).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from ara.services.llm import LLMClient
from ara.services.prompts import PromptLoader
from ara.state import PipelineState, StageReport


@dataclass
class QueryAgent:
    """Implements the `Stage` protocol (structural conformance, not inheritance)."""

    llm: LLMClient
    prompts: PromptLoader
    name: str = "query"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "query.marker"

    def run(self, state: PipelineState) -> PipelineState:
        expansion_prompt = self.prompts.render(
            "query_expansion.j2", question=state.question
        )
        expansion_resp = self.llm.complete(
            expansion_prompt, temperature=0.3, schema={}
        )
        try:
            terms = json.loads(expansion_resp.text)["terms"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(
                f"QueryAgent expansion produced unparseable JSON: "
                f"{expansion_resp.text[:200]!r} ({e})"
            ) from e
        if not isinstance(terms, list) or not all(isinstance(t, str) for t in terms):
            raise ValueError(
                f"QueryAgent expected terms: list[str], got {terms!r}"
            )

        judge_prompt = self.prompts.render(
            "query_judge.j2", question=state.question, candidate_terms=terms
        )
        judge_resp = self.llm.complete(judge_prompt, temperature=0.0, schema={})
        try:
            verdicts = json.loads(judge_resp.text)["verdicts"]
        except (json.JSONDecodeError, KeyError, TypeError) as e:
            raise ValueError(
                f"QueryAgent judge produced unparseable JSON: "
                f"{judge_resp.text[:200]!r} ({e})"
            ) from e

        on_topic = {v["term"] for v in verdicts if v.get("on_topic")}
        dropped = [t for t in terms if t not in on_topic]
        kept = [t for t in terms if t in on_topic]

        state.search_terms = kept
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=1,
                outputs_count=len(kept),
                warnings=[f"judge dropped off-topic term: {t}" for t in dropped],
            )
        )

        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state
