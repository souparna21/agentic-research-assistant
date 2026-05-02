"""One-off regenerator for smoke.json + golden_state.json (plan 03-09).

Runs the full 6-stage smoke pipeline with a RECORDING FakeLLM that:

  1. Starts seeded with the Phase-2 committed QueryAgent hash map
     (retrieval search terms + on-topic verdicts for the smoke corpus).
  2. On any UNKNOWN prompt hash, inspects the prompt body to determine
     which of the 6 Phase-3 analysis prompts it is (per_paper_summary,
     methodology_extraction, limitations_extraction, verifier,
     methodology_matrix_synthesis, gap_synthesis) and returns a canned
     response body adapted to the smoke corpus's real paper_id
     ``fixture:test-rag-001``. The hash→response mapping is recorded.

After the run completes:

  3. Writes the merged hash→response map (Phase-2 seeds + 6 new Phase-3
     entries) into ``tests/fixtures/fake_llm/smoke.json``.
  4. Copies the produced ``runs/<rid>/state.json``, applies
     ``tests.e2e.test_smoke._normalize_state_dict``, and writes the
     normalized dict into ``tests/fixtures/golden_state.json`` sorted-key +
     indent=2 for diff-stability.

Run once from the project root:

    uv run python scripts/regen_smoke_fixtures.py

Idempotent — safe to re-run if something changes in the prompt templates
or the 2-paper smoke corpus. Commit both updated fixtures after.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
TESTS_DIR = REPO_ROOT / "tests"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Force HF offline mode so MiniLMEmbedder does not attempt a network fetch.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

from pydantic import BaseModel  # noqa: E402

from ara.analysis_schemas import (  # noqa: E402
    GapAnalysis,
    LimitationsExtraction,
    MethodologyExtraction,
    MethodologyMatrix,
    PerPaperSummary,
    VerifierVerdict,
)
from ara.services.llm import LLMClient, LLMResponse  # noqa: E402

# The smoke corpus has ONE full-text paper: fixture:test-rag-001 (from
# tests/fixtures/papers/paper_with_pdf.json). We pin the canned-response
# paper_id to this value so every Claim.citations entry validates against
# the [P<paper_id>-N] regex with matching paper_id.
SMOKE_PAPER_ID = "fixture:test-rag-001"
SMOKE_PAPER_TITLE = "Test Paper - RAG Fixture"

SMOKE_JSON = TESTS_DIR / "fixtures" / "fake_llm" / "smoke.json"
GOLDEN_JSON = TESTS_DIR / "fixtures" / "golden_state.json"


# --------------------------------------------------------------------------
# Canned response bodies adapted for SMOKE_PAPER_ID
# --------------------------------------------------------------------------


def _per_paper_summary_body() -> str:
    return PerPaperSummary(
        paper_id=SMOKE_PAPER_ID,
        objective={
            "text": "Introduce retrieval-augmented generation to reduce factual hallucination.",
            "citations": [f"[P{SMOKE_PAPER_ID}-1]"],
        },
        methodology={
            "text": "Combine a dense passage retriever with a seq2seq generator trained end-to-end.",
            "citations": [f"[P{SMOKE_PAPER_ID}-2]"],
        },
        findings=[
            {
                "text": "Outperforms parametric seq2seq on open-domain QA.",
                "citations": [f"[P{SMOKE_PAPER_ID}-3]"],
            },
            {
                "text": "Retrieval source attribution improves provenance.",
                "citations": [f"[P{SMOKE_PAPER_ID}-4]"],
            },
        ],
        limitations=[
            {
                "text": "Requires a high-quality dense retriever index.",
                "citations": [f"[P{SMOKE_PAPER_ID}-5]"],
            },
        ],
    ).model_dump_json()


def _methodology_extraction_body() -> str:
    return MethodologyExtraction(
        paper_id=SMOKE_PAPER_ID,
        datasets=["Natural Questions", "TriviaQA"],
        techniques=["Dense Passage Retrieval", "BART seq2seq"],
        metrics=["Exact Match", "F1"],
    ).model_dump_json()


def _limitations_extraction_body() -> str:
    return LimitationsExtraction(
        paper_id=SMOKE_PAPER_ID,
        limitations=[
            {
                "text": "Retrieval accuracy bounds generation quality.",
                "citations": [f"[P{SMOKE_PAPER_ID}-1]"],
            },
        ],
        future_work=[
            {
                "text": "Investigate HyDE-style query transformation.",
                "citations": [f"[P{SMOKE_PAPER_ID}-2]"],
            },
        ],
    ).model_dump_json()


def _verifier_verdict_body() -> str:
    # Must mention EVERY claim from the summary above. One supported + one
    # unsupported verdict demonstrates the [UNVERIFIED] tagging path without
    # running the cross-paper synthesis on a single-paper corpus.
    return VerifierVerdict(
        claims=[
            {
                "claim_text": "Introduce retrieval-augmented generation to reduce factual hallucination.",
                "citations": [f"[P{SMOKE_PAPER_ID}-1]"],
                "verdict": "supported",
                "reason": "Passage 1 describes RAG as a hallucination-mitigation technique.",
            },
            {
                "claim_text": "Combine a dense passage retriever with a seq2seq generator trained end-to-end.",
                "citations": [f"[P{SMOKE_PAPER_ID}-2]"],
                "verdict": "supported",
                "reason": "Passage 2 describes the retriever-generator composition.",
            },
            {
                "claim_text": "Outperforms parametric seq2seq on open-domain QA.",
                "citations": [f"[P{SMOKE_PAPER_ID}-3]"],
                "verdict": "supported",
                "reason": "Passage 3 reports EM gains over seq2seq baselines.",
            },
            {
                "claim_text": "Retrieval source attribution improves provenance.",
                "citations": [f"[P{SMOKE_PAPER_ID}-4]"],
                "verdict": "partial",
                "reason": "Passage 4 hints at provenance but does not quantify improvements.",
            },
            {
                "claim_text": "Requires a high-quality dense retriever index.",
                "citations": [f"[P{SMOKE_PAPER_ID}-5]"],
                "verdict": "supported",
                "reason": "Passage 5 states that retrieval quality bounds the generator.",
            },
        ]
    ).model_dump_json()


def _methodology_matrix_body() -> str:
    return MethodologyMatrix(
        rows=[
            {
                "paper_id": SMOKE_PAPER_ID,
                "paper_title": SMOKE_PAPER_TITLE,
                "datasets": ["Natural Questions", "TriviaQA"],
                "techniques": ["Dense Passage Retrieval", "BART seq2seq"],
                "metrics": ["Exact Match", "F1"],
            }
        ]
    ).model_dump_json()


def _gap_analysis_body() -> str:
    return GapAnalysis(
        gaps=[
            {
                "text": "No corpus covers both scientific and legal domains simultaneously.",
                "citations": [f"[P{SMOKE_PAPER_ID}-1]"],
            },
            {
                "text": "Evaluation of citation groundedness is underspecified.",
                "citations": [f"[P{SMOKE_PAPER_ID}-2]"],
            },
        ]
    ).model_dump_json()


# --------------------------------------------------------------------------
# Recording FakeLLM — classifies unknown prompts by body + records hashes
# --------------------------------------------------------------------------


@dataclass
class _RecordingFakeLLM(LLMClient):
    """FakeLLM wrapper that classifies unknown prompts and records hashes."""

    seed_responses: dict[str, str]
    recorded: dict[str, str] = field(default_factory=dict)

    @staticmethod
    def _hash(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]

    def _classify_and_respond(self, prompt: str, schema: Any) -> str:
        # Classify by unique OUTPUT-spec substring. Each analysis template's
        # ``OUTPUT: STRICT JSON matching/conforming to <ClassName>`` line is
        # the unambiguous discriminator. Order matters — verifier.j2 embeds
        # a PerPaperSummary dict in its SUMMARY block, and gap_synthesis.j2
        # embeds a MethodologyMatrix dict in its METHODOLOGY MATRIX block,
        # so we test each prompt by its OUTPUT-spec-unique phrase FIRST.
        if "matching VerifierVerdict" in prompt:
            return _verifier_verdict_body()
        if "matching GapAnalysis" in prompt:
            return _gap_analysis_body()
        if "matching MethodologyMatrix" in prompt:
            return _methodology_matrix_body()
        if "matching MethodologyExtraction" in prompt:
            return _methodology_extraction_body()
        if "matching LimitationsExtraction" in prompt:
            return _limitations_extraction_body()
        if "conforming to PerPaperSummary" in prompt:
            return _per_paper_summary_body()
        raise RuntimeError(
            f"Recording FakeLLM cannot classify prompt (first 400 chars): {prompt[:400]!r}"
        )

    def complete(
        self,
        prompt: str,
        *,
        temperature: float = 0.0,
        schema: type[BaseModel] | dict[str, Any] | None = None,
    ) -> LLMResponse:
        key = self._hash(prompt)
        if key in self.seed_responses:
            text = self.seed_responses[key]
        elif key in self.recorded:
            text = self.recorded[key]
        else:
            text = self._classify_and_respond(prompt, schema)
            self.recorded[key] = text
        return LLMResponse(text=text, model="fake-recording", prompt_tokens=0, completion_tokens=0)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------


def main() -> None:
    from ara.agents.analysis import AnalysisAgent
    from ara.agents.extraction import ExtractionAgent
    from ara.agents.indexing import IndexingAgent
    from ara.agents.query import QueryAgent
    from ara.agents.report import ReportAgent
    from ara.agents.retrieval import RetrievalAgent
    from ara.agents.stubs import _SmokePdfDownloader, _SmokeS2, _SmokeArxiv, FIXTURE_DIR
    from ara.config import Settings
    from ara.orchestrator import Orchestrator
    from ara.services.embed import MiniLMEmbedder
    from ara.services.pdf import PdfExtractor
    from ara.services.prompts import LatexTemplateLoader, PromptLoader

    # Load the existing Phase-2 seed hashes
    seed = json.loads(SMOKE_JSON.read_text(encoding="utf-8"))
    recorder = _RecordingFakeLLM(seed_responses=seed)

    prompts_dir = REPO_ROOT / "prompts"
    papers_dir = FIXTURE_DIR / "papers"

    prompts = PromptLoader(prompts_dir=prompts_dir)
    latex_templates = LatexTemplateLoader(prompts_dir=prompts_dir)
    embedder = MiniLMEmbedder()

    query = QueryAgent(llm=recorder, prompts=prompts)
    retrieval = RetrievalAgent(
        s2=_SmokeS2(papers_dir=papers_dir),  # type: ignore[arg-type]
        arxiv=_SmokeArxiv(),  # type: ignore[arg-type]
        downloader=_SmokePdfDownloader(papers_dir=papers_dir),  # type: ignore[arg-type]
    )
    extraction = ExtractionAgent(pdf_extractor=PdfExtractor())
    indexing = IndexingAgent(embedder=embedder)
    analysis = AnalysisAgent(llm=recorder, prompts=prompts)
    report = ReportAgent(latex_templates=latex_templates)

    stages = [query, retrieval, extraction, indexing, analysis, report]

    # Run the full pipeline in a scratch tmp dir
    with tempfile.TemporaryDirectory(prefix="regen-smoke-") as tmp:
        tmp_path = Path(tmp)
        settings = Settings(runs_dir=str(tmp_path / "runs"))
        run_id = "20260101-000000-regen-smoke"
        git_sha = "GOLDEN"  # matches GOLDEN_GIT_SHA in test_smoke.py

        orch = Orchestrator(
            settings,
            run_id,
            git_sha=git_sha,
            question="What is retrieval-augmented generation?",
            stages=stages,
        )
        orch.run()

        produced_state = json.loads(
            (tmp_path / "runs" / run_id / "state.json").read_text(encoding="utf-8")
        )

        # Copy the completed run dir to a stable debug location (optional)
        debug_runs = REPO_ROOT / ".tmp" / "regen-smoke-last"
        if debug_runs.exists():
            shutil.rmtree(debug_runs)
        debug_runs.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(tmp_path / "runs", debug_runs)
        print(f"Debug run copied to: {debug_runs}")

    # Merge recorded hashes into smoke.json (preserving Phase-2 seeds)
    merged = dict(seed)
    for key, text in recorder.recorded.items():
        merged[key] = text
    SMOKE_JSON.write_text(
        json.dumps(merged, indent=2, sort_keys=False), encoding="utf-8"
    )
    print(f"Wrote smoke.json with {len(merged)} entries (Phase-2 seeds={len(seed)}, "
          f"Phase-3 new={len(recorder.recorded)}).")

    # Normalize the produced state using the smoke test's helper
    sys.path.insert(0, str(TESTS_DIR))
    from tests.e2e.test_smoke import _normalize_state_dict  # type: ignore[import]

    normalized = _normalize_state_dict(produced_state)
    GOLDEN_JSON.write_text(
        json.dumps(normalized, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"Wrote golden_state.json ({len(json.dumps(normalized))} bytes normalized).")


if __name__ == "__main__":
    main()
