"""Unit tests for QueryAgent (QRY-01..QRY-03).

CLI filter passthrough (QRY-04) is in test_cli.py::test_retrieval_filters_passthrough.
Network is blocked by pytest-socket; every test uses FakeLLM + a tmp_path Settings.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from ara.agents.query import QueryAgent
from ara.services.llm import FakeLLM
from ara.services.prompts import PromptLoader
from ara.state import PipelineState


REPO_ROOT = Path(__file__).resolve().parents[2]
PROMPTS_DIR = REPO_ROOT / "prompts"


def _hash(prompt: str) -> str:
    return hashlib.sha256(prompt.encode()).hexdigest()[:16]


def _make_state(tmp_path: Path, question: str = "How does retrieval-augmented generation reduce hallucination?") -> PipelineState:
    return PipelineState(
        run_id="R-QRY-01",
        question=question,
        config_snapshot={},
        runs_dir=str(tmp_path / "runs"),
    )


def _build_fake_llm(tmp_path: Path, *, expansion_terms, verdicts) -> tuple[FakeLLM, PromptLoader, PipelineState]:
    """Build a FakeLLM whose fixture answers the exact expansion + judge prompts for a given question."""
    state = _make_state(tmp_path)
    loader = PromptLoader(PROMPTS_DIR)
    expansion_prompt = loader.render("query_expansion.j2", question=state.question)
    judge_prompt = loader.render("query_judge.j2", question=state.question, candidate_terms=expansion_terms)

    fixture_data = {
        _hash(expansion_prompt): json.dumps({"terms": expansion_terms}),
        _hash(judge_prompt): json.dumps({"verdicts": verdicts}),
    }
    fixture_path = tmp_path / "fake.json"
    fixture_path.write_text(json.dumps(fixture_data))
    return FakeLLM(fixture_path=fixture_path), loader, state


def test_query_reads_question_from_state(tmp_path):
    """QRY-01: QueryAgent reads state.question (not a config var, not a CLI arg)."""
    llm, loader, state = _build_fake_llm(
        tmp_path,
        expansion_terms=["retrieval-augmented generation", "RAG hallucination", "grounded generation"],
        verdicts=[
            {"term": "retrieval-augmented generation", "on_topic": True, "reason": "core"},
            {"term": "RAG hallucination", "on_topic": True, "reason": "direct"},
            {"term": "grounded generation", "on_topic": True, "reason": "related"},
        ],
    )
    agent = QueryAgent(llm=llm, prompts=loader)
    out = agent.run(state)
    assert len(out.search_terms) == 3


def test_expansion_produces_three_to_five_terms(tmp_path):
    """QRY-02: 3–5 terms land in state.search_terms (when all pass judge)."""
    llm, loader, state = _build_fake_llm(
        tmp_path,
        expansion_terms=["RAG", "retrieval-augmented generation", "grounded generation", "retrieval augmentation"],
        verdicts=[{"term": t, "on_topic": True, "reason": "ok"}
                  for t in ["RAG", "retrieval-augmented generation", "grounded generation", "retrieval augmentation"]],
    )
    out = QueryAgent(llm=llm, prompts=loader).run(state)
    assert 3 <= len(out.search_terms) <= 5
    assert "RAG" in out.search_terms


def test_expansion_uses_temperature_point_three(tmp_path):
    """QRY-02: the expansion call runs at temperature=0.3 (diversity)."""
    llm, loader, state = _build_fake_llm(
        tmp_path,
        expansion_terms=["t1", "t2", "t3"],
        verdicts=[{"term": "t1", "on_topic": True, "reason": "x"},
                  {"term": "t2", "on_topic": True, "reason": "x"},
                  {"term": "t3", "on_topic": True, "reason": "x"}],
    )
    QueryAgent(llm=llm, prompts=loader).run(state)
    assert len(llm.calls) >= 1
    first_prompt, first_temp = llm.calls[0]
    assert first_temp == pytest.approx(0.3), f"expansion call temp must be 0.3; got {first_temp}"
    assert "STRICT JSON" in first_prompt or "terms" in first_prompt


def test_judge_filter_drops_off_topic_terms(tmp_path):
    """QRY-03: off-topic terms flagged by the judge are removed from search_terms."""
    llm, loader, state = _build_fake_llm(
        tmp_path,
        expansion_terms=["RAG", "retrieval-augmented generation", "deep learning", "machine learning"],
        verdicts=[
            {"term": "RAG", "on_topic": True, "reason": "core"},
            {"term": "retrieval-augmented generation", "on_topic": True, "reason": "core"},
            {"term": "deep learning", "on_topic": False, "reason": "too generic"},
            {"term": "machine learning", "on_topic": False, "reason": "too generic"},
        ],
    )
    out = QueryAgent(llm=llm, prompts=loader).run(state)
    assert "deep learning" not in out.search_terms
    assert "machine learning" not in out.search_terms
    assert "RAG" in out.search_terms
    assert "retrieval-augmented generation" in out.search_terms


def test_judge_uses_temperature_zero(tmp_path):
    """QRY-03: judge call runs at temperature=0.0 for reproducibility."""
    llm, loader, state = _build_fake_llm(
        tmp_path,
        expansion_terms=["RAG", "retrieval-augmented generation", "grounded generation"],
        verdicts=[{"term": t, "on_topic": True, "reason": "x"}
                  for t in ["RAG", "retrieval-augmented generation", "grounded generation"]],
    )
    QueryAgent(llm=llm, prompts=loader).run(state)
    assert len(llm.calls) == 2
    _, judge_temp = llm.calls[1]
    assert judge_temp == pytest.approx(0.0), f"judge call temp must be 0.0; got {judge_temp}"


def test_query_agent_writes_artifact_marker(tmp_path):
    """Delete-to-invalidate idempotency: output_artifact_path returns runs/<run_id>/query.marker."""
    llm, loader, state = _build_fake_llm(
        tmp_path,
        expansion_terms=["RAG", "retrieval augmentation", "grounded generation"],
        verdicts=[{"term": t, "on_topic": True, "reason": "x"}
                  for t in ["RAG", "retrieval augmentation", "grounded generation"]],
    )
    agent = QueryAgent(llm=llm, prompts=loader)
    out = agent.run(state)
    marker = agent.output_artifact_path(out)
    assert marker == Path(state.runs_dir) / state.run_id / "query.marker"
    assert marker.exists()


def test_query_agent_appends_stage_report(tmp_path):
    """QueryAgent appends exactly one StageReport with outputs_count == len(search_terms)."""
    llm, loader, state = _build_fake_llm(
        tmp_path,
        expansion_terms=["RAG", "retrieval augmentation", "deep learning"],
        verdicts=[
            {"term": "RAG", "on_topic": True, "reason": "ok"},
            {"term": "retrieval augmentation", "on_topic": True, "reason": "ok"},
            {"term": "deep learning", "on_topic": False, "reason": "generic"},
        ],
    )
    before_count = len(state.stage_reports)
    out = QueryAgent(llm=llm, prompts=loader).run(state)
    assert len(out.stage_reports) == before_count + 1
    rep = out.stage_reports[-1]
    assert rep.stage == "query"
    assert rep.outputs_count == len(out.search_terms) == 2
    assert any("deep learning" in w for w in rep.warnings)
