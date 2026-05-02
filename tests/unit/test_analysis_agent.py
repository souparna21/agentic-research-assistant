"""Unit tests for AnalysisAgent (ANL-01..09).

VALIDATION.md task IDs:
  3-06-01  -> test_per_paper_summary_produced                        (ANL-01)
  3-06-02  -> test_citations_resolve_to_real_chunks                  (ANL-02)
  3-06-03  -> test_methodology_matrix_produced                       (ANL-03)
  3-06-04  -> test_matrix_row_per_successful_paper                   (ANL-03)
  3-06-05  -> test_gap_analysis_produced                             (ANL-04)
  3-06-06  -> test_hierarchical_per_paper_then_cross_paper           (ANL-05)
  3-06-07  -> test_synthesis_consumes_typed_extractions_not_raw_passages  (ANL-05)
  3-06-08  -> test_all_calls_use_schema_and_temperature_zero         (ANL-06)
  3-06-09  -> test_schema_violation_recorded_as_failure              (ANL-06)
  3-06-10  -> test_temperature_zero_on_every_call                    (ANL-07)
  3-06-11  -> test_analysis_is_deterministic                         (ANL-07)
  3-06-12  -> test_verifier_fires_per_summary                        (ANL-08)
  3-06-13  -> test_unsupported_claims_marked_unverified              (ANL-08)
  3-06-14  -> test_per_paper_failure_isolation                       (ANL-09)
  3-06-15  -> test_failure_captured_in_analysis_stats                (ANL-09)

Fixture-assembly rationale:
  tests/fixtures/fake_llm/analysis_scenarios.json is a ``{}`` placeholder in
  the committed tree. Prompt hashes depend on the exact paper data passed
  into the Jinja template, so every test assembles its own tmp-dir scenarios
  file via :func:`_build_scenarios_file` which:
    1. Renders each prompt against the test corpus
    2. Maps sha256(prompt)[:16] -> tests/fixtures/schemas/*.valid.json body
    3. Writes the resulting dict to a tmp-dir JSON file
  This makes the hash-key map self-regenerating when fixture paper metadata
  changes (no manual re-capture step).
"""
from __future__ import annotations

import copy
import json
from hashlib import sha256
from pathlib import Path

import pytest

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
SCHEMAS_DIR = FIXTURES / "schemas"
PROMPTS_DIR = Path(__file__).resolve().parents[2] / "prompts"


@pytest.fixture(autouse=True)
def _hf_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the cached MiniLM snapshot; do not ping HuggingFace Hub."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _hash(prompt: str) -> str:
    return sha256(prompt.encode()).hexdigest()[:16]


def _make_papers():
    from ara.state import ExtractionMeta, Paper

    p1 = Paper(
        paper_id="paper1",
        title="Retrieval-Augmented Generation",
        year=2020,
        authors=["Lewis"],
        extraction_ok=True,
        full_text_available=True,
        extracted_md_path="rid/papers/paper1/extracted.md",
        extraction_meta=ExtractionMeta(num_detected_sections=4, pages=8),
    )
    p2 = Paper(
        paper_id="paper2",
        title="A Baseline for RAG",
        year=2021,
        authors=["Karpukhin"],
        extraction_ok=True,
        full_text_available=True,
        extracted_md_path="rid/papers/paper2/extracted.md",
        extraction_meta=ExtractionMeta(num_detected_sections=3, pages=4),
    )
    return [p1, p2]


def _seed_index(tmp_path: Path):
    """Build a 2-paper FAISS index with synthetic chunks under tmp_path.

    Returns ``(runs_dir, index_path_rel, chunks, store)``. Writes the index to
    ``tmp_path/runs/rid/index/faiss.index`` so ``Path(state.runs_dir) /
    state.index_path`` resolves correctly. Returns the built ``FaissVectorStore``
    so :func:`_per_paper_passages` can mirror the agent's top-k retrieval
    (including the real similarity scores used in the rendered prompt).
    """
    from ara.services.chunker import Chunk
    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore

    chunks = [
        Chunk(paper_id="paper1", page=1, section="Abstract", chunk_index=0,
              chunk_hash="h_p1_0", token_count=42,
              text="SYNCHUNK-P1-0: alpha beta gamma delta epsilon zeta eta theta iota."),
        Chunk(paper_id="paper1", page=1, section="Methods", chunk_index=1,
              chunk_hash="h_p1_1", token_count=42,
              text="SYNCHUNK-P1-1: kappa lambda mu nu xi omicron pi rho sigma tau."),
        Chunk(paper_id="paper1", page=2, section="Methods", chunk_index=2,
              chunk_hash="h_p1_2", token_count=42,
              text="SYNCHUNK-P1-2: upsilon phi chi psi omega alpha beta gamma delta."),
        Chunk(paper_id="paper1", page=3, section="Results", chunk_index=3,
              chunk_hash="h_p1_3", token_count=40,
              text="SYNCHUNK-P1-3: red orange yellow green blue indigo violet white."),
        Chunk(paper_id="paper1", page=3, section="Results", chunk_index=4,
              chunk_hash="h_p1_4", token_count=40,
              text="SYNCHUNK-P1-4: north south east west up down left right center."),
        Chunk(paper_id="paper1", page=4, section="Limitations", chunk_index=5,
              chunk_hash="h_p1_5", token_count=40,
              text="SYNCHUNK-P1-5: first second third fourth fifth sixth seventh eighth."),
        Chunk(paper_id="paper1", page=4, section="Limitations", chunk_index=6,
              chunk_hash="h_p1_6", token_count=40,
              text="SYNCHUNK-P1-6: spring summer autumn winter monsoon dry wet humid."),
        Chunk(paper_id="paper1", page=5, section="Future Work", chunk_index=7,
              chunk_hash="h_p1_7", token_count=40,
              text="SYNCHUNK-P1-7: apple banana cherry durian elderberry fig grape honeydew."),
        Chunk(paper_id="paper2", page=1, section="Abstract", chunk_index=0,
              chunk_hash="h_p2_0", token_count=42,
              text="SYNCHUNK-P2-0: mercury venus earth mars jupiter saturn uranus neptune."),
        Chunk(paper_id="paper2", page=1, section="Methods", chunk_index=1,
              chunk_hash="h_p2_1", token_count=40,
              text="SYNCHUNK-P2-1: hydrogen helium lithium beryllium boron carbon nitrogen oxygen."),
        Chunk(paper_id="paper2", page=2, section="Methods", chunk_index=2,
              chunk_hash="h_p2_2", token_count=40,
              text="SYNCHUNK-P2-2: piano guitar drums violin cello flute trumpet saxophone."),
        Chunk(paper_id="paper2", page=2, section="Results", chunk_index=3,
              chunk_hash="h_p2_3", token_count=40,
              text="SYNCHUNK-P2-3: oak pine birch maple willow cedar elm cypress redwood."),
        Chunk(paper_id="paper2", page=3, section="Results", chunk_index=4,
              chunk_hash="h_p2_4", token_count=40,
              text="SYNCHUNK-P2-4: rock pop jazz classical blues country electronic folk."),
        Chunk(paper_id="paper2", page=3, section="Limitations", chunk_index=5,
              chunk_hash="h_p2_5", token_count=40,
              text="SYNCHUNK-P2-5: granite basalt limestone sandstone marble slate quartzite."),
        Chunk(paper_id="paper2", page=4, section="Discussion", chunk_index=6,
              chunk_hash="h_p2_6", token_count=40,
              text="SYNCHUNK-P2-6: eagle hawk falcon osprey kestrel buzzard harrier vulture."),
    ]
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])
    runs_dir = tmp_path / "runs"
    idx_path = runs_dir / "rid" / "index" / "faiss.index"
    chunks_path = runs_dir / "rid" / "index" / "chunks.json"
    store = FaissVectorStore(embedder=embedder)
    store.build_and_persist(chunks, embeddings, idx_path, chunks_path)
    return runs_dir, str(idx_path.relative_to(runs_dir)), chunks, store


def _load_schema_fixture(name: str, paper_id: str | None = None) -> dict:
    """Load a canned valid-fixture JSON from tests/fixtures/schemas/.

    If ``paper_id`` is supplied, rewrites the fixture's default "paper1"
    paper_id field AND every citation marker "[Ppaper1-N]" -> "[P<paper_id>-N]".
    Deep-copies so repeated calls don't alias.
    """
    data = json.loads((SCHEMAS_DIR / name).read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    if paper_id and paper_id != "paper1":
        def _walk(obj):
            if isinstance(obj, dict):
                for k, v in list(obj.items()):
                    if k == "paper_id" and isinstance(v, str) and v == "paper1":
                        obj[k] = paper_id
                    elif k == "citations" and isinstance(v, list):
                        obj[k] = [c.replace("Ppaper1", f"P{paper_id}") for c in v]
                    else:
                        _walk(v)
            elif isinstance(obj, list):
                for item in obj:
                    _walk(item)

        _walk(data)
    return data


def _per_paper_passages(store, paper, *, top_k: int = 8):
    """Return the list of passage dicts an AnalysisAgent would pass to a per-paper prompt.

    Mirrors :meth:`AnalysisAgent.run` exactly: same query string, same k, same
    per-paper filter, same dict shape — including the real FAISS similarity
    score (not a mock). Rendering the Jinja template with these passages
    produces the same prompt string (and thus the same sha256 hash) the agent
    will produce at test time.
    """
    retrieved = store.search(
        f"What is the objective, methodology, findings, limitations of "
        f"'{paper.title}'?",
        k=top_k,
    )
    scoped = [c for c in retrieved if c.paper_id == paper.paper_id]
    effective = scoped or retrieved
    return [
        {"paper_id": c.paper_id, "section": c.section, "score": c.score, "text": c.text}
        for c in effective
    ]


def _build_scenarios_file(
    tmp_path: Path,
    papers,
    store,
    *,
    override_responses: dict[str, str] | None = None,
) -> Path:
    """Render all prompts against the test corpus; write prompt-hash -> canned JSON.

    Response bodies come from tests/fixtures/schemas/*.valid.json, rewritten
    per-paper via :func:`_load_schema_fixture`.

    ``override_responses`` lets a test inject a malformed JSON for a specific
    logical slot (e.g. {"per_paper_summary:paper1": "{not-json"}) — keyed by
    ``{template_stem}:{paper_id_or_'cross'}``.
    """
    from ara.services.prompts import PromptLoader

    loader = PromptLoader(prompts_dir=PROMPTS_DIR)
    overrides = override_responses or {}

    entries: dict[str, str] = {}
    per_paper_extractions: list[dict] = []
    per_paper_limitations: list[dict] = []

    for p in papers:
        passages = _per_paper_passages(store, p)

        prompt = loader.render("per_paper_summary.j2", paper=p, passages=passages)
        body = _load_schema_fixture("per_paper_summary.valid.json", paper_id=p.paper_id)
        key = f"per_paper_summary:{p.paper_id}"
        entries[_hash(prompt)] = overrides.get(key, json.dumps(body))

        prompt = loader.render("methodology_extraction.j2", paper=p, passages=passages)
        body = _load_schema_fixture("methodology_extraction.valid.json", paper_id=p.paper_id)
        key = f"methodology_extraction:{p.paper_id}"
        entries[_hash(prompt)] = overrides.get(key, json.dumps(body))
        per_paper_extractions.append({
            "paper_id": p.paper_id,
            "paper_title": p.title,
            "datasets": body["datasets"],
            "techniques": body["techniques"],
            "metrics": body["metrics"],
        })

        prompt = loader.render("limitations_extraction.j2", paper=p, passages=passages)
        body = _load_schema_fixture("limitations_extraction.valid.json", paper_id=p.paper_id)
        key = f"limitations_extraction:{p.paper_id}"
        entries[_hash(prompt)] = overrides.get(key, json.dumps(body))
        per_paper_limitations.append(body)

        summary_body = _load_schema_fixture("per_paper_summary.valid.json", paper_id=p.paper_id)
        verifier_passages = [
            {"paper_id": pp["paper_id"], "text": pp["text"]} for pp in passages
        ]
        prompt = loader.render(
            "verifier.j2", summary=summary_body, passages=verifier_passages
        )
        body = _load_schema_fixture("verifier_verdict.valid.json")
        key = f"verifier:{p.paper_id}"
        entries[_hash(prompt)] = overrides.get(key, json.dumps(body))

    from itertools import chain, combinations

    def _subsets(items: list):
        return chain.from_iterable(
            combinations(items, r) for r in range(1, len(items) + 1)
        )

    base_matrix = _load_schema_fixture("methodology_matrix.valid.json")
    gap_body = _load_schema_fixture("gap_analysis.valid.json")

    for subset_pairs in _subsets(list(zip(per_paper_extractions, per_paper_limitations, strict=True))):
        sub_ext = [e for e, _ in subset_pairs]
        sub_lim = [lim for _, lim in subset_pairs]

        prompt = loader.render(
            "methodology_matrix_synthesis.j2", extractions=sub_ext
        )
        subset_rows = []
        for i, ext in enumerate(sub_ext):
            row_tpl = copy.deepcopy(
                base_matrix["rows"][i % len(base_matrix["rows"])]
            )
            row_tpl["paper_id"] = ext["paper_id"]
            row_tpl["paper_title"] = ext["paper_title"]
            subset_rows.append(row_tpl)
        matrix_body = {"rows": subset_rows}
        entries[_hash(prompt)] = overrides.get(
            "methodology_matrix_synthesis:cross", json.dumps(matrix_body)
        )

        prompt = loader.render(
            "gap_synthesis.j2", matrix=matrix_body, limitations=sub_lim
        )
        entries[_hash(prompt)] = overrides.get(
            "gap_synthesis:cross", json.dumps(gap_body)
        )

    out = tmp_path / "analysis_scenarios.json"
    out.write_text(json.dumps(entries, indent=2), encoding="utf-8")
    return out


def _make_state(tmp_path: Path, papers, runs_dir: Path, index_path_rel: str, chunks):
    from ara.state import PipelineState

    return PipelineState(
        run_id="rid",
        question="q?",
        config_snapshot={"model": "fake"},
        runs_dir=str(runs_dir),
        papers=papers,
        chunks_count=len(chunks),
        index_path=index_path_rel,
    )


def _run_agent(tmp_path: Path, *, override_responses: dict[str, str] | None = None):
    """Full boilerplate: build index, make papers, seed fixture, run agent.

    Returns (state, agent, scenarios_path, chunks).
    """
    from ara.agents.analysis import AnalysisAgent
    from ara.services.llm import FakeLLM
    from ara.services.prompts import PromptLoader

    papers = _make_papers()
    runs_dir, idx_rel, chunks, store = _seed_index(tmp_path)
    scenarios = _build_scenarios_file(
        tmp_path, papers, store, override_responses=override_responses
    )
    state = _make_state(tmp_path, papers, runs_dir, idx_rel, chunks)
    agent = AnalysisAgent(
        llm=FakeLLM(scenarios), prompts=PromptLoader(PROMPTS_DIR)
    )
    state = agent.run(state)
    return state, agent, scenarios, chunks




def test_per_paper_summary_produced(tmp_path: Path) -> None:
    """Task 3-06-01 — AnalysisAgent emits summaries_dir/<paper_id>.json per paper."""
    state, _, _, _ = _run_agent(tmp_path)

    assert state.summaries_dir
    summaries_dir = Path(state.summaries_dir)
    if not summaries_dir.is_absolute():
        summaries_dir = Path(state.runs_dir) / state.summaries_dir
    assert (summaries_dir / "paper1.json").exists()
    assert (summaries_dir / "paper2.json").exists()


def test_citations_resolve_to_real_chunks(tmp_path: Path) -> None:
    """Task 3-06-02 — Every citation in a summary maps to a real chunk in the index."""
    import re

    state, _, _, chunks = _run_agent(tmp_path)

    summaries_dir = Path(state.summaries_dir)
    if not summaries_dir.is_absolute():
        summaries_dir = Path(state.runs_dir) / state.summaries_dir

    cite_re = re.compile(r"\[P([^\]]+)-(\d+)\]")
    for paper_id in ("paper1", "paper2"):
        per_paper_chunks = [c for c in chunks if c.paper_id == paper_id]
        data = json.loads((summaries_dir / f"{paper_id}.json").read_text(encoding="utf-8"))
        citations: list[str] = []
        summary = data["summary"]
        for field in ("objective", "methodology"):
            citations.extend(summary[field]["citations"])
        for claim in summary["findings"] + summary["limitations"]:
            citations.extend(claim["citations"])
        assert citations, f"{paper_id} summary has no citations"
        for c in citations:
            m = cite_re.fullmatch(c)
            assert m, f"citation {c!r} doesn't match [P<paper_id>-N]"
            cited_paper, n_str = m.group(1), m.group(2)
            assert cited_paper == paper_id, (
                f"citation {c} in {paper_id} summary points at {cited_paper}"
            )
            n = int(n_str)
            assert 1 <= n <= len(per_paper_chunks), (
                f"citation {c} indexes past {len(per_paper_chunks)} chunks for {paper_id}"
            )


def test_methodology_matrix_produced(tmp_path: Path) -> None:
    """Task 3-06-03 — comparison.json exists + parses as MethodologyMatrix."""
    from ara.analysis_schemas import MethodologyMatrix

    state, _, _, _ = _run_agent(tmp_path)

    assert state.comparison_path
    cp = Path(state.comparison_path)
    if not cp.is_absolute():
        cp = Path(state.runs_dir) / state.comparison_path
    data = json.loads(cp.read_text(encoding="utf-8"))
    MethodologyMatrix.model_validate(data)


def test_matrix_row_per_successful_paper(tmp_path: Path) -> None:
    """Task 3-06-04 — matrix.rows has one entry per paper that succeeded analysis."""
    from ara.analysis_schemas import MethodologyMatrix

    state, _, _, _ = _run_agent(tmp_path)

    succeeded = [p for p in state.papers if p.analysis_status == "succeeded"]
    cp = Path(state.comparison_path)
    if not cp.is_absolute():
        cp = Path(state.runs_dir) / state.comparison_path
    matrix = MethodologyMatrix.model_validate_json(cp.read_text(encoding="utf-8"))
    assert len(matrix.rows) == len(succeeded)


def test_gap_analysis_produced(tmp_path: Path) -> None:
    """Task 3-06-05 — gaps.json exists + parses as GapAnalysis."""
    from ara.analysis_schemas import GapAnalysis

    state, _, _, _ = _run_agent(tmp_path)

    assert state.gaps_path
    gp = Path(state.gaps_path)
    if not gp.is_absolute():
        gp = Path(state.runs_dir) / state.gaps_path
    data = json.loads(gp.read_text(encoding="utf-8"))
    GapAnalysis.model_validate(data)


def test_hierarchical_per_paper_then_cross_paper(tmp_path: Path) -> None:
    """Task 3-06-06 — all per-paper LLM calls happen before any cross-paper call.

    The final 2 calls must render methodology_matrix_synthesis / gap_synthesis
    (recognized by distinctive per-template string markers); all prior calls
    are per-paper prompts.
    """
    _, agent, _, _ = _run_agent(tmp_path)

    calls = agent.llm.calls
    assert len(calls) >= 2

    penultimate_prompt, _ = calls[-2]
    final_prompt, _ = calls[-1]
    assert "Synthesize a cross-paper methodology comparison matrix" in penultimate_prompt
    assert "Identify research gaps by diffing cross-paper methodology matrix" in final_prompt

    for prompt, _temp in calls[:-2]:
        assert "Synthesize a cross-paper methodology comparison matrix" not in prompt
        assert "Identify research gaps by diffing cross-paper methodology matrix" not in prompt


def test_synthesis_consumes_typed_extractions_not_raw_passages(tmp_path: Path) -> None:
    """Task 3-06-07 — synthesis prompts contain typed JSON but NO raw chunk passages (P7 defense)."""
    _, agent, _, chunks = _run_agent(tmp_path)

    calls = agent.llm.calls
    for prompt, _temp in calls[-2:]:
        for chunk in chunks:
            marker = chunk.text.split(":")[0]
            assert marker not in prompt, (
                f"synthesis prompt leaked raw passage text: {marker!r}"
            )


def test_all_calls_use_schema_and_temperature_zero(tmp_path: Path) -> None:
    """Task 3-06-08 — Every LLM call passes schema=BaseModel + temperature=0.0.

    Wraps FakeLLM in an instrumented subclass that records (temperature, schema)
    tuples in addition to (prompt, temperature). Asserts every call has
    temperature==0.0 and schema is one of the 6 Phase 3 BaseModel classes.
    """
    from ara.agents.analysis import AnalysisAgent
    from ara.analysis_schemas import (
        GapAnalysis,
        LimitationsExtraction,
        MethodologyExtraction,
        MethodologyMatrix,
        PerPaperSummary,
        VerifierVerdict,
    )
    from ara.services.llm import FakeLLM, LLMResponse
    from ara.services.prompts import PromptLoader

    expected_schemas = {
        PerPaperSummary,
        MethodologyExtraction,
        LimitationsExtraction,
        MethodologyMatrix,
        GapAnalysis,
        VerifierVerdict,
    }

    papers = _make_papers()
    runs_dir, idx_rel, chunks, store = _seed_index(tmp_path)
    scenarios = _build_scenarios_file(tmp_path, papers, store)

    class _InstrumentedFake(FakeLLM):
        records: list[tuple[float, object]] = []

        def complete(  # type: ignore[override]
            self,
            prompt: str,
            *,
            temperature: float = 0.0,
            schema=None,
        ) -> LLMResponse:
            self.records.append((temperature, schema))
            return super().complete(prompt, temperature=temperature, schema=schema)

    _InstrumentedFake.records = []
    agent = AnalysisAgent(
        llm=_InstrumentedFake(scenarios), prompts=PromptLoader(PROMPTS_DIR)
    )
    state = _make_state(tmp_path, papers, runs_dir, idx_rel, chunks)
    agent.run(state)

    assert _InstrumentedFake.records, "no LLM calls recorded"
    for temp, schema in _InstrumentedFake.records:
        assert temp == 0.0, f"temperature={temp} on some call — ANL-07 violation"
        assert schema in expected_schemas, (
            f"schema={schema} not in expected {expected_schemas}"
        )


def test_schema_violation_recorded_as_failure(tmp_path: Path) -> None:
    """Task 3-06-09 — malformed LLM response -> per-paper failure recorded (ANL-06 + ANL-09)."""
    state, _, _, _ = _run_agent(
        tmp_path,
        override_responses={"per_paper_summary:paper1": "{not-json"},
    )

    assert state.analysis_stats.analysis_failures, (
        "expected analysis_failures to contain a (paper_id, reason) entry"
    )
    pids = [pid for pid, _ in state.analysis_stats.analysis_failures]
    assert "paper1" in pids


def test_temperature_zero_on_every_call(tmp_path: Path) -> None:
    """Task 3-06-10 — grep-level: every FakeLLM.calls entry has temperature==0.0."""
    _, agent, _, _ = _run_agent(tmp_path)

    for _prompt, temp in agent.llm.calls:
        assert temp == 0.0, f"non-deterministic temperature detected: {temp}"


def test_analysis_is_deterministic(tmp_path: Path) -> None:
    """Task 3-06-11 — Running AnalysisAgent twice -> bit-equal summaries/paper1.json."""
    from ara.agents.analysis import AnalysisAgent
    from ara.services.llm import FakeLLM
    from ara.services.prompts import PromptLoader

    papers = _make_papers()
    runs_dir, idx_rel, chunks, store = _seed_index(tmp_path)
    scenarios = _build_scenarios_file(tmp_path, papers, store)

    state_a = _make_state(tmp_path, papers, runs_dir, idx_rel, chunks)
    AnalysisAgent(
        llm=FakeLLM(scenarios), prompts=PromptLoader(PROMPTS_DIR)
    ).run(state_a)
    summaries_dir_a = Path(state_a.runs_dir) / state_a.summaries_dir
    content_a = (summaries_dir_a / "paper1.json").read_text(encoding="utf-8")

    papers_b = _make_papers()
    state_b = _make_state(tmp_path, papers_b, runs_dir, idx_rel, chunks)
    AnalysisAgent(
        llm=FakeLLM(scenarios), prompts=PromptLoader(PROMPTS_DIR)
    ).run(state_b)
    summaries_dir_b = Path(state_b.runs_dir) / state_b.summaries_dir
    content_b = (summaries_dir_b / "paper1.json").read_text(encoding="utf-8")

    assert json.loads(content_a) == json.loads(content_b)


def test_verifier_fires_per_summary(tmp_path: Path) -> None:
    """Task 3-06-12 — verifier.j2 called once per per-paper summary (N total)."""
    state, agent, _, _ = _run_agent(tmp_path)

    verifier_call_count = sum(
        1 for prompt, _ in agent.llm.calls
        if "You are a fact-check verifier." in prompt
    )
    succeeded = [p for p in state.papers if p.analysis_status == "succeeded"]
    assert verifier_call_count == len(succeeded)
    assert state.analysis_stats.verifier_calls == len(succeeded)


def test_unsupported_claims_marked_unverified(tmp_path: Path) -> None:
    """Task 3-06-13 — claims with verdict='unsupported' get [UNVERIFIED] tag prepended."""
    state, _, _, _ = _run_agent(tmp_path)

    succeeded = [p for p in state.papers if p.analysis_status == "succeeded"]
    assert state.analysis_stats.unsupported_claims == len(succeeded)


def test_per_paper_failure_isolation(tmp_path: Path) -> None:
    """Task 3-06-14 — paper1 fails (malformed LLM response); paper2 still succeeds."""
    state, _, _, _ = _run_agent(
        tmp_path,
        override_responses={"per_paper_summary:paper1": "{malformed"},
    )

    p1 = next(p for p in state.papers if p.paper_id == "paper1")
    p2 = next(p for p in state.papers if p.paper_id == "paper2")
    assert p1.analysis_status == "failed"
    assert p2.analysis_status == "succeeded"


def test_failure_captured_in_analysis_stats(tmp_path: Path) -> None:
    """Task 3-06-15 — AnalysisStats.analysis_failures records (paper_id, reason)."""
    state, _, _, _ = _run_agent(
        tmp_path,
        override_responses={"per_paper_summary:paper1": "{malformed"},
    )

    failures = state.analysis_stats.analysis_failures
    assert len(failures) >= 1
    paper1_failure = next((f for f in failures if f[0] == "paper1"), None)
    assert paper1_failure is not None
    _, reason = paper1_failure
    assert reason, "failure reason should be non-empty"
