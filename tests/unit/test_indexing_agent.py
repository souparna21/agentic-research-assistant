"""Unit tests for IndexingAgent orchestration (IDX-04 dedup, IDX-08 integrity).

VALIDATION.md task IDs:
  3-04-01 → test_duplicate_chunks_deduplicated           (IDX-04)
  3-04-02 → test_ntotal_equals_unique_hashes_asserted    (IDX-08)
"""
from __future__ import annotations

from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _hf_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the cached MiniLM snapshot; do not ping HuggingFace Hub.

    Mirrors tests/unit/test_embed.py + test_vector_store.py. Lets the real
    embedder run under pytest-socket --disable-socket by suppressing the
    update-check network call HuggingFace Hub would otherwise attempt.
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _seed_paper_with_extracted_md(
    tmp_path: Path, paper_id: str, text: str, num_sections: int = 4
):
    """Write extracted.md under runs/<run_id>/papers/<paper_id>/ and build a Paper."""
    from ara.state import ExtractionMeta, Paper

    run_id = "rid"
    rel = f"{run_id}/papers/{paper_id}/extracted.md"
    full = tmp_path / "runs" / rel
    full.parent.mkdir(parents=True, exist_ok=True)
    full.write_text(text, encoding="utf-8")
    return Paper(
        paper_id=paper_id,
        title=f"Paper {paper_id}",
        extraction_ok=True,
        full_text_available=True,
        extracted_md_path=rel,
        extraction_meta=ExtractionMeta(num_detected_sections=num_sections, pages=4),
    )


def _state(tmp_path: Path, papers: list):
    from ara.state import PipelineState
    return PipelineState(
        run_id="rid",
        question="q?",
        config_snapshot={"model": "gemini/gemini-2.5-flash"},
        runs_dir=str(tmp_path / "runs"),
        papers=papers,
    )


def _long_section_body(n_paragraphs: int = 6, tokens_per_para: int = 55) -> str:
    word = "chunker "
    para = (word * tokens_per_para).strip() + "."
    return "\n\n".join(para for _ in range(n_paragraphs))


def test_duplicate_chunks_deduplicated(tmp_path: Path) -> None:
    """Task 3-04-01 — Identical extracted text across two papers dedupes via chunk_hash.

    Two papers carrying the same extracted markdown produce overlapping chunk_hash
    values. IndexingAgent drops duplicates before building FAISS;
    indexing_stats.duplicate_chunks_dropped > 0 and index.ntotal == unique_chunks.
    """
    from ara.agents.indexing import IndexingAgent

    shared_text = (
        "# Introduction\n\n" + _long_section_body(n_paragraphs=6)
        + "\n\n# Methods\n\n" + _long_section_body(n_paragraphs=6)
        + "\n\n# Results\n\n" + _long_section_body(n_paragraphs=6)
    )
    p1 = _seed_paper_with_extracted_md(tmp_path, "A", shared_text)
    p2 = _seed_paper_with_extracted_md(tmp_path, "B", shared_text)
    state = _state(tmp_path, [p1, p2])

    agent = IndexingAgent()
    state = agent.run(state)

    assert state.indexing_stats.duplicate_chunks_dropped > 0, (
        "two papers with identical markdown should produce duplicate chunk_hashes"
    )
    assert state.indexing_stats.unique_chunks == state.chunks_count
    assert state.indexing_stats.papers_chunked == 2
    assert state.index_path
    idx_path = Path(state.index_path)
    if not idx_path.is_absolute():
        idx_path = Path(state.runs_dir) / state.index_path
    assert idx_path.exists()


def test_ntotal_equals_unique_hashes_asserted(tmp_path: Path, monkeypatch) -> None:
    """Task 3-04-02 — IndexingAgent asserts index.ntotal == len(unique_hashes); mismatch raises.

    Simulate sidecar/index desync by monkeypatching FaissVectorStore.build_and_persist
    to return a wrong ntotal; IndexingAgent must raise IndexingIntegrityError.
    """
    from ara.agents.indexing import IndexingAgent, IndexingIntegrityError

    import ara.services.vector_store as vs_mod

    text = (
        "# A\n\n" + _long_section_body() + "\n\n"
        "# B\n\n" + _long_section_body() + "\n\n"
        "# C\n\n" + _long_section_body()
    )
    paper = _seed_paper_with_extracted_md(tmp_path, "X", text)
    state = _state(tmp_path, [paper])

    orig_build = vs_mod.FaissVectorStore.build_and_persist

    def broken_build(self, chunks, embeddings, index_path, chunks_path):
        orig_build(self, chunks, embeddings, index_path, chunks_path)
        return 9999

    monkeypatch.setattr(vs_mod.FaissVectorStore, "build_and_persist", broken_build)

    with pytest.raises(IndexingIntegrityError, match="ntotal|unique"):
        IndexingAgent().run(state)
