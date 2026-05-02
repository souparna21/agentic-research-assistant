"""Unit tests for FaissVectorStore (IDX-06..09).

VALIDATION.md task IDs:
  3-03-01 -> test_build_uses_indexflatip                    (IDX-06)
  3-03-02 -> test_ntotal_matches_embeddings                 (IDX-06)
  3-03-03 -> test_write_and_read_index_roundtrip            (IDX-07)
  3-03-04 -> test_chunks_sidecar_written_atomically         (IDX-07)
  3-03-05 -> test_vector_store_roundtrip_returns_same_top_k (IDX-07)
  3-03-06 -> test_sidecar_mismatch_raises_on_load           (IDX-08)
  3-03-07 -> test_search_returns_retrieved_chunk_with_metadata (IDX-09)
  3-03-08 -> test_search_k_greater_than_ntotal_is_safe      (IDX-09)
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _hf_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Serve the cached MiniLM snapshot without opening a socket.

    Mirror of tests/unit/test_embed.py — pytest-socket --disable-socket would
    otherwise fire on HF Hub's update-check ping. Model must already be in
    ~/.cache/huggingface/hub (first fresh-machine run populates the cache).
    """
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def _synth_chunks() -> list:
    from ara.services.chunker import Chunk
    return [
        Chunk(paper_id="P1", page=1, section="Methods", chunk_index=0,
              chunk_hash="h1h1h1h1h1h1h1h1", token_count=42,
              text="We train a transformer on RAG tasks."),
        Chunk(paper_id="P1", page=2, section="Results", chunk_index=1,
              chunk_hash="h2h2h2h2h2h2h2h2", token_count=48,
              text="Accuracy is 0.93 on the benchmark test set."),
        Chunk(paper_id="P2", page=1, section="Methods", chunk_index=0,
              chunk_hash="h3h3h3h3h3h3h3h3", token_count=44,
              text="An LSTM baseline with dropout regularization."),
    ]



def test_build_uses_indexflatip(tmp_path: Path) -> None:
    """Task 3-03-01 — build_and_persist creates a faiss.IndexFlatIP(384)."""
    import faiss

    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore

    chunks = _synth_chunks()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])
    store = FaissVectorStore(embedder=embedder)
    store.build_and_persist(
        chunks, embeddings,
        index_path=tmp_path / "faiss.index",
        chunks_path=tmp_path / "chunks.json",
    )
    reloaded = faiss.read_index(str(tmp_path / "faiss.index"))
    assert isinstance(reloaded, faiss.IndexFlatIP), (
        f"expected IndexFlatIP, got {type(reloaded)}"
    )
    assert reloaded.d == 384


def test_ntotal_matches_embeddings(tmp_path: Path) -> None:
    """Task 3-03-02 — index.ntotal == len(embeddings) after build."""
    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore

    chunks = _synth_chunks()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])
    store = FaissVectorStore(embedder=embedder)
    ntotal = store.build_and_persist(
        chunks, embeddings,
        index_path=tmp_path / "faiss.index",
        chunks_path=tmp_path / "chunks.json",
    )
    assert ntotal == len(embeddings) == len(chunks)



def test_write_and_read_index_roundtrip(tmp_path: Path) -> None:
    """Task 3-03-03 — write_index produces a file read_index restores to same ntotal."""
    import faiss

    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore

    chunks = _synth_chunks()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])
    store = FaissVectorStore(embedder=embedder)
    store.build_and_persist(
        chunks, embeddings,
        index_path=tmp_path / "faiss.index",
        chunks_path=tmp_path / "chunks.json",
    )
    idx = faiss.read_index(str(tmp_path / "faiss.index"))
    assert idx.ntotal == len(chunks)


def test_chunks_sidecar_written_atomically(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Task 3-03-04 — chunks.json written via atomic_write_text (observable via spy)."""
    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore

    calls: list[tuple[str, str]] = []

    import ara.persistence as pers_mod
    import ara.services.vector_store as vs_mod
    from ara.persistence import atomic_write_text as real_atomic

    def spy(text: str, path: Path) -> None:
        calls.append((str(path), text[:40]))
        real_atomic(text, path)

    monkeypatch.setattr(pers_mod, "atomic_write_text", spy)
    if hasattr(vs_mod, "atomic_write_text"):
        monkeypatch.setattr(vs_mod, "atomic_write_text", spy)

    chunks = _synth_chunks()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])
    store = FaissVectorStore(embedder=embedder)
    store.build_and_persist(
        chunks, embeddings,
        index_path=tmp_path / "faiss.index",
        chunks_path=tmp_path / "chunks.json",
    )

    chunks_path_str = str(tmp_path / "chunks.json")
    assert any(p == chunks_path_str for p, _ in calls), (
        "sidecar chunks.json was not written via atomic_write_text "
        "— partial-write safety lost"
    )
    data = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    assert data["dim"] == 384
    assert data["ntotal"] == len(chunks)
    assert len(data["chunks"]) == len(chunks)


def test_vector_store_roundtrip_returns_same_top_k(tmp_path: Path) -> None:
    """Task 3-03-05 — fresh-process reload + search(q, k=2) returns same top-k as pre-save."""
    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore

    chunks = _synth_chunks()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])

    store_build = FaissVectorStore(embedder=embedder)
    store_build.build_and_persist(
        chunks, embeddings,
        index_path=tmp_path / "faiss.index",
        chunks_path=tmp_path / "chunks.json",
    )

    store_load = FaissVectorStore(embedder=MiniLMEmbedder())
    store_load.load(tmp_path / "faiss.index", tmp_path / "chunks.json")

    q = "what methodology did paper P1 use?"
    pre = store_build.search(q, k=2)
    post = store_load.search(q, k=2)
    assert [c.chunk_hash for c in pre] == [c.chunk_hash for c in post]



def test_sidecar_mismatch_raises_on_load(tmp_path: Path) -> None:
    """Task 3-03-06 — corrupting chunks.json (remove entries) causes load() to raise."""
    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore

    chunks = _synth_chunks()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])
    FaissVectorStore(embedder=embedder).build_and_persist(
        chunks, embeddings,
        index_path=tmp_path / "faiss.index",
        chunks_path=tmp_path / "chunks.json",
    )

    data = json.loads((tmp_path / "chunks.json").read_text(encoding="utf-8"))
    data["chunks"] = data["chunks"][:-1]
    (tmp_path / "chunks.json").write_text(json.dumps(data), encoding="utf-8")

    store = FaissVectorStore(embedder=MiniLMEmbedder())
    with pytest.raises(RuntimeError, match="ntotal|chunks"):
        store.load(tmp_path / "faiss.index", tmp_path / "chunks.json")



def test_search_returns_retrieved_chunk_with_metadata(tmp_path: Path) -> None:
    """Task 3-03-07 — search returns list[RetrievedChunk] with score + metadata."""
    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore, RetrievedChunk

    chunks = _synth_chunks()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])
    store = FaissVectorStore(embedder=embedder)
    store.build_and_persist(
        chunks, embeddings,
        index_path=tmp_path / "faiss.index",
        chunks_path=tmp_path / "chunks.json",
    )

    results = store.search("transformer training methodology", k=2)
    assert len(results) == 2
    assert all(isinstance(r, RetrievedChunk) for r in results)
    r = results[0]
    assert r.paper_id in {"P1", "P2"}
    assert isinstance(r.score, float)
    assert r.chunk_hash.startswith("h")
    assert r.text
    assert -1.0 - 1e-5 <= r.score <= 1.0 + 1e-5


def test_search_k_greater_than_ntotal_is_safe(tmp_path: Path) -> None:
    """Task 3-03-08 — k > ntotal silently returns ntotal results (no FAISS -1 leakage)."""
    from ara.services.embed import MiniLMEmbedder
    from ara.services.vector_store import FaissVectorStore

    chunks = _synth_chunks()
    embedder = MiniLMEmbedder()
    embeddings = embedder.encode([c.text for c in chunks])
    store = FaissVectorStore(embedder=embedder)
    store.build_and_persist(
        chunks, embeddings,
        index_path=tmp_path / "faiss.index",
        chunks_path=tmp_path / "chunks.json",
    )

    results = store.search("irrelevant query text", k=10)
    assert len(results) == 3, f"expected 3 (ntotal), got {len(results)}"
    for r in results:
        assert r.chunk_hash
