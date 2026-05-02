"""Unit tests for MiniLMEmbedder (IDX-05).

VALIDATION.md task IDs:
  3-02-01 → test_encode_shape_and_dtype       (IDX-05)
  3-02-02 → test_encode_vectors_are_unit_norm (IDX-05)
  3-02-03 → test_encode_empty_returns_empty   (IDX-05)

Offline discipline: these tests load the real MiniLM model from the local
HuggingFace cache (~/.cache/huggingface/hub). To keep them zero-network (matching
the project-wide pytest-socket rule), we set HF_HUB_OFFLINE=1 + TRANSFORMERS_OFFLINE=1
via an autouse fixture — HF then serves the cached snapshot without phoning home.
First-time setup requires a one-shot online download; documented in README.
"""
from __future__ import annotations

import numpy as np
import pytest


@pytest.fixture(autouse=True)
def _hf_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Force HuggingFace to serve the cached model snapshot (no network pings)."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")


def test_encode_shape_and_dtype() -> None:
    """Task 3-02-01 — encode(texts) returns np.float32 array of shape (len(texts), 384)."""
    from ara.services.embed import EMBEDDING_DIM, MiniLMEmbedder

    embedder = MiniLMEmbedder()
    texts = ["retrieval augmented generation", "transformer attention mechanism", "knowledge graph"]
    vecs = embedder.encode(texts)

    assert isinstance(vecs, np.ndarray), f"expected np.ndarray, got {type(vecs)}"
    assert vecs.shape == (len(texts), EMBEDDING_DIM), (
        f"expected shape ({len(texts)}, {EMBEDDING_DIM}), got {vecs.shape}"
    )
    assert vecs.dtype == np.float32, f"expected float32, got {vecs.dtype}"
    assert EMBEDDING_DIM == 384


def test_encode_vectors_are_unit_norm() -> None:
    """Task 3-02-02 — every returned row has L2 norm ≈ 1.0 (normalize_embeddings=True applied).

    Without this property, FAISS IndexFlatIP ≠ cosine similarity — which contradicts
    the report.tex stack-decision claim. This is a load-bearing invariant.
    """
    from ara.services.embed import MiniLMEmbedder

    embedder = MiniLMEmbedder()
    vecs = embedder.encode([
        "attention is all you need",
        "contrastive learning for sentence embeddings",
    ])
    norms = np.linalg.norm(vecs, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-5), f"non-unit norms detected: {norms}"


def test_encode_empty_returns_empty() -> None:
    """Task 3-02-03 — encode([]) returns (0, 384) float32 array without crash or exception."""
    from ara.services.embed import EMBEDDING_DIM, MiniLMEmbedder

    embedder = MiniLMEmbedder()
    vecs = embedder.encode([])
    assert vecs.shape == (0, EMBEDDING_DIM)
    assert vecs.dtype == np.float32
