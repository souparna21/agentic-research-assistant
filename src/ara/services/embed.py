"""MiniLM embedder — lazy-loaded wrapper around sentence-transformers.

IDX-05: Embed chunks with all-MiniLM-L6-v2 (384-dim) using normalize_embeddings=True.
normalize_embeddings=True is MANDATORY — it makes FAISS IndexFlatIP equal cosine
similarity (the stack-decision invariant). Without it, IP != cosine, which silently
breaks the report.tex claim.

Lazy import: `from sentence_transformers import SentenceTransformer` lives INSIDE
_lazy_load() so `ara --help` doesn't pay the torch-import cost. Phase 1 locked this
via the AST-scan test test_cli_has_no_heavy_top_level_imports.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
EMBEDDING_DIM = 384


@dataclass
class MiniLMEmbedder:
    """Lazy-loaded SentenceTransformer. Encodes batched, normalize-on-encode."""

    model_name: str = EMBEDDING_MODEL_NAME
    _model: object | None = field(default=None, init=False, repr=False)

    def _lazy_load(self) -> object:
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(self, texts: list[str], *, batch_size: int = 32) -> np.ndarray:
        """Return normalized float32 embeddings of shape (len(texts), 384)."""
        if not texts:
            return np.empty((0, EMBEDDING_DIM), dtype=np.float32)
        model = self._lazy_load()
        vecs = model.encode(                    # type: ignore[attr-defined]
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
            batch_size=batch_size,
        )
        arr: np.ndarray = vecs.astype(np.float32, copy=False)
        return arr

    def encode_query(self, query: str) -> np.ndarray:
        """Return a single (1, 384) query vector, normalized."""
        return self.encode([query])
