"""FaissVectorStore — build, persist, reload, and search over chunk embeddings.

IDX-06/07/08/09. Uses faiss.IndexFlatIP(384) — sub-millisecond exact search at 1k-5k
vectors. Lazy imports of faiss keep ``ara --help`` cold start unaffected (Phase-1
AST-scan rule in tests/unit/test_cli.py).

Metadata discipline: FAISS stores only the dense vectors. Chunk text + metadata
(paper_id / page / section / chunk_index / chunk_hash / token_count) live in a
sidecar JSON written via :func:`ara.persistence.atomic_write_text` (pitfall P7
defense — Ctrl-C mid-write never corrupts).

Sidecar/index desync at load raises RuntimeError (IDX-08 tripwire — matches the
"pitfall P16 duplicate-chunk retrieval-score inflation" defense at plan 03-04's
IndexingAgent layer).

Requirements covered (REQUIREMENTS.md):
  - IDX-06: ``faiss.IndexFlatIP(EMBEDDING_DIM)`` + ``index.add(embeddings)``; IP
    on unit vectors == cosine similarity.
  - IDX-07: ``faiss.write_index`` + atomic JSON sidecar; fresh-process
    :meth:`load` + :meth:`search` returns same top-k as pre-save.
  - IDX-08: sidecar ``len(chunks) != index.ntotal`` at load raises RuntimeError.
  - IDX-09: :meth:`search` returns ``list[RetrievedChunk]`` with score +
    metadata; safe when ``k > ntotal`` (filters the ``-1`` sentinel FAISS
    returns for empty slots).
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ara.persistence import atomic_write_text
from ara.services.chunker import Chunk
from ara.services.embed import EMBEDDING_DIM, MiniLMEmbedder


@dataclass(frozen=True)
class RetrievedChunk:
    """Return shape of :meth:`FaissVectorStore.search`.

    ``paper_id`` is the handle the analysis-prompt's ``[P<paper_id>-N]``
    citation format pulls from (plan 03-06 AnalysisAgent). ``score`` is the
    inner product on unit-norm vectors — numerically equivalent to cosine
    similarity, bounded in ``[-1, 1]`` (not ``[0, 1]``).
    """

    paper_id: str
    page: int | None
    section: str | None
    chunk_index: int
    chunk_hash: str
    text: str
    score: float


@dataclass
class FaissVectorStore:
    """Build/persist/reload/search a FAISS IndexFlatIP over chunk embeddings.

    Instances are stateful: :meth:`build_and_persist` or :meth:`load`
    populates ``_index`` + ``_chunks``; :meth:`search` reads both. One
    instance per run is sufficient — IndexingAgent (plan 03-04) will build
    then persist; AnalysisAgent (plan 03-06) loads then searches.
    """

    embedder: MiniLMEmbedder
    _index: object | None = field(default=None, init=False, repr=False)
    _chunks: list[Chunk] = field(default_factory=list, init=False, repr=False)

    def build_and_persist(
        self,
        chunks: list[Chunk],
        embeddings: np.ndarray,
        index_path: Path,
        chunks_path: Path,
    ) -> int:
        """Build IndexFlatIP, add vectors, persist index + sidecar. Returns ntotal.

        Pre-conditions (enforced by IndexingAgent in plan 03-04, not here):
          * ``chunks`` already deduplicated by ``chunk_hash``.
          * ``embeddings`` produced by
            :meth:`ara.services.embed.MiniLMEmbedder.encode` so the unit-norm
            invariant holds (IP == cosine).

        Raises:
          ValueError: ``embeddings.shape != (len(chunks), EMBEDDING_DIM)``.
          TypeError: ``embeddings.dtype != float32`` — FAISS's hard requirement.
        """
        import faiss  # type: ignore[import-untyped]  # lazy import — heavy dep, no stubs

        if embeddings.shape != (len(chunks), EMBEDDING_DIM):
            raise ValueError(
                f"Embedding shape {embeddings.shape} != expected "
                f"({len(chunks)}, {EMBEDDING_DIM})"
            )
        if embeddings.dtype != np.float32:
            raise TypeError(f"FAISS requires float32; got {embeddings.dtype}")

        self._index = faiss.IndexFlatIP(EMBEDDING_DIM)
        self._index.add(embeddings)  # type: ignore[attr-defined]
        self._chunks = list(chunks)

        index_path.parent.mkdir(parents=True, exist_ok=True)
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))

        sidecar = {
            "dim": EMBEDDING_DIM,
            "model": self.embedder.model_name,
            "ntotal": int(self._index.ntotal),  # type: ignore[attr-defined]
            "chunks": [
                {
                    "paper_id": c.paper_id,
                    "page": c.page,
                    "section": c.section,
                    "chunk_index": c.chunk_index,
                    "chunk_hash": c.chunk_hash,
                    "token_count": c.token_count,
                    "text": c.text,
                }
                for c in chunks
            ],
        }
        atomic_write_text(json.dumps(sidecar, indent=2), chunks_path)
        return int(self._index.ntotal)  # type: ignore[attr-defined]

    def load(self, index_path: Path, chunks_path: Path) -> None:
        """Load existing index + sidecar; raise on shape disagreement (IDX-08 tripwire).

        Raises:
          RuntimeError: ``len(sidecar["chunks"]) != index.ntotal`` — surfaces a
            corrupt or desynced persist (e.g. interrupted :meth:`build_and_persist`
            OR manual sidecar editing).
        """
        import faiss

        self._index = faiss.read_index(str(index_path))
        sidecar = json.loads(chunks_path.read_text(encoding="utf-8"))
        self._chunks = [
            Chunk(
                paper_id=d["paper_id"],
                page=d["page"],
                section=d["section"],
                chunk_index=d["chunk_index"],
                chunk_hash=d["chunk_hash"],
                token_count=d["token_count"],
                text=d["text"],
            )
            for d in sidecar["chunks"]
        ]
        if len(self._chunks) != int(self._index.ntotal):  # type: ignore[attr-defined]
            raise RuntimeError(
                f"chunks sidecar has {len(self._chunks)} entries but index "
                f"ntotal={self._index.ntotal} — corrupt or desynced persist"  # type: ignore[attr-defined]
            )

    def search(self, query_text: str, k: int = 5) -> list[RetrievedChunk]:
        """Encode the query, FAISS-search top-k, join metadata.

        Safe when ``k > ntotal``: FAISS pads the result with ``idx == -1``
        sentinel entries; we filter them so callers never see a spurious
        "-1"-indexed chunk. Result length <= min(k, ntotal).
        """
        if self._index is None or not self._chunks:
            raise RuntimeError("VectorStore not built or loaded yet.")
        q_vec = self.embedder.encode_query(query_text)
        scores, ids = self._index.search(q_vec, k)  # type: ignore[attr-defined]
        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            if idx < 0 or idx >= len(self._chunks):
                continue
            c = self._chunks[idx]
            out.append(
                RetrievedChunk(
                    paper_id=c.paper_id,
                    page=c.page,
                    section=c.section,
                    chunk_index=c.chunk_index,
                    chunk_hash=c.chunk_hash,
                    text=c.text,
                    score=float(score),
                )
            )
        return out
