"""IndexingAgent — chunks every paper, dedupes, embeds, builds persistent FAISS index.

IDX-01..09 orchestration layer. The chunker/embedder/vector_store services own the
heavy lifting (plans 03-01/02/03). This agent:

1. Iterates papers with extracted_md_path set and extraction_ok=True.
2. Reads extracted.md (relative to state.runs_dir), chunks via SectionAwareChunker,
   accumulates chunks.
3. Dedupes by chunk_hash across the corpus (IDX-04).
4. Batched encode via MiniLMEmbedder.encode (single call for all unique chunks).
5. FaissVectorStore.build_and_persist → runs/<run_id>/index/faiss.index + chunks.json.
6. Asserts ntotal == len(unique_hashes) (IDX-08 tripwire) — raises IndexingIntegrityError.
7. Writes IndexingStats + chunks_count + index_path + index_meta to PipelineState.

Per-paper failure isolation (P8 defense): narrow try/except (OSError, UnicodeDecodeError,
ValueError) around each paper's chunking call. One paper's failure appends an entry
to the returned StageReport.warnings and continues. Bare ``except Exception`` is
explicitly REJECTED — fail-loud on unknown bugs (same discipline as Phase 2
ExtractionAgent).

Exception list justifications:
  - OSError: extracted.md missing, unreadable, or filesystem I/O failure at read.
  - UnicodeDecodeError: extracted.md corrupted (non-UTF-8 bytes) — Phase-2
    atomic_write_text writes UTF-8, but defensive catch covers external edits.
  - ValueError: chunker raises on malformed Paper shape (e.g. ExtractionMeta
    attribute access on None) or tiktoken errors on pathological input.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import structlog

from ara.services.chunker import Chunk, SectionAwareChunker
from ara.services.embed import MiniLMEmbedder
from ara.services.vector_store import FaissVectorStore
from ara.state import IndexingStats, PipelineState, StageReport

log = structlog.get_logger(__name__)


class IndexingIntegrityError(RuntimeError):
    """Raised when index.ntotal != len(unique_hashes) after build (IDX-08 tripwire)."""


@dataclass
class IndexingAgent:
    """Stage Protocol conformant; structural only — @runtime_checkable Stage."""

    name: str = "indexing"
    chunker: SectionAwareChunker | None = None
    embedder: MiniLMEmbedder | None = None

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "index" / "faiss.index"

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.perf_counter()
        chunker = self.chunker or SectionAwareChunker()
        embedder = self.embedder or MiniLMEmbedder()

        all_chunks: list[Chunk] = []
        papers_chunked = 0
        per_paper_failures: list[tuple[str, str]] = []

        for paper in state.papers:
            if not paper.extraction_ok or not paper.extracted_md_path:
                continue
            md_path = Path(paper.extracted_md_path)
            if not md_path.is_absolute():
                md_path = Path(state.runs_dir) / paper.extracted_md_path
            try:
                markdown = md_path.read_text(encoding="utf-8")
                chunks = chunker.chunk(paper, markdown)
            except (OSError, UnicodeDecodeError, ValueError) as exc:
                per_paper_failures.append((paper.paper_id, repr(exc)))
                log.warning(
                    "indexing_paper_failed", paper_id=paper.paper_id, reason=repr(exc)
                )
                continue
            all_chunks.extend(chunks)
            papers_chunked += 1

        seen_hashes: set[str] = set()
        unique_chunks: list[Chunk] = []
        for c in all_chunks:
            if c.chunk_hash in seen_hashes:
                continue
            seen_hashes.add(c.chunk_hash)
            unique_chunks.append(c)
        duplicate_chunks_dropped = len(all_chunks) - len(unique_chunks)

        embed_t0 = time.perf_counter()
        embeddings = embedder.encode([c.text for c in unique_chunks])
        embedding_duration_ms = int((time.perf_counter() - embed_t0) * 1000)

        index_path = self.output_artifact_path(state)
        chunks_path = index_path.parent / "chunks.json"

        build_t0 = time.perf_counter()
        store = FaissVectorStore(embedder=embedder)
        ntotal = store.build_and_persist(
            unique_chunks, embeddings, index_path=index_path, chunks_path=chunks_path
        )
        index_build_duration_ms = int((time.perf_counter() - build_t0) * 1000)

        if ntotal != len(seen_hashes):
            raise IndexingIntegrityError(
                f"index.ntotal={ntotal} != len(unique_hashes)={len(seen_hashes)} — "
                f"FAISS-sidecar desync detected at build time"
            )

        try:
            index_bytes = int(index_path.stat().st_size) if index_path.exists() else 0
        except OSError:
            index_bytes = 0

        state.chunks_count = len(unique_chunks)
        try:
            state.index_path = str(index_path.relative_to(Path(state.runs_dir)))
        except ValueError:
            state.index_path = str(index_path)
        state.index_meta = {
            "dim": 384,
            "model": embedder.model_name,
            "built_at": "GOLDEN-TIMESTAMP",
            "cache_key": f"{embedder.model_name}:{len(unique_chunks)}",
        }
        state.indexing_stats = IndexingStats(
            papers_chunked=papers_chunked,
            total_chunks=len(all_chunks),
            unique_chunks=len(unique_chunks),
            duplicate_chunks_dropped=duplicate_chunks_dropped,
            embedding_duration_ms=embedding_duration_ms,
            index_build_duration_ms=index_build_duration_ms,
            index_bytes=index_bytes,
        )

        duration_ms = (time.perf_counter() - t0) * 1000.0
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=papers_chunked,
                outputs_count=len(unique_chunks),
                errors_count=len(per_paper_failures),
                duration_ms=duration_ms,
                warnings=[f"{pid}: {r}" for pid, r in per_paper_failures],
            )
        )
        return state
