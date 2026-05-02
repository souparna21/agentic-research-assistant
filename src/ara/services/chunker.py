"""Section-aware chunker with tiktoken cl100k_base + paragraph-boundary fallback.

Feeds the Phase-3 IndexingAgent (plan 03-04) with ~500-token chunks carrying
50-token intra-section overlap. Never crosses section boundaries. Section-aware
path gates on ``paper.extraction_meta.num_detected_sections >= 3`` (Phase 2
EXT-04 contract); otherwise falls back to paragraph-boundary chunking that
never splits mid-paragraph.

Requirements covered (REQUIREMENTS.md):
  - IDX-01: ~500-token chunks + 50-token intra-section overlap (never crosses sections)
  - IDX-02: paragraph-boundary fallback when ``num_detected_sections < 3``
  - IDX-03: chunk metadata schema (paper_id, page, section, chunk_index,
    chunk_hash, token_count, text)
  - IDX-04: ``chunk_hash = sha256(text.strip())[:16]`` — stable dedup key across runs

Module-level constants (TARGET_TOKENS / OVERLAP_TOKENS / MIN_TOKENS) and the
cl100k_base encoder (``_ENCODING``) are loaded once at import-time and reused
across all calls — the encoder initialization is the expensive part.

Pure CPU token math. No LLM calls. No network. Safe under pytest-socket.
"""
from __future__ import annotations

import hashlib
import re
from collections.abc import Iterator
from dataclasses import dataclass

import tiktoken

from ara.state import ExtractionMeta, Paper

_SECTION_RE = re.compile(r"^(#{1,3})\s+(.+?)\s*$", re.MULTILINE)
_ENCODING = tiktoken.get_encoding("cl100k_base")

TARGET_TOKENS = 500
OVERLAP_TOKENS = 50
MIN_TOKENS = 50


@dataclass(frozen=True)
class Chunk:
    """Single chunk record. Consumed by IndexingAgent (03-04) -> VectorStore (03-03).

    ``chunk_hash`` is the first 16 hex chars of ``sha256(text.strip())`` —
    enables corpus-level dedup in 03-04 (``assert index.ntotal == len(unique_hashes)``).
    ``page`` is None in v1; per-page mapping requires ``pymupdf4llm page_chunks=True``
    (deferred to Phase 4 hardening per 03-RESEARCH.md Open Question #1).
    """

    paper_id: str
    page: int | None
    section: str | None
    chunk_index: int
    chunk_hash: str
    token_count: int
    text: str


def _count_tokens(text: str) -> int:
    return len(_ENCODING.encode(text, disallowed_special=()))


def _encode(text: str) -> list[int]:
    return _ENCODING.encode(text, disallowed_special=())


def _decode(ids: list[int]) -> str:
    return _ENCODING.decode(ids)


def _split_into_sections(markdown: str) -> list[tuple[str | None, str]]:
    """Return list of (section_title, section_body). section_title is None for preamble."""
    sections: list[tuple[str | None, str]] = []
    matches = list(_SECTION_RE.finditer(markdown))
    if not matches:
        return [(None, markdown)]
    if matches[0].start() > 0:
        sections.append((None, markdown[: matches[0].start()]))
    for i, m in enumerate(matches):
        title = m.group(2).strip()
        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(markdown)
        sections.append((title, markdown[body_start:body_end]))
    return sections


def _sliding_window_over_paragraphs(
    paragraphs: list[str],
    target_tokens: int = TARGET_TOKENS,
    overlap_tokens: int = OVERLAP_TOKENS,
) -> Iterator[str]:
    """Accumulate paragraphs until target_tokens reached; emit chunk; advance with overlap.

    Overlap is applied by re-including the tail tokens (decoded back to text) of the
    previous chunk at the head of the next. Paragraphs are NEVER split mid-paragraph —
    if a single paragraph exceeds target_tokens, it becomes its own (oversized) chunk.
    """
    if not paragraphs:
        return
    buffer: list[str] = []
    buffer_tokens = 0
    for para in paragraphs:
        p_tokens = _count_tokens(para)
        if buffer_tokens + p_tokens > target_tokens and buffer:
            chunk_text = "\n\n".join(buffer).strip()
            yield chunk_text
            tail_ids = _encode(chunk_text)[-overlap_tokens:]
            overlap_text = _decode(tail_ids).strip()
            buffer = [overlap_text, para] if overlap_text else [para]
            buffer_tokens = _count_tokens("\n\n".join(buffer))
        else:
            buffer.append(para)
            buffer_tokens += p_tokens
    if buffer:
        chunk_text = "\n\n".join(buffer).strip()
        yield chunk_text


class SectionAwareChunker:
    """Stateless chunker. Call ``.chunk(paper, markdown)`` per paper.

    Gate: when ``paper.extraction_meta.num_detected_sections >= 3`` use the
    section-aware path (split on ``^#{1,3}`` headings, chunk within each section);
    otherwise use the paragraph-boundary fallback (single pseudo-section containing
    the whole document, paragraphs accumulated up to TARGET_TOKENS).

    The 50-token overlap lives WITHIN a section — the outer loop resets
    ``buffer`` across sections so overlap never leaks section-A text into
    section-B chunks.
    """

    def chunk(self, paper: Paper, markdown: str) -> list[Chunk]:
        meta: ExtractionMeta | None = paper.extraction_meta
        use_sections = bool(meta and meta.num_detected_sections >= 3)
        sections = _split_into_sections(markdown) if use_sections else [(None, markdown)]

        out: list[Chunk] = []
        chunk_idx = 0
        for section_title, body in sections:
            paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
            if not paragraphs:
                continue
            for chunk_text in _sliding_window_over_paragraphs(paragraphs):
                token_count = _count_tokens(chunk_text)
                if token_count < MIN_TOKENS:
                    continue
                chunk_hash = hashlib.sha256(chunk_text.strip().encode("utf-8")).hexdigest()[:16]
                out.append(
                    Chunk(
                        paper_id=paper.paper_id,
                        page=None,
                        section=section_title,
                        chunk_index=chunk_idx,
                        chunk_hash=chunk_hash,
                        token_count=token_count,
                        text=chunk_text,
                    )
                )
                chunk_idx += 1
        return out
