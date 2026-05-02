"""Unit tests for SectionAwareChunker (IDX-01..04).

VALIDATION.md task IDs:
  3-01-01 -> test_intra_section_overlap_is_fifty_tokens      (IDX-01)
  3-01-02 -> test_chunks_respect_target_token_size           (IDX-01)
  3-01-03 -> test_overlap_never_crosses_section_boundary     (IDX-01)
  3-01-04 -> test_section_aware_when_three_or_more_sections  (IDX-02)
  3-01-05 -> test_paragraph_fallback_when_few_sections       (IDX-02)
  3-01-06 -> test_fallback_never_splits_midparagraph         (IDX-02)
  3-01-07 -> test_chunk_metadata_schema                      (IDX-03)
  3-01-08 -> test_chunk_hash_stable_and_short                (IDX-03)
  3-01-09 -> test_duplicate_text_same_hash                   (IDX-04)
"""
from __future__ import annotations

import hashlib
import re

import pytest
import tiktoken

from ara.state import ExtractionMeta, Paper

ENC = tiktoken.get_encoding("cl100k_base")


def _tokens(text: str) -> int:
    return len(ENC.encode(text, disallowed_special=()))


def _long_section_body(n_paragraphs: int = 12, tokens_per_para: int = 60) -> str:
    """Synthesize a section body with n paragraphs, each ~tokens_per_para tokens."""
    word = "transformer "
    para_text = (word * tokens_per_para).strip() + "."
    return "\n\n".join(para_text for _ in range(n_paragraphs))


@pytest.fixture
def paper_with_sections() -> Paper:
    return Paper(
        paper_id="P1",
        title="Sample",
        extraction_meta=ExtractionMeta(num_detected_sections=4),
    )


@pytest.fixture
def paper_with_two_sections() -> Paper:
    return Paper(
        paper_id="P2",
        title="Few sections",
        extraction_meta=ExtractionMeta(num_detected_sections=2),
    )



def test_intra_section_overlap_is_fifty_tokens(paper_with_sections: Paper) -> None:
    """Task 3-01-01 - Adjacent chunks WITHIN A SECTION share a ~50-token tail/head overlap."""
    from ara.services.chunker import SectionAwareChunker

    md = (
        "# Introduction\n\n"
        + _long_section_body(n_paragraphs=20)
        + "\n\n# Methods\n\n"
        + _long_section_body(n_paragraphs=8)
        + "\n\n# Results\n\n short body\n\n# Discussion\n\n short\n"
    )
    chunks = SectionAwareChunker().chunk(paper_with_sections, md)
    intro_chunks = [c for c in chunks if c.section == "Introduction"]
    assert len(intro_chunks) >= 2, "introduction should span multiple chunks for overlap test"

    first_tail_ids = ENC.encode(intro_chunks[0].text, disallowed_special=())[-50:]
    first_tail_text = ENC.decode(first_tail_ids).strip()
    assert first_tail_text and first_tail_text[:20] in intro_chunks[1].text, (
        "second intra-section chunk missing overlap prefix from first chunk's tail"
    )


def test_chunks_respect_target_token_size(paper_with_sections: Paper) -> None:
    """Task 3-01-02 - No chunk exceeds ~500 tokens unless a single paragraph is oversized."""
    from ara.services.chunker import SectionAwareChunker

    md = "# Intro\n\n" + _long_section_body(n_paragraphs=30) + "\n\n# Methods\n\nshort\n"
    chunks = SectionAwareChunker().chunk(paper_with_sections, md)
    for c in chunks:
        assert c.token_count <= 600, f"chunk {c.chunk_index} has {c.token_count} tokens"


def test_overlap_never_crosses_section_boundary(paper_with_sections: Paper) -> None:
    """Task 3-01-03 - The first chunk of a new section does NOT contain text from the previous section."""
    from ara.services.chunker import SectionAwareChunker

    unique_marker_a = "ZZ_ALPHA_MARKER_XY"
    unique_marker_b = "ZZ_BETA_MARKER_XY"
    md = (
        f"# SectionA\n\n{_long_section_body(n_paragraphs=15)} {unique_marker_a}\n\n"
        f"# SectionB\n\n{unique_marker_b} " + _long_section_body(n_paragraphs=2) + "\n"
    )
    chunks = SectionAwareChunker().chunk(paper_with_sections, md)
    b_chunks = [c for c in chunks if c.section == "SectionB"]
    assert b_chunks, "Section B produced no chunks"
    for c in b_chunks:
        assert unique_marker_a not in c.text, (
            "SectionB chunk leaked SectionA marker: overlap crossed section boundary"
        )



def test_section_aware_when_three_or_more_sections(paper_with_sections: Paper) -> None:
    """Task 3-01-04 - num_detected_sections >= 3 -> section-aware path (chunks carry section names)."""
    from ara.services.chunker import SectionAwareChunker

    md = (
        "# A\n\n" + _long_section_body(n_paragraphs=3) + "\n\n"
        "# B\n\nBeta body text here.\n\n"
        "# C\n\nGamma body text here.\n"
    )
    chunks = SectionAwareChunker().chunk(paper_with_sections, md)
    assert chunks, "section-aware path emitted no chunks on a substantive body"
    sections_seen = {c.section for c in chunks}
    assert sections_seen.intersection({"A", "B", "C"}), (
        "section-aware path should preserve section titles on chunks"
    )


def test_paragraph_fallback_when_few_sections(paper_with_two_sections: Paper) -> None:
    """Task 3-01-05 - num_detected_sections < 3 -> paragraph fallback (section=None)."""
    from ara.services.chunker import SectionAwareChunker

    md = "# A\n\n" + _long_section_body(n_paragraphs=6) + "\n\n# B\n\n" + _long_section_body(n_paragraphs=6) + "\n"
    chunks = SectionAwareChunker().chunk(paper_with_two_sections, md)
    assert chunks, "fallback should still emit chunks"
    assert all(c.section is None for c in chunks), (
        "paragraph fallback must NOT attach section titles (section=None)"
    )


def test_fallback_never_splits_midparagraph(paper_with_two_sections: Paper) -> None:
    """Task 3-01-06 - Paragraph fallback emits whole paragraphs only; no mid-paragraph split."""
    from ara.services.chunker import SectionAwareChunker

    para_a = "Alpha " * 30 + "END_OF_ALPHA."
    para_b = "Beta " * 30 + "END_OF_BETA."
    md = para_a + "\n\n" + para_b
    paper = Paper(paper_id="P3", title="fb", extraction_meta=ExtractionMeta(num_detected_sections=0))
    chunks = SectionAwareChunker().chunk(paper, md)
    for c in chunks:
        if "Alpha " in c.text:
            assert "END_OF_ALPHA." in c.text, "Alpha paragraph was split mid-paragraph"
        if "Beta " in c.text:
            assert "END_OF_BETA." in c.text, "Beta paragraph was split mid-paragraph"



def test_chunk_metadata_schema(paper_with_sections: Paper) -> None:
    """Task 3-01-07 - Every chunk carries paper_id, page, section, chunk_index, chunk_hash, token_count."""
    from ara.services.chunker import Chunk, SectionAwareChunker

    md = "# Intro\n\n" + _long_section_body(n_paragraphs=5) + "\n"
    chunks = SectionAwareChunker().chunk(paper_with_sections, md)
    assert chunks
    for c in chunks:
        assert isinstance(c, Chunk)
        assert c.paper_id == "P1"
        assert c.section == "Intro" or c.section is None
        assert isinstance(c.chunk_index, int) and c.chunk_index >= 0
        assert isinstance(c.chunk_hash, str) and len(c.chunk_hash) == 16
        assert isinstance(c.token_count, int) and c.token_count > 0
        assert hasattr(c, "page")


def test_chunk_hash_stable_and_short(paper_with_sections: Paper) -> None:
    """Task 3-01-08 - chunk_hash = first 16 hex chars of sha256(text.strip())."""
    from ara.services.chunker import SectionAwareChunker

    md = "# S\n\n" + _long_section_body(n_paragraphs=2) + "\n"
    chunks = SectionAwareChunker().chunk(paper_with_sections, md)
    assert chunks, "expected at least one chunk from a 2-paragraph section"
    c = chunks[0]
    expected = hashlib.sha256(c.text.strip().encode("utf-8")).hexdigest()[:16]
    assert c.chunk_hash == expected
    assert re.fullmatch(r"[0-9a-f]{16}", c.chunk_hash)



def test_duplicate_text_same_hash(paper_with_sections: Paper) -> None:
    """Task 3-01-09 - Two chunks carrying identical text produce identical chunk_hash."""
    from ara.services.chunker import SectionAwareChunker

    md = (
        "# A\n\n" + _long_section_body(n_paragraphs=8) + "\n\n"
        + "# B\n\n" + _long_section_body(n_paragraphs=8) + "\n"
    )
    chunker = SectionAwareChunker()
    chunks_a = chunker.chunk(paper_with_sections, md)
    chunks_b = chunker.chunk(paper_with_sections, md)
    hashes_a = [c.chunk_hash for c in chunks_a]
    hashes_b = [c.chunk_hash for c in chunks_b]
    assert hashes_a == hashes_b
    unique_pairs = {(c.text.strip(), c.chunk_hash) for c in chunks_a}
    assert len({h for _, h in unique_pairs}) == len(unique_pairs)
