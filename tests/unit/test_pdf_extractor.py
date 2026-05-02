"""Unit tests for ``PdfExtractor`` (EXT-01..04).

EXT-02 uses the pre-committed ``adversarial_prompt_injection.pdf``; other
scenarios synthesize small PDFs inline so each test controls exactly
the text that should/shouldn't appear in the extracted markdown.

VALIDATION rows covered: 2-05-01..2-05-08.
"""
from __future__ import annotations

import re as _re
from pathlib import Path

import pymupdf
import pytest

from ara.services.pdf import ExtractionResult, ExtractionTooShort, PdfExtractor

REPO_ROOT = Path(__file__).resolve().parents[2]
FIXTURES = REPO_ROOT / "tests" / "fixtures" / "papers"


def _write_pdf(tmp_path: Path, name: str, builder) -> Path:
    """Helper: build a PDF in ``tmp_path/name`` via a callback taking a ``pymupdf.Document``."""
    doc = pymupdf.open()
    builder(doc)
    path = tmp_path / name
    doc.save(str(path))
    doc.close()
    return path


_DEFAULT_BODY = (
    "Large language models show strong performance on many tasks but "
    "suffer from hallucination; retrieval-augmented generation grounds "
    "outputs in external evidence to mitigate this failure mode. "
    "We evaluate several retrieval configurations across standard "
    "benchmarks and report detailed error analyses in later sections. "
    "This paragraph is padded with sufficient words to ensure the "
    "extracted text comfortably exceeds the two-hundred character "
    "minimum budget enforced by the PdfExtractor min_output_chars gate."
)


def _insert_body(
    page: pymupdf.Page,
    x: float = 72,
    y: float = 72,
    text: str | None = None,
    width: float = 470,
    height: float = 500,
) -> None:
    """Insert >= 200 chars of normal body so extraction doesn't hit the min_output_chars floor.

    Uses ``insert_textbox`` so long paragraphs wrap inside the printable
    area rather than overflow the page boundary (which ``insert_text``
    silently does, dropping most of the content).
    """
    rect = pymupdf.Rect(x, y, x + width, y + height)
    page.insert_textbox(
        rect,
        text if text is not None else _DEFAULT_BODY,
        fontsize=11,
        color=(0, 0, 0),
    )




def test_extracts_markdown() -> None:
    """EXT-01: baseline PDF -> ExtractionResult with non-empty markdown."""
    extractor = PdfExtractor()
    res = extractor.extract(FIXTURES / "paper_with_pdf.pdf")
    assert isinstance(res, ExtractionResult)
    assert len(res.markdown) >= 50, res.markdown
    assert res.pages >= 1




def test_extraction_strips_hidden_text_attack() -> None:
    """EXT-02 CRITICAL: adversarial fixture -> 'PWN' is NOT in extracted markdown."""
    extractor = PdfExtractor()
    res = extractor.extract(FIXTURES / "adversarial_prompt_injection.pdf")
    assert "PWN" not in res.markdown, (
        "HIDDEN-TEXT DEFENSE FAILED: 'PWN' leaked into extracted markdown:\n"
        f"{res.markdown[:500]}"
    )
    assert "IGNORE PREVIOUS INSTRUCTIONS" not in res.markdown


def test_hidden_span_counter_incremented() -> None:
    """EXT-02: hidden_text_spans_stripped > 0 on the adversarial fixture."""
    extractor = PdfExtractor()
    res = extractor.extract(FIXTURES / "adversarial_prompt_injection.pdf")
    assert res.hidden_text_spans_stripped > 0, (
        f"expected > 0 stripped spans, got {res.hidden_text_spans_stripped}"
    )


def test_tiny_font_span_redacted(tmp_path: Path) -> None:
    """EXT-02: a span smaller than min_font_size_pt (0.5pt default) is removed."""

    def build(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        _insert_body(page)
        page.insert_text((72, 500), "TINY_INJECTION", fontsize=0.3, color=(0, 0, 0))

    pdf = _write_pdf(tmp_path, "tiny.pdf", build)
    res = PdfExtractor().extract(pdf)
    assert "TINY_INJECTION" not in res.markdown
    assert res.hidden_text_spans_stripped >= 1


def test_off_mediabox_span_redacted(tmp_path: Path) -> None:
    """EXT-02: span whose bbox falls entirely outside the page mediabox is removed.

    Default pymupdf new_page() returns A4 (595x842 pt); placing text at
    x=2000 puts it well beyond the mediabox intersection.
    """

    def build(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        _insert_body(page)
        page.insert_text((2000, 100), "OFFPAGE_INJECTION", fontsize=11, color=(0, 0, 0))

    pdf = _write_pdf(tmp_path, "offpage.pdf", build)
    res = PdfExtractor().extract(pdf)
    assert "OFFPAGE_INJECTION" not in res.markdown




def test_header_footer_scrubbed(tmp_path: Path) -> None:
    """EXT-03: a line repeating on >= 3 pages is stripped."""

    def build(doc: pymupdf.Document) -> None:
        for i in range(4):
            page = doc.new_page()
            page.insert_text(
                (72, 50), "Journal of Unlikely Results, Vol. 42", fontsize=9, color=(0, 0, 0)
            )
            _insert_body(
                page,
                y=120,
                text=(
                    f"Page {i} unique body paragraph content here with enough filler to "
                    f"exceed the 200 character minimum output budget and avoid the "
                    f"ExtractionTooShort gate for small PDFs.\n\n"
                ),
            )
            page.insert_text(
                (72, 780), "Journal of Unlikely Results, Vol. 42", fontsize=9, color=(0, 0, 0)
            )

    pdf = _write_pdf(tmp_path, "hdrftr.pdf", build)
    res = PdfExtractor().extract(pdf)
    assert "Journal of Unlikely Results" not in res.markdown, res.markdown
    assert res.header_footer_lines_stripped >= 1


def test_page_numbers_stripped(tmp_path: Path) -> None:
    """EXT-03: standalone footer-marker lines repeating across pages are stripped.

    We use a distinctive string marker instead of bare page numbers because
    pymupdf4llm's heuristic may not echo standalone digits as their own line —
    the CONTRACT is that ANY line repeating >= 3 times in the header/footer
    region gets dropped, which is exactly what a page-number line degrades
    into once the digits are rendered.
    """

    def build(doc: pymupdf.Document) -> None:
        for i in range(4):
            page = doc.new_page()
            _insert_body(
                page,
                y=120,
                text=(
                    f"## Section {i}\n\n"
                    f"Unique body text for page {i}. We discuss diverse topics "
                    f"and expand on methodology choices at length to ensure the "
                    f"character budget is generous enough for extraction.\n\n"
                ),
            )
            page.insert_text((300, 780), "PAGE_FOOTER_MARKER", fontsize=9, color=(0, 0, 0))

    pdf = _write_pdf(tmp_path, "pageno.pdf", build)
    res = PdfExtractor().extract(pdf)
    assert "PAGE_FOOTER_MARKER" not in res.markdown, res.markdown




def test_num_sections_counts_markdown_headings(tmp_path: Path) -> None:
    """EXT-04: num_detected_sections counts ^#{1,3}\\s+ occurrences consistently."""

    def build(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        page.insert_text((72, 60), "Introduction", fontsize=18, color=(0, 0, 0))
        page.insert_textbox(
            pymupdf.Rect(72, 100, 540, 230),
            "Paragraph of body text explaining context. " * 5,
            fontsize=11,
            color=(0, 0, 0),
        )
        page.insert_text((72, 260), "Methods", fontsize=18, color=(0, 0, 0))
        page.insert_textbox(
            pymupdf.Rect(72, 300, 540, 470),
            "Paragraph about methodology. " * 10,
            fontsize=11,
            color=(0, 0, 0),
        )
        page.insert_text((72, 500), "Results", fontsize=18, color=(0, 0, 0))
        page.insert_textbox(
            pymupdf.Rect(72, 540, 540, 770),
            "Paragraph about findings. " * 10,
            fontsize=11,
            color=(0, 0, 0),
        )

    pdf = _write_pdf(tmp_path, "sections.pdf", build)
    res = PdfExtractor().extract(pdf)
    in_md = len(_re.findall(r"^#{1,3}\s+", res.markdown, flags=_re.MULTILINE))
    assert res.num_detected_sections == in_md, (
        f"num_detected_sections ({res.num_detected_sections}) must equal "
        f"the ^#{{1,3}} count in markdown ({in_md}). Markdown:\n{res.markdown[:1000]}"
    )




def test_short_output_raises_extraction_too_short(tmp_path: Path) -> None:
    """A PDF with less than min_output_chars text raises ExtractionTooShort.

    ExtractionAgent (plan 02-06) catches this per-paper to set
    fallback_to_abstract without halting the stage (pitfall P8).
    """

    def build(doc: pymupdf.Document) -> None:
        page = doc.new_page()
        page.insert_text((72, 72), "tiny", fontsize=11)

    pdf = _write_pdf(tmp_path, "short.pdf", build)
    with pytest.raises(ExtractionTooShort):
        PdfExtractor().extract(pdf)
