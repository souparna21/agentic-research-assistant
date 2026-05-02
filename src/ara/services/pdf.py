"""PDF extraction service — hidden-text defense + header/footer scrub + section count.

Pipeline per PDF:
  1. ``pymupdf.open(pdf_path)`` — load Document
  2. Iterate pages; for each span, check (size < 0.5pt | white-on-white
     within epsilon=0.05 | zero-area bbox | bbox outside mediabox) -> redact
     via ``page.add_redact_annot + page.apply_redactions`` (EXT-02 / P6 defense)
  3. ``pymupdf4llm.to_markdown(doc, page_chunks=True, fontsize_limit=3)`` —
     generates per-page Markdown with preserved headings
  4. Scrub running headers/footers — lines in first-2 / last-2 per page that
     repeat (SequenceMatcher.ratio >= 0.9) across >= 3 pages are dropped
     (EXT-03)
  5. Count ``^#{1,3}\\s+`` in final markdown -> ``num_detected_sections``
     (EXT-04)
  6. If final markdown < ``min_output_chars``: raise ``ExtractionTooShort``
     (``ExtractionAgent`` in plan 02-06 catches and flags
     ``fallback_to_abstract``; single-paper failure never halts the stage,
     per pitfall P8 defense).

Belt-and-suspenders: ``pymupdf4llm``'s ``fontsize_limit=3`` default also strips
tiny text. Our 0.5pt check is stricter and runs first; the library's filter is
a safety net in case a previously-unseen injection vector slips past us.

This module is a pure stateless service. It takes a ``Path`` and returns an
``ExtractionResult``; it never touches ``PipelineState``. ``ExtractionAgent``
(plan 02-06) is the orchestration wrapper that handles per-paper failure
isolation and persistence.
"""
from __future__ import annotations

import difflib
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import pymupdf
import pymupdf4llm

log = logging.getLogger(__name__)

_HEADING_RE = re.compile(r"^#{1,3}\s+", re.MULTILINE)


class ExtractionTooShort(Exception):
    """Raised when ``pymupdf4llm`` produced less than ``min_output_chars`` of text.

    Callers (``ExtractionAgent`` in plan 02-06) catch this per-paper and set
    ``Paper.fallback_to_abstract=True`` so the stage never halts on one bad
    paper (pitfall P8 defense).
    """


@dataclass(frozen=True)
class ExtractionResult:
    """Stable boundary for ``ExtractionAgent`` + tests."""

    markdown: str
    num_detected_sections: int
    hidden_text_spans_stripped: int
    header_footer_lines_stripped: int
    pages: int


@dataclass
class PdfExtractor:
    """Stateless extractor. One instance may be reused across all papers in a stage."""

    min_font_size_pt: float = 0.5
    white_epsilon: float = 0.05
    min_output_chars: int = 200
    max_header_footer_chars: int = 120

    def extract(self, pdf_path: Path) -> ExtractionResult:
        doc = pymupdf.open(str(pdf_path))
        try:
            hidden_stripped = self._redact_adversarial_spans(doc)
            chunks = pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                fontsize_limit=3,
            )
            if isinstance(chunks, str):
                chunks = [{"text": chunks, "metadata": {}}]
            markdown, header_footer_stripped = self._scrub_headers_footers(chunks)
            num_sections = len(_HEADING_RE.findall(markdown))
            pages = len(doc)
        finally:
            doc.close()

        if len(markdown) < self.min_output_chars:
            raise ExtractionTooShort(
                f"Extracted only {len(markdown)} chars from {pdf_path.name}; "
                f"minimum {self.min_output_chars}."
            )

        return ExtractionResult(
            markdown=markdown,
            num_detected_sections=num_sections,
            hidden_text_spans_stripped=hidden_stripped,
            header_footer_lines_stripped=header_footer_stripped,
            pages=pages,
        )


    def _redact_adversarial_spans(self, doc: pymupdf.Document) -> int:
        """P6 defense: scan per-span and redact anything matching the adversarial profile.

        Matches ANY of: tiny font (< ``min_font_size_pt``), white-on-white
        (RGB components within ``white_epsilon`` of 1.0), zero-area bbox,
        bbox outside ``page.mediabox``.
        """
        count = 0
        for page in doc:
            mediabox = page.mediabox
            page_flagged = 0
            try:
                page_dict = page.get_text("dict")
            except Exception as exc:  # noqa: BLE001 — PyMuPDF may raise various subclasses
                log.warning(
                    "pdf_page_text_dict_failed page=%s error=%s",
                    page.number,
                    exc,
                )
                continue
            for block in page_dict.get("blocks", []):
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        if self._is_adversarial(span, mediabox):
                            try:
                                bbox = pymupdf.Rect(span["bbox"])
                                page.add_redact_annot(bbox, fill=(1, 1, 1))
                                page_flagged += 1
                                count += 1
                            except Exception as exc:  # noqa: BLE001
                                log.warning(
                                    "pdf_redact_annot_failed page=%s error=%s",
                                    page.number,
                                    exc,
                                )
            if page_flagged:
                try:
                    page.apply_redactions(images=pymupdf.PDF_REDACT_IMAGE_NONE)
                except Exception as exc:  # noqa: BLE001
                    log.warning(
                        "pdf_apply_redactions_failed page=%s error=%s",
                        page.number,
                        exc,
                    )
        return count

    def _is_adversarial(self, span: dict, mediabox: pymupdf.Rect) -> bool:
        if span.get("size", 999.0) < self.min_font_size_pt:
            return True
        color_int = span.get("color", 0)
        try:
            r, g, b = pymupdf.sRGB_to_pdf(color_int)
        except Exception:  # noqa: BLE001 — malformed color
            return False
        if min(r, g, b) > (1.0 - self.white_epsilon):
            return True
        bbox = span.get("bbox")
        if not bbox or len(bbox) != 4:
            return False
        x0, y0, x1, y1 = bbox
        if (x1 - x0) * (y1 - y0) <= 0.0:
            return True
        try:
            if not pymupdf.Rect(bbox).intersects(mediabox):
                return True
        except Exception:  # noqa: BLE001
            return False
        return False

    def _scrub_headers_footers(self, chunks: list[dict]) -> tuple[str, int]:
        """Remove lines that repeat (fuzzy, ratio >= 0.9) across >= 3 pages.

        Scope: first 2 + last 2 non-empty lines of each page are candidates.
        Secondary benefit: standalone short digit-only lines (page numbers)
        get stripped by the same mechanism since they repeat structurally.
        """
        per_page_candidates: list[list[str]] = []
        for ch in chunks:
            text = ch.get("text", "") if isinstance(ch, dict) else str(ch)
            lines = [ln for ln in text.splitlines() if ln.strip()]
            if not lines:
                per_page_candidates.append([])
                continue
            candidates = lines[:2] + lines[-2:]
            seen: set[str] = set()
            filtered: list[str] = []
            for ln in candidates:
                key = ln.strip()
                if key in seen:
                    continue
                if len(key) > self.max_header_footer_chars:
                    continue
                seen.add(key)
                filtered.append(ln)
            per_page_candidates.append(filtered)

        flat_with_page: list[tuple[int, str]] = []
        for page_idx, page_lines in enumerate(per_page_candidates):
            for ln in page_lines:
                flat_with_page.append((page_idx, ln))

        flagged: set[str] = set()
        distinct_lines = list({ln.strip() for _, ln in flat_with_page})
        for ln in distinct_lines:
            if not ln:
                continue
            pages_seen: set[int] = set()
            for pidx, other in flat_with_page:
                if difflib.SequenceMatcher(None, ln, other.strip()).ratio() >= 0.9:
                    pages_seen.add(pidx)
                    if len(pages_seen) >= 3:
                        break
            if len(pages_seen) >= 3:
                flagged.add(ln)

        cleaned_pages: list[str] = []
        total_stripped = 0
        for ch in chunks:
            text = ch.get("text", "") if isinstance(ch, dict) else str(ch)
            out_lines: list[str] = []
            for ln in text.splitlines():
                if ln.strip() in flagged:
                    total_stripped += 1
                    continue
                out_lines.append(ln)
            cleaned_pages.append("\n".join(out_lines))

        return "\n\n".join(cleaned_pages), total_stripped
