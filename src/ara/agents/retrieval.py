"""RetrievalAgent — orchestrate S2 queries + PDF fallback chain (RET-03..09).

Per-run flow:
    1. Call ``S2Client.search(term)`` for each term in ``state.search_terms``.
       A per-term transport failure is caught narrowly (httpx.HTTPError /
       httpx.TimeoutException / tenacity.RetryError) and the term is
       silently skipped — retrieval continues with remaining terms.
    2. Merge results across terms with two dedup passes:
         (a) Primary key = ``S2SearchResult.paper_id``.
         (b) Secondary key = ``(normalize_title(title), first_author.lower(), year)``
             — catches the same paper indexed under different S2 IDs (e.g.,
             an S2 ID and an arXiv-sourced ID for the same preprint).
    3. For each deduped paper, resolve PDF via the fallback chain:
         (a) ``openAccessPdf.url`` from S2 -> ``PdfDownloader.download`` -> ``pdf_source='s2'``
         (b) ``externalIds.ArXiv`` present -> ``ArxivClient.lookup_by_id`` ->
             download -> ``pdf_source='arxiv'``
         (c) Title-search via ``ArxivClient.lookup_by_title`` (Jaccard >= 0.8)
             -> download -> ``pdf_source='arxiv'``
         (d) Otherwise ``pdf_source='abstract-only'``,
             ``full_text_available=False``.
    4. Withdrawal detection (belt-and-suspenders):
         - ``ArxivMatch.withdrawn=True`` (ArxivClient's comment-string check)
         - Downloaded PDF < 10KB (arXiv tombstone heuristic)
       Either trips the abstract-only fallback for that paper.
    5. **Per-paper narrow try/except** around every network call —
       ``(httpx.HTTPError, httpx.TimeoutException, tenacity.RetryError, OSError)``.
       Never ``except Exception`` (pitfall P8: silent masking of unknown
       bugs as "fallback_to_abstract" is explicitly rejected).
    6. Populate ``state.retrieval_stats = RetrievalStats(...)``; append one
       ``StageReport(stage='retrieval', ...)`` to ``state.stage_reports``.

Writes ONLY ``state.papers`` + ``state.retrieval_stats`` (+ append to the
shared append-only ``stage_reports``). Per ``FIELD_OWNERS['retrieval']``.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path

import httpx
import structlog
import tenacity

from ara.services.fetchers import (
    ArxivClient,
    ArxivMatch,
    PdfDownloader,
    S2Client,
    S2SearchResult,
)
from ara.state import Paper, PipelineState, RetrievalStats, StageReport

log = structlog.get_logger(__name__)

_TINY_PDF_THRESHOLD_BYTES = 10_000


def _normalize_title(title: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace.

    Used as the first component of the secondary dedup key (RET-03).
    """
    t = re.sub(r"[^a-z0-9\s]+", " ", title.lower())
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _secondary_key(sr: S2SearchResult) -> tuple[str, str, int | None]:
    """``(normalized_title, first_author_surname.lower(), year)`` dedup key.

    Papers missing authors get an empty surname — sort last within a
    title/year bucket but still deduped against other author-less entries.
    """
    first_author = ""
    if sr.authors:
        parts = sr.authors[0].split()
        if parts:
            first_author = parts[-1].lower()
    return (_normalize_title(sr.title or ""), first_author, sr.year)


@dataclass
class RetrievalAgent:
    """Implements the Stage protocol. FIELD_OWNERS-owned: papers, retrieval_stats."""

    s2: S2Client
    arxiv: ArxivClient
    downloader: PdfDownloader
    per_term_limit: int = 10
    year_since: int | None = None
    min_citation_count: int | None = None
    name: str = "retrieval"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "retrieval.marker"

    def run(self, state: PipelineState) -> PipelineState:
        t0 = time.perf_counter()
        s2_requests = 0
        s2_total_results = 0
        empty_terms: list[str] = []

        all_hits: list[S2SearchResult] = []
        for i, term in enumerate(state.search_terms):
            if i > 0:
                time.sleep(2.0)
            try:
                results = self.s2.search(
                    term,
                    limit=self.per_term_limit,
                    year_since=self.year_since,
                    min_citation_count=self.min_citation_count,
                )
            except (
                httpx.HTTPError,
                httpx.TimeoutException,
                tenacity.RetryError,
                ConnectionRefusedError,
            ) as exc:
                log.warning(
                    "s2_term_failed",
                    term=term,
                    error=str(exc),
                    error_type=type(exc).__name__,
                )
                continue
            s2_requests += 1
            s2_total_results += len(results)
            if not results:
                empty_terms.append(term)
            else:
                all_hits.extend(results)

        by_id: dict[str, S2SearchResult] = {}
        for hit in all_hits:
            if hit.paper_id and hit.paper_id not in by_id:
                by_id[hit.paper_id] = hit
        deduped: list[S2SearchResult] = []
        seen_secondary: set[tuple[str, str, int | None]] = set()
        for hit in by_id.values():
            key = _secondary_key(hit)
            if key in seen_secondary:
                continue
            seen_secondary.add(key)
            deduped.append(hit)

        papers: list[Paper] = []
        arxiv_fallback_hits = 0
        pdf_download_count = 0
        pdf_download_bytes = 0
        pdf_download_failures: list[tuple[str, str]] = []
        abstract_only_count = 0

        runs_dir = Path(state.runs_dir)
        pdf_dir = runs_dir / state.run_id / "pdfs"

        for hit in deduped:
            paper = Paper(
                paper_id=hit.paper_id,
                title=hit.title,
                authors=hit.authors,
                year=hit.year,
                venue=hit.venue,
                citation_count=hit.citation_count,
                abstract=hit.abstract,
            )
            resolved = False

            if hit.pdf_url:
                dest = pdf_dir / f"{hit.paper_id}.pdf"
                try:
                    bytes_written = self.downloader.download(hit.pdf_url, dest)
                except (
                    httpx.HTTPError,
                    httpx.TimeoutException,
                    tenacity.RetryError,
                    OSError,
                ) as exc:
                    pdf_download_failures.append((hit.paper_id, f"s2: {exc}"))
                else:
                    paper.pdf_url = hit.pdf_url
                    paper.pdf_path = str(dest)
                    paper.pdf_source = "s2"
                    paper.full_text_available = True
                    pdf_download_count += 1
                    pdf_download_bytes += bytes_written
                    resolved = True

            if not resolved and hit.arxiv_id:
                try:
                    match = self.arxiv.lookup_by_id(hit.arxiv_id)
                except (
                    httpx.HTTPError,
                    httpx.TimeoutException,
                    tenacity.RetryError,
                    OSError,
                ) as exc:
                    match = None
                    pdf_download_failures.append((hit.paper_id, f"arxiv-id: {exc}"))
                if match is not None and self._try_arxiv_download(
                    match, hit, paper, pdf_dir, pdf_download_failures
                ):
                    arxiv_fallback_hits += 1
                    pdf_download_count += 1
                    pdf_download_bytes += (
                        int(Path(paper.pdf_path).stat().st_size)
                        if paper.pdf_path
                        else 0
                    )
                    resolved = True

            if not resolved and hit.title:
                try:
                    match = self.arxiv.lookup_by_title(hit.title, year=hit.year)
                except (
                    httpx.HTTPError,
                    httpx.TimeoutException,
                    tenacity.RetryError,
                    OSError,
                ) as exc:
                    match = None
                    pdf_download_failures.append(
                        (hit.paper_id, f"arxiv-title: {exc}")
                    )
                if match is not None and self._try_arxiv_download(
                    match, hit, paper, pdf_dir, pdf_download_failures
                ):
                    arxiv_fallback_hits += 1
                    pdf_download_count += 1
                    pdf_download_bytes += (
                        int(Path(paper.pdf_path).stat().st_size)
                        if paper.pdf_path
                        else 0
                    )
                    resolved = True

            if not resolved:
                paper.pdf_source = "abstract-only"
                paper.full_text_available = False
                paper.fallback_to_abstract = True
                abstract_only_count += 1

            papers.append(paper)

        duration_ms = int((time.perf_counter() - t0) * 1000)
        state.papers = papers
        state.retrieval_stats = RetrievalStats(
            search_terms_submitted=len(state.search_terms),
            s2_requests=s2_requests,
            s2_429_retries=0,
            s2_total_results=s2_total_results,
            arxiv_fallback_hits=arxiv_fallback_hits,
            pdf_download_count=pdf_download_count,
            pdf_download_bytes=pdf_download_bytes,
            pdf_download_failures=pdf_download_failures,
            abstract_only_count=abstract_only_count,
            duration_ms=duration_ms,
            empty_query_terms=empty_terms,
        )
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=len(state.search_terms),
                outputs_count=len(papers),
                errors_count=len(pdf_download_failures),
                duration_ms=float(duration_ms),
            )
        )

        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state

    def _try_arxiv_download(
        self,
        match: ArxivMatch,
        hit: S2SearchResult,
        paper: Paper,
        pdf_dir: Path,
        failures: list[tuple[str, str]],
    ) -> bool:
        """Download an arXiv PDF; return True iff success + not withdrawn + >= 10KB.

        On failure (withdrawn flag, HTTP error, timeout, tiny tombstone)
        records a failure row and returns False — caller falls through to
        next link in the fallback chain (or abstract-only).
        """
        if match.withdrawn:
            failures.append((hit.paper_id, "arxiv-withdrawn"))
            return False
        dest = pdf_dir / f"{hit.paper_id}.pdf"
        try:
            bytes_written = self.downloader.download(match.pdf_url, dest)
        except (
            httpx.HTTPError,
            httpx.TimeoutException,
            tenacity.RetryError,
            OSError,
        ) as exc:
            failures.append((hit.paper_id, f"arxiv-download: {exc}"))
            return False
        if bytes_written < _TINY_PDF_THRESHOLD_BYTES:
            failures.append((hit.paper_id, f"arxiv-tiny-{bytes_written}B"))
            try:
                dest.unlink()
            except OSError:
                pass
            return False
        paper.pdf_url = match.pdf_url
        paper.pdf_path = str(dest)
        paper.pdf_source = "arxiv"
        paper.full_text_available = True
        return True
