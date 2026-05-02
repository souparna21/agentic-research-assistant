"""HTTP fetchers for Semantic Scholar + arXiv.

- ``S2Client`` — thin wrapper over the ``semanticscholar`` 0.12 PyPI client
  with tenacity-driven exponential backoff on HTTP 429 (RET-04).
- ``ArxivClient`` (plan 02-03) — thin wrapper over ``arxiv`` 3.0 on
  export.arxiv.org subdomain (P15 defense).
- ``PdfDownloader`` (plan 02-04) — httpx.Client with 60s timeout for
  streaming PDF GETs.

Design principle: agents depend on these stable dataclass boundaries
(``S2SearchResult``, ``ArxivMatch``) so library versions can drift
without breaking agent logic or test fixtures.

Library integration notes (``semanticscholar`` 0.12.0):
    - ``SemanticScholar`` is a sync facade over ``AsyncSemanticScholar``;
      transport is ``httpx.AsyncClient`` under the hood (respx intercepts).
    - On HTTP 429 the library raises ``ConnectionRefusedError`` (NOT
      ``httpx.HTTPStatusError``) — verified against
      ``semanticscholar/ApiRequester.py`` lines 139-140. Our tenacity
      predicate therefore lists ``ConnectionRefusedError`` alongside
      ``httpx.HTTPStatusError`` for forward compatibility.
    - The library ships its own ``@tenacity.retry`` (stop=10,
      wait_exponential(min=5, max=60), retry_if=ConnectionRefusedError) on
      ``_get_data_async``. We disable that inner retry via
      ``SemanticScholar(retry=False, ...)`` so our single outer tenacity
      decorator is the only retry layer — deterministic 2/4/8/16 backoff
      matching RET-04 and VALIDATION rows 2-02-03..2-02-04.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import arxiv
import httpx
from semanticscholar import SemanticScholar
from tenacity import (
    RetryError,
    before_sleep_log,
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

log = logging.getLogger(__name__)


S2_FIELDS: list[str] = [
    "paperId",
    "title",
    "authors",
    "year",
    "venue",
    "citationCount",
    "abstract",
    "openAccessPdf",
    "externalIds",
    "tldr",
]


@dataclass(frozen=True)
class S2SearchResult:
    """Normalized S2 response — stable boundary for RetrievalAgent + tests.

    Frozen so downstream agents cannot accidentally mutate retrieval output.
    The ``authors`` field is ``list[str]`` per the plan's must-haves.truths
    contract; instances are therefore NOT hashable — RetrievalAgent (plan
    02-04) should dedup via ``paper_id`` (primary) and
    ``(normalize_title(title), first_author_surname.lower(), year)``
    (secondary) tuple keys, not ``hash(result)``.
    """

    paper_id: str
    title: str
    authors: list[str]
    year: int | None
    venue: str | None
    citation_count: int | None
    abstract: str | None
    pdf_url: str | None
    arxiv_id: str | None
    doi: str | None


class S2RateLimit(Exception):
    """Explicit marker — distinguishes 'truly rate limited' from other HTTP errors.

    Reserved for callers that want to raise their own signal without wrapping
    an httpx-specific exception. The tenacity decorator on ``S2Client.search``
    retries on ``ConnectionRefusedError`` (which is what the ``semanticscholar``
    library itself surfaces for HTTP 429 per its ``ApiRequester`` source) AND
    ``httpx.HTTPStatusError`` AND ``S2RateLimit`` — belt-and-suspenders if the
    library's 429 surfacing changes in a minor release.
    """


@dataclass
class S2Client:
    """Semantic Scholar search client with 2/4/8/16s backoff on HTTP 429.

    Usage::

        client = S2Client(timeout=30.0)
        results: list[S2SearchResult] = client.search("RAG", limit=10)

    The ``retry=False`` we pass to the underlying ``SemanticScholar`` client
    disables its built-in 10-attempt/5-60s retry — our outer tenacity
    decorator is the single source of truth for retry semantics.
    """

    timeout: float = 30.0
    api_key: str | None = None

    _client: SemanticScholar = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._client = SemanticScholar(
            timeout=self.timeout,
            api_key=self.api_key,
            retry=False,
        )

    @retry(
        stop=stop_after_attempt(4),
        wait=wait_exponential(multiplier=2, min=2, max=16),
        retry=retry_if_exception_type(
            (httpx.HTTPStatusError, ConnectionRefusedError, S2RateLimit)
        ),
        reraise=True,
        before_sleep=before_sleep_log(log, logging.WARNING),
    )
    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        year_since: int | None = None,
        min_citation_count: int | None = None,
    ) -> list[S2SearchResult]:
        """One page of up to ``limit`` results for ``query``.

        Tenacity retries on HTTP 429 with 2 / 4 / 8 / 16 second backoff,
        4 attempts total. The ``semanticscholar`` 0.12 library surfaces 429
        as ``ConnectionRefusedError`` (see module docstring); we also
        retry on ``httpx.HTTPStatusError`` for cross-version robustness.
        On four consecutive failures the last exception is re-raised
        (``reraise=True``) — RetrievalAgent records it in ``retrieval_stats``.
        """
        year_kwarg = f"{year_since}-" if year_since else None
        kwargs: dict[str, Any] = {
            "query": query,
            "limit": limit,
            "fields": S2_FIELDS,
        }
        if year_kwarg:
            kwargs["year"] = year_kwarg
        if min_citation_count:
            kwargs["min_citation_count"] = min_citation_count

        try:
            results = self._client.search_paper(**kwargs)
        except RetryError as exc:
            cause = exc.last_attempt.exception() if exc.last_attempt else None
            if cause is not None:
                raise cause from exc
            raise

        return [self._normalize(p) for p in results]

    @staticmethod
    def _normalize(p: Any) -> S2SearchResult:
        """Adapt a semanticscholar ``Paper`` (or dict) to ``S2SearchResult``.

        Robust to attribute-access (typed ``Paper``) and dict-access shapes
        the library may produce across minor versions.
        """

        def _g(obj: Any, key: str, default: Any = None) -> Any:
            if obj is None:
                return default
            if isinstance(obj, dict):
                return obj.get(key, default)
            if hasattr(obj, key):
                val = getattr(obj, key)
                return val if val is not None else default
            return default

        ext = _g(p, "externalIds") or {}
        pdf = _g(p, "openAccessPdf") or {}

        authors_raw = _g(p, "authors") or []
        authors: list[str] = []
        for a in authors_raw:
            name = _g(a, "name")
            if name:
                authors.append(name)

        return S2SearchResult(
            paper_id=_g(p, "paperId") or "",
            title=_g(p, "title") or "",
            authors=authors,
            year=_g(p, "year"),
            venue=_g(p, "venue"),
            citation_count=_g(p, "citationCount"),
            abstract=_g(p, "abstract"),
            pdf_url=pdf.get("url") if isinstance(pdf, dict) else None,
            arxiv_id=ext.get("ArXiv") if isinstance(ext, dict) else None,
            doi=ext.get("DOI") if isinstance(ext, dict) else None,
        )



_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "of", "for", "and", "or",
    "on", "in", "to", "with", "via", "using",
})


def _tokenize_title(title: str) -> set[str]:
    """Lowercase, strip punctuation, drop stopwords and length-1 tokens.

    Used by :func:`jaccard` for title-overlap comparison in
    :meth:`ArxivClient.lookup_by_title`. Numbers are kept (arXiv titles
    sometimes contain meaningful digits, e.g., ``GPT-3``, ``ResNet-50``).
    """
    tokens = re.findall(r"[a-z0-9]+", title.lower())
    return {t for t in tokens if t not in _STOPWORDS and len(t) > 1}


def jaccard(a: str, b: str) -> float:
    """Jaccard token-overlap over normalized title tokens. Range ``[0, 1]``.

    Used as the accept predicate for arXiv title-search fallback
    (RET-06: accept the top search result only when Jaccard ≥ 0.8).
    Two empty token sets produce ``0.0`` (never a false positive).
    """
    ta, tb = _tokenize_title(a), _tokenize_title(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


@dataclass(frozen=True)
class ArxivMatch:
    """Normalized arXiv response — stable boundary for RetrievalAgent + tests.

    Invariants:
        - ``pdf_url`` is ALWAYS on the ``export.arxiv.org`` subdomain (P15).
          :meth:`ArxivClient._to_match` rewrites the host before construction.
        - ``withdrawn`` is ``True`` iff ``Result.comment`` contains
          ``"withdrawn"`` (case-insensitive). Tiny-PDF belt-and-suspenders
          detection happens downstream in RetrievalAgent (plan 02-04).
    """

    arxiv_id: str
    title: str
    pdf_url: str
    withdrawn: bool


class ArxivClient:
    """Wrapper around ``arxiv`` 3.0 configured for the ``export.arxiv.org`` subdomain.

    The primary ``arxiv.org`` host rate-limits (and sometimes 403s) automated
    traffic; ``export.arxiv.org`` is explicitly intended for programmatic
    access. This client routes BOTH the query URL (via
    ``arxiv.Client.query_url_format``) AND the returned ``pdf_url`` (via
    host rewrite in :meth:`_to_match`) through ``export.arxiv.org``
    (P15 defense).

    Usage::

        client = ArxivClient()
        match = client.lookup_by_id("2005.11401")
        if match and not match.withdrawn:
            download(match.pdf_url)  # goes to export.arxiv.org

    The underlying ``arxiv.Client(page_size=5, delay_seconds=3.0,
    num_retries=3)`` provides arXiv's community-recommended 3-second
    politeness delay and the library's own retry semantics. We do not wrap
    this client in a project-level tenacity decorator — the library's built-in
    retry + delay is sufficient for arXiv's published rate limits, and
    adding a second retry layer would double the backoff budget.
    """

    def __init__(self) -> None:
        self._client = arxiv.Client(
            page_size=5,
            delay_seconds=3.0,
            num_retries=3,
        )
        self._client.query_url_format = "https://export.arxiv.org/api/query?{}"

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        year_since: int | None = None,
    ) -> list[S2SearchResult]:
        """Free-text discovery search — used as fallback when S2 is unavailable.

        Returns ``S2SearchResult``-shaped rows so the RetrievalAgent's
        downstream dedup + PDF-fallback chain stays unchanged. ``paper_id``
        is prefixed with ``arxiv:`` to keep ID space disjoint from S2.

        ``citation_count`` is ``None`` (arXiv API does not expose it).
        ``pdf_url`` is left ``None`` so the agent's PDF resolution path
        uses the ``arxiv_id`` route through ``lookup_by_id`` — identical
        to the regular S2-then-arXiv flow, just without S2.
        """
        arxiv_query = query
        if year_since:
            arxiv_query = (
                f"({query}) AND submittedDate:"
                f"[{year_since}01010000 TO 99991231235959]"
            )
        search = arxiv.Search(
            query=arxiv_query,
            max_results=limit,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        out: list[S2SearchResult] = []
        for result in self._client.results(search):
            match = self._to_match(result)
            authors = [a.name for a in (result.authors or [])]
            year = result.published.year if result.published else None
            out.append(
                S2SearchResult(
                    paper_id=f"arxiv:{match.arxiv_id}",
                    title=match.title,
                    authors=authors,
                    year=year,
                    venue="arXiv",
                    citation_count=None,
                    abstract=result.summary,
                    pdf_url=None,
                    arxiv_id=match.arxiv_id,
                    doi=None,
                )
            )
        return out

    def lookup_by_id(self, arxiv_id: str) -> ArxivMatch | None:
        """Direct ID lookup — used when ``S2SearchResult.arxiv_id`` is present.

        Returns ``None`` if arXiv returned no results for the given ID
        (e.g., the ID is malformed or the paper has been fully removed,
        distinct from "withdrawn"). Withdrawn papers ARE returned — the
        caller inspects :attr:`ArxivMatch.withdrawn` and decides whether
        to skip.
        """
        search = arxiv.Search(id_list=[arxiv_id])
        try:
            result = next(self._client.results(search))
        except StopIteration:
            return None
        return self._to_match(result)

    def lookup_by_title(
        self,
        title: str,
        year: int | None = None,
        accept_threshold: float = 0.8,
    ) -> ArxivMatch | None:
        """Title-search fallback — accept the TOP result only if Jaccard ≥ threshold.

        Returns ``None`` when:
            - No arXiv search results match the query, OR
            - The top result's Jaccard token-overlap with ``title`` is
              below ``accept_threshold`` (default 0.8, per RET-06).

        The ≥ 0.8 bar is deliberately strict: a false-positive arXiv match
        would pollute the downstream RAG corpus with an unrelated paper.
        Prefer ``None`` (graceful fallback to abstract-only) over a weak
        match.
        """
        query = f'ti:"{title}"'
        if year:
            query += f" AND submittedDate:[{year}01010000 TO {year}12312359]"
        search = arxiv.Search(
            query=query,
            max_results=5,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        for result in self._client.results(search):
            if jaccard(title, result.title) >= accept_threshold:
                return self._to_match(result)
        return None

    @staticmethod
    def _to_match(result: arxiv.Result) -> ArxivMatch:
        """Adapt an ``arxiv.Result`` into a normalized :class:`ArxivMatch`.

        - ``arxiv_id`` is the final path segment of ``entry_id`` (e.g.,
          ``"2005.11401v4"`` from ``"http://arxiv.org/abs/2005.11401v4"``).
        - ``pdf_url`` host is rewritten from ``arxiv.org`` →
          ``export.arxiv.org`` (first occurrence only; idempotent on URLs
          already on the subdomain). The scheme is also normalized to
          ``https://`` so downstream PDF downloads go over TLS.
        - ``withdrawn`` is ``True`` iff ``result.comment`` contains
          ``"withdrawn"`` (case-insensitive).
        """
        arxiv_id = (result.entry_id or "").rsplit("/", 1)[-1]

        raw_pdf = result.pdf_url or ""
        pdf_url = raw_pdf.replace("arxiv.org", "export.arxiv.org", 1)
        if pdf_url.startswith("http://export.arxiv.org"):
            pdf_url = pdf_url.replace("http://", "https://", 1)

        withdrawn = bool(
            result.comment and "withdrawn" in result.comment.lower()
        )

        return ArxivMatch(
            arxiv_id=arxiv_id,
            title=result.title or "",
            pdf_url=pdf_url,
            withdrawn=withdrawn,
        )



_USER_AGENT = "ara/0.1 (IE624 course project; mailto:jsouparna2@gmail.com)"


class PdfDownloader:
    """Streams PDF GETs with a 60-second wall-clock timeout.

    Separate from the S2/arXiv API clients: the transport options differ
    (follow_redirects=True, Accept: application/pdf, long timeout for
    5-50MB files). Writes to ``dest.with_suffix(dest.suffix + '.tmp')``
    then ``tmp.replace(dest)`` so Ctrl-C mid-download leaves the target
    ABSENT, never truncated (pitfall P7 parallel to atomic_write_json).

    Usage::

        dl = PdfDownloader(timeout=60.0)
        try:
            bytes_written = dl.download("https://example.com/p.pdf", dest)
        except (httpx.HTTPError, httpx.TimeoutException) as exc:
            # caller records failure in retrieval_stats.pdf_download_failures
            ...
        finally:
            dl.close()

    The caller (RetrievalAgent) is responsible for narrow per-paper
    exception handling — PdfDownloader itself does not retry.
    """

    def __init__(self, timeout: float = 60.0) -> None:
        self._client = httpx.Client(
            timeout=timeout,
            follow_redirects=True,
            headers={"User-Agent": _USER_AGENT, "Accept": "application/pdf"},
        )

    def download(self, url: str, dest: Path) -> int:
        """GET ``url``, stream to ``dest``, return bytes written on success.

        Raises ``httpx.HTTPError`` / ``httpx.TimeoutException`` on any
        failure. RetrievalAgent is responsible for narrow per-paper
        exception handling at the call site.
        """
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        total = 0
        with self._client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(tmp, "wb") as fp:
                for chunk in resp.iter_bytes(chunk_size=64 * 1024):
                    fp.write(chunk)
                    total += len(chunk)
        tmp.replace(dest)
        return total

    def close(self) -> None:
        self._client.close()
