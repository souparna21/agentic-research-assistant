"""Unit tests for RetrievalAgent (RET-03, RET-05..09).

We inject fake ``S2Client`` / ``ArxivClient`` / ``PdfDownloader`` instances.
HTTP-layer correctness is owned by ``test_fetchers_s2.py`` +
``test_fetchers_arxiv.py``. Here we test ORCHESTRATION: dedup, fallback
chain, stats, stage-report emission, per-paper failure isolation.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from ara.agents.retrieval import RetrievalAgent
from ara.services.fetchers import ArxivMatch, S2SearchResult
from ara.state import PipelineState, RetrievalStats




@dataclass
class _FakeS2:
    """Returns pre-scripted results keyed by query string."""

    responses: dict[str, list[S2SearchResult]] = field(default_factory=dict)
    calls: list[tuple[str, dict]] = field(default_factory=list)
    errors: dict[str, Exception] = field(default_factory=dict)

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        year_since=None,
        min_citation_count=None,
    ) -> list[S2SearchResult]:
        self.calls.append(
            (
                query,
                {
                    "limit": limit,
                    "year_since": year_since,
                    "min_citation_count": min_citation_count,
                },
            )
        )
        if query in self.errors:
            raise self.errors[query]
        return list(self.responses.get(query, []))


@dataclass
class _FakeArxiv:
    by_id: dict[str, ArxivMatch | None] = field(default_factory=dict)
    by_title: dict[str, ArxivMatch | None] = field(default_factory=dict)
    search_responses: dict[str, list[S2SearchResult]] = field(default_factory=dict)
    search_calls: list[str] = field(default_factory=list)

    def lookup_by_id(self, arxiv_id: str) -> ArxivMatch | None:
        return self.by_id.get(arxiv_id)

    def lookup_by_title(
        self, title: str, year=None, accept_threshold: float = 0.8
    ) -> ArxivMatch | None:
        return self.by_title.get(title)

    def search(
        self, query: str, *, limit: int = 10, year_since=None
    ) -> list[S2SearchResult]:
        self.search_calls.append(query)
        return list(self.search_responses.get(query, []))


@dataclass
class _FakePdfDownloader:
    """url -> (bytes_written_int | Exception_instance)."""

    responses: dict[str, int | Exception] = field(default_factory=dict)
    calls: list[tuple[str, Path]] = field(default_factory=list)

    def download(self, url: str, dest: Path) -> int:
        self.calls.append((url, dest))
        resp = self.responses.get(url)
        if isinstance(resp, Exception):
            raise resp
        if resp is None:
            resp = 50_000
        dest.parent.mkdir(parents=True, exist_ok=True)
        payload_size = max(resp, 0)
        dest.write_bytes(b"%PDF-" + b"x" * max(payload_size - 5, 0))
        return resp

    def close(self) -> None:  # pragma: no cover - not exercised
        pass




def _s2_result(paper_id: str, title: str, **overrides) -> S2SearchResult:
    base = dict(
        paper_id=paper_id,
        title=title,
        authors=["Alice"],
        year=2020,
        venue="NeurIPS",
        citation_count=100,
        abstract="abstract...",
        pdf_url=None,
        arxiv_id=None,
        doi=None,
    )
    base.update(overrides)
    return S2SearchResult(**base)


def _make_state(tmp_path: Path, terms: list[str]) -> PipelineState:
    state = PipelineState(
        run_id="R-RET",
        question="q",
        config_snapshot={},
        runs_dir=str(tmp_path / "runs"),
    )
    state.search_terms = terms
    return state




def test_dedup_by_paper_id(tmp_path):
    """RET-03 / 2-04-01: same paperId across multiple terms -> one Paper entry."""
    same = _s2_result("paper-A", "RAG for NLP")
    s2 = _FakeS2(responses={"RAG": [same], "retrieval": [same]})
    state = _make_state(tmp_path, ["RAG", "retrieval"])
    out = RetrievalAgent(s2=s2, arxiv=_FakeArxiv(), downloader=_FakePdfDownloader()).run(state)
    assert len(out.papers) == 1
    assert out.papers[0].paper_id == "paper-A"


def test_dedup_by_title_author_year(tmp_path):
    """RET-03 / 2-04-02: same (title, first_author.lower(), year) merges."""
    a = _s2_result("id-v1", "Retrieval Augmented Generation", authors=["Alice", "Bob"], year=2020)
    b = _s2_result("id-v2", "retrieval-augmented generation", authors=["Alice"], year=2020)
    s2 = _FakeS2(responses={"t1": [a], "t2": [b]})
    state = _make_state(tmp_path, ["t1", "t2"])
    out = RetrievalAgent(s2=s2, arxiv=_FakeArxiv(), downloader=_FakePdfDownloader()).run(state)
    assert len(out.papers) == 1


def test_s2_pdf_download_when_url_present(tmp_path):
    """RET-05 / 2-04-03: openAccessPdf.url present -> PdfDownloader, pdf_source='s2'."""
    r = _s2_result("p1", "T", pdf_url="https://example.com/p1.pdf")
    s2 = _FakeS2(responses={"q": [r]})
    dl = _FakePdfDownloader(responses={"https://example.com/p1.pdf": 50_000})
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(s2=s2, arxiv=_FakeArxiv(), downloader=dl).run(state)
    assert len(out.papers) == 1
    p = out.papers[0]
    assert p.pdf_source == "s2"
    assert p.pdf_path and Path(p.pdf_path).exists()
    assert p.full_text_available is True
    assert len(dl.calls) == 1
    assert dl.calls[0][0] == "https://example.com/p1.pdf"


def test_arxiv_fallback_by_external_id(tmp_path):
    """RET-06 / 2-04-04: no openAccessPdf, externalIds.ArXiv present -> arXiv path."""
    r = _s2_result("p2", "Arxiv Paper", arxiv_id="2005.11401")
    s2 = _FakeS2(responses={"q": [r]})
    arx = _FakeArxiv(
        by_id={
            "2005.11401": ArxivMatch(
                arxiv_id="2005.11401v4",
                title="Arxiv Paper",
                pdf_url="https://export.arxiv.org/pdf/2005.11401v4",
                withdrawn=False,
            ),
        }
    )
    dl = _FakePdfDownloader(
        responses={"https://export.arxiv.org/pdf/2005.11401v4": 50_000}
    )
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(s2=s2, arxiv=arx, downloader=dl).run(state)
    assert len(out.papers) == 1
    p = out.papers[0]
    assert p.pdf_source == "arxiv"
    assert p.full_text_available is True
    assert dl.calls[0][0].startswith("https://export.arxiv.org/")


def test_tiny_arxiv_pdf_marked_withdrawn(tmp_path):
    """RET-07 / 2-04-05: arXiv PDF < 10KB -> treated as withdrawn, abstract-only."""
    r = _s2_result("p3", "Withdrawn Paper", arxiv_id="1234.5678")
    s2 = _FakeS2(responses={"q": [r]})
    arx = _FakeArxiv(
        by_id={
            "1234.5678": ArxivMatch(
                arxiv_id="1234.5678",
                title="Withdrawn Paper",
                pdf_url="https://export.arxiv.org/pdf/1234.5678",
                withdrawn=False,
            ),
        }
    )
    dl = _FakePdfDownloader(
        responses={"https://export.arxiv.org/pdf/1234.5678": 5_000}
    )
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(s2=s2, arxiv=arx, downloader=dl).run(state)
    p = out.papers[0]
    assert p.pdf_source == "abstract-only", f"expected abstract-only, got {p.pdf_source}"
    assert p.full_text_available is False


def test_withdrawn_comment_triggers_abstract_only(tmp_path):
    """RET-07 primary path: ArxivMatch.withdrawn=True -> skip, abstract-only."""
    r = _s2_result("p-w", "Withdrawn by comment", arxiv_id="9999.0001")
    s2 = _FakeS2(responses={"q": [r]})
    arx = _FakeArxiv(
        by_id={
            "9999.0001": ArxivMatch(
                arxiv_id="9999.0001",
                title="Withdrawn by comment",
                pdf_url="https://export.arxiv.org/pdf/9999.0001",
                withdrawn=True,
            ),
        }
    )
    dl = _FakePdfDownloader()
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(s2=s2, arxiv=arx, downloader=dl).run(state)
    assert out.papers[0].pdf_source == "abstract-only"
    assert len(dl.calls) == 0


def test_full_text_available_flag_correctness(tmp_path):
    """RET-08 / 2-04-06: full_text_available == (pdf_source in {'s2','arxiv'})."""
    r_s2 = _s2_result("a", "A", pdf_url="https://e/a.pdf")
    r_arx = _s2_result("b", "B", arxiv_id="0000.0001")
    r_abs = _s2_result("c", "C")
    s2 = _FakeS2(responses={"q": [r_s2, r_arx, r_abs]})
    arx = _FakeArxiv(
        by_id={
            "0000.0001": ArxivMatch(
                arxiv_id="0000.0001",
                title="B",
                pdf_url="https://export.arxiv.org/pdf/0000.0001",
                withdrawn=False,
            )
        }
    )
    dl = _FakePdfDownloader(
        responses={
            "https://e/a.pdf": 50_000,
            "https://export.arxiv.org/pdf/0000.0001": 50_000,
        }
    )
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(s2=s2, arxiv=arx, downloader=dl).run(state)
    for p in out.papers:
        assert p.full_text_available == (p.pdf_source in {"s2", "arxiv"}), p


def test_pdf_source_attribution(tmp_path):
    """RET-08 / 2-04-07: three outcomes yield three distinct pdf_source values."""
    r_s2 = _s2_result("a", "A", pdf_url="https://e/a.pdf")
    r_arx = _s2_result("b", "B", arxiv_id="0000.0001")
    r_abs = _s2_result("c", "C")
    s2 = _FakeS2(responses={"q": [r_s2, r_arx, r_abs]})
    arx = _FakeArxiv(
        by_id={
            "0000.0001": ArxivMatch(
                arxiv_id="0000.0001",
                title="B",
                pdf_url="https://export.arxiv.org/pdf/0000.0001",
                withdrawn=False,
            )
        }
    )
    dl = _FakePdfDownloader(
        responses={
            "https://e/a.pdf": 50_000,
            "https://export.arxiv.org/pdf/0000.0001": 50_000,
        }
    )
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(s2=s2, arxiv=arx, downloader=dl).run(state)
    sources = {p.paper_id: p.pdf_source for p in out.papers}
    assert sources["a"] == "s2"
    assert sources["b"] == "arxiv"
    assert sources["c"] == "abstract-only"


def test_retrieval_stats_populated(tmp_path):
    """RET-09 / 2-04-08: state.retrieval_stats is a RetrievalStats with populated fields."""
    r = _s2_result("p1", "T", pdf_url="https://e/p1.pdf")
    s2 = _FakeS2(responses={"term1": [r], "term2": []})
    dl = _FakePdfDownloader(responses={"https://e/p1.pdf": 50_000})
    state = _make_state(tmp_path, ["term1", "term2"])
    out = RetrievalAgent(s2=s2, arxiv=_FakeArxiv(), downloader=dl).run(state)
    assert isinstance(out.retrieval_stats, RetrievalStats)
    assert out.retrieval_stats.search_terms_submitted == 2
    assert out.retrieval_stats.s2_requests == 2
    assert out.retrieval_stats.pdf_download_count == 1
    assert out.retrieval_stats.pdf_download_bytes == 50_000
    assert "term2" in out.retrieval_stats.empty_query_terms


def test_retrieval_emits_stage_report(tmp_path):
    """RET-09 / 2-04-09: exactly one StageReport(stage='retrieval') per run."""
    r = _s2_result("p1", "T")
    s2 = _FakeS2(responses={"q": [r]})
    state = _make_state(tmp_path, ["q"])
    before = len(state.stage_reports)
    out = RetrievalAgent(
        s2=s2, arxiv=_FakeArxiv(), downloader=_FakePdfDownloader()
    ).run(state)
    assert len(out.stage_reports) == before + 1
    rep = out.stage_reports[-1]
    assert rep.stage == "retrieval"
    assert rep.inputs_count == 1
    assert rep.outputs_count == 1


def test_zero_results_empty_papers_continues_cleanly(tmp_path):
    """All terms return zero results AND arxiv-fallback also empty -> no papers."""
    s2 = _FakeS2(responses={"q1": [], "q2": []})
    arx = _FakeArxiv(search_responses={"q1": [], "q2": []})
    state = _make_state(tmp_path, ["q1", "q2"])
    out = RetrievalAgent(
        s2=s2, arxiv=arx, downloader=_FakePdfDownloader()
    ).run(state)
    assert out.papers == []
    assert set(out.retrieval_stats.empty_query_terms) == {"q1", "q2"}
    assert out.retrieval_stats.arxiv_discovery_hits == 0


def test_arxiv_discovery_fallback_when_s2_returns_empty(tmp_path):
    """S2 returns 0 across all terms -> ArxivClient.search() takes over."""
    arxiv_hit = _s2_result(
        "arxiv:2005.11401",
        "Discovered Via Arxiv",
        arxiv_id="2005.11401",
    )
    s2 = _FakeS2(responses={"q": []})
    arx = _FakeArxiv(
        search_responses={"q": [arxiv_hit]},
        by_id={
            "2005.11401": ArxivMatch(
                arxiv_id="2005.11401v1",
                title="Discovered Via Arxiv",
                pdf_url="https://export.arxiv.org/pdf/2005.11401v1",
                withdrawn=False,
            )
        },
    )
    dl = _FakePdfDownloader(
        responses={"https://export.arxiv.org/pdf/2005.11401v1": 50_000}
    )
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(s2=s2, arxiv=arx, downloader=dl).run(state)
    assert len(out.papers) == 1
    assert out.papers[0].title == "Discovered Via Arxiv"
    assert out.papers[0].pdf_source == "arxiv"
    assert out.retrieval_stats.arxiv_discovery_hits == 1
    assert arx.search_calls == ["q"]


def test_arxiv_discovery_fallback_when_all_s2_terms_429(tmp_path):
    """All S2 terms raise 429 -> arxiv-discovery fallback still runs."""
    req = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search")
    s2 = _FakeS2(
        errors={
            "t1": httpx.HTTPStatusError(
                "429", request=req, response=httpx.Response(429, request=req)
            ),
            "t2": httpx.HTTPStatusError(
                "429", request=req, response=httpx.Response(429, request=req)
            ),
        }
    )
    arxiv_hit = _s2_result(
        "arxiv:1234.5678", "Saved By Arxiv", arxiv_id="1234.5678"
    )
    arx = _FakeArxiv(
        search_responses={"t1": [arxiv_hit], "t2": []},
        by_id={
            "1234.5678": ArxivMatch(
                arxiv_id="1234.5678",
                title="Saved By Arxiv",
                pdf_url="https://export.arxiv.org/pdf/1234.5678",
                withdrawn=False,
            )
        },
    )
    dl = _FakePdfDownloader(
        responses={"https://export.arxiv.org/pdf/1234.5678": 50_000}
    )
    state = _make_state(tmp_path, ["t1", "t2"])
    out = RetrievalAgent(s2=s2, arxiv=arx, downloader=dl).run(state)
    assert len(out.papers) == 1
    assert out.papers[0].title == "Saved By Arxiv"
    assert out.retrieval_stats.arxiv_discovery_hits == 1
    assert "t2" in out.retrieval_stats.empty_query_terms
    assert arx.search_calls == ["t1", "t2"]


def test_arxiv_discovery_not_invoked_when_s2_has_any_hits(tmp_path):
    """S2 returning even one paper means no arxiv-discovery fallback fires."""
    s2_hit = _s2_result("p1", "From S2")
    s2 = _FakeS2(responses={"q": [s2_hit]})
    arx = _FakeArxiv(
        search_responses={"q": [_s2_result("arxiv:9999.0001", "Should Not Be Used")]}
    )
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(
        s2=s2, arxiv=arx, downloader=_FakePdfDownloader()
    ).run(state)
    assert len(out.papers) == 1
    assert out.papers[0].title == "From S2"
    assert out.retrieval_stats.arxiv_discovery_hits == 0
    assert arx.search_calls == []


def test_per_paper_download_failure_isolated(tmp_path):
    """P8 defense: a single download failure does NOT halt the stage."""
    a = _s2_result("ok", "Good", pdf_url="https://e/good.pdf")
    b = _s2_result("bad", "Bad", pdf_url="https://e/bad.pdf")
    s2 = _FakeS2(responses={"q": [a, b]})
    req = httpx.Request("GET", "https://e/bad.pdf")
    dl = _FakePdfDownloader(
        responses={
            "https://e/good.pdf": 50_000,
            "https://e/bad.pdf": httpx.HTTPStatusError(
                "404", request=req, response=httpx.Response(404, request=req)
            ),
        }
    )
    state = _make_state(tmp_path, ["q"])
    out = RetrievalAgent(s2=s2, arxiv=_FakeArxiv(), downloader=dl).run(state)
    sources = {p.paper_id: p.pdf_source for p in out.papers}
    assert sources["ok"] == "s2"
    assert sources["bad"] == "abstract-only"
    assert any(pid == "bad" for pid, _reason in out.retrieval_stats.pdf_download_failures)


def test_s2_per_term_failure_narrow_exception_caught(tmp_path):
    """A transport error on ONE term does not halt retrieval of other terms."""
    r = _s2_result("p", "Good", pdf_url="https://e/p.pdf")
    req = httpx.Request("GET", "https://api.semanticscholar.org/graph/v1/paper/search")
    s2 = _FakeS2(
        responses={"good-term": [r]},
        errors={"bad-term": httpx.HTTPStatusError("429", request=req, response=httpx.Response(429, request=req))},
    )
    dl = _FakePdfDownloader(responses={"https://e/p.pdf": 50_000})
    state = _make_state(tmp_path, ["bad-term", "good-term"])
    out = RetrievalAgent(s2=s2, arxiv=_FakeArxiv(), downloader=dl).run(state)
    assert [p.paper_id for p in out.papers] == ["p"]
    assert out.retrieval_stats.s2_requests == 1
