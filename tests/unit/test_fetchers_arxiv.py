"""Unit tests for ArxivClient (RET-06, RET-07).

Cassettes live in ``tests/fixtures/http/``. Network is globally blocked
by ``pytest-socket --disable-socket`` (set in ``pyproject.toml``
addopts); any un-mocked HTTP call fails loudly.

Strategy note: the ``arxiv`` 3.0 library uses ``urllib`` (feedparser)
under the hood, NOT ``httpx``. Because ``respx`` intercepts only
``httpx`` calls, we mock at a cleaner boundary — stubbing
``client._client.results`` to yield fake ``arxiv.Result``-like objects
with the attributes our code reads (``entry_id``, ``title``,
``pdf_url``, ``comment``, ``primary_category``). This is also the
pattern the 02-RESEARCH.md §"Pattern 2" section recommends.

One test (``test_uses_export_arxiv_subdomain``) asserts the
``query_url_format`` override directly on the underlying
``arxiv.Client`` — the P15 defense is a configuration contract, not a
behavior that requires HTTP to be exercised.

VALIDATION.md required tests (3):
    - test_title_search_accepts_high_jaccard  (row 2-03-01, RET-06)
    - test_uses_export_arxiv_subdomain        (row 2-03-02, RET-06)
    - test_detects_withdrawn_via_comment      (row 2-03-03, RET-07)

Additional tests: inverse cases + jaccard() unit-level sanity.
"""
from __future__ import annotations

import types
from pathlib import Path

from ara.services.fetchers import ArxivClient, ArxivMatch, jaccard

REPO_ROOT = Path(__file__).resolve().parents[2]

ATOM_GOOD = (REPO_ROOT / "tests/fixtures/http/arxiv_rag_result.atom.xml").read_text()
ATOM_WITHDRAWN = (REPO_ROOT / "tests/fixtures/http/arxiv_withdrawn.atom.xml").read_text()
ATOM_NOMATCH = (REPO_ROOT / "tests/fixtures/http/arxiv_title_search_no_match.atom.xml").read_text()


def _make_result_stub(
    arxiv_id: str,
    title: str,
    pdf_url: str,
    comment: str | None,
) -> types.SimpleNamespace:
    """Build a fake ``arxiv.Result``-like object with our-code-read attrs.

    ArxivClient._to_match reads: entry_id, title, pdf_url, comment.
    lookup_by_title also reads result.title (via jaccard). No other
    attributes are touched by the implementation.
    """
    r = types.SimpleNamespace()
    r.entry_id = f"http://arxiv.org/abs/{arxiv_id}"
    r.title = title
    r.pdf_url = pdf_url
    r.comment = comment
    r.primary_category = "cs.CL"
    return r



def test_title_search_accepts_high_jaccard(monkeypatch):
    """RET-06 (row 2-03-01): lookup_by_title accepts the top result when Jaccard >= 0.8.

    The cassette ``arxiv_rag_result.atom.xml`` documents the real RAG
    paper's Atom entry. Here we stub the library's results() iterator
    to yield a single ``Result`` whose title closely matches the query.
    jaccard("...", "...") on identical titles is 1.0, well above 0.8.
    """
    client = ArxivClient()
    matching = _make_result_stub(
        arxiv_id="2005.11401v4",
        title="Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        pdf_url="http://arxiv.org/pdf/2005.11401v4",
        comment="Accepted at NeurIPS 2020.",
    )
    monkeypatch.setattr(client._client, "results", lambda search: iter([matching]))

    match = client.lookup_by_title(
        "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
        year=2020,
    )

    assert match is not None
    assert isinstance(match, ArxivMatch)
    assert match.arxiv_id == "2005.11401v4"
    assert match.title.startswith("Retrieval-Augmented Generation")


def test_title_search_rejects_low_jaccard(monkeypatch):
    """RET-06 inverse: a clearly unrelated title is rejected (returns None).

    The ``arxiv_title_search_no_match.atom.xml`` cassette represents
    an arXiv result about Quantum Cryptography — Jaccard overlap with
    "Retrieval-Augmented Generation" is 0. Below 0.8 threshold -> None.
    """
    client = ArxivClient()
    bad = _make_result_stub(
        arxiv_id="9999.0001v1",
        title="Quantum Cryptography in Satellite Mesh Networks",
        pdf_url="http://arxiv.org/pdf/9999.0001v1",
        comment="Unrelated paper.",
    )
    monkeypatch.setattr(client._client, "results", lambda search: iter([bad]))

    match = client.lookup_by_title("Retrieval-Augmented Generation", year=2020)

    assert match is None, "Low-Jaccard result must be rejected, not matched"


def test_uses_export_arxiv_subdomain():
    """RET-06 / P15 (row 2-03-02): query routes to export.arxiv.org AND pdf_url is rewritten.

    Two assertions (belt-and-suspenders):
      1. ``client._client.query_url_format`` points at export.arxiv.org
         (the arxiv 3.0 library's documented override hook).
      2. Any ``ArxivMatch`` produced by ``_to_match`` has its
         ``pdf_url`` rewritten to ``https://export.arxiv.org/...``
         — so when RetrievalAgent later GETs the PDF in plan 02-04,
         the request targets the subdomain, never the primary host.
    """
    client = ArxivClient()

    assert "export.arxiv.org" in client._client.query_url_format, (
        "query_url_format must point to export.arxiv.org subdomain (P15)"
    )
    assert "arxiv.org" not in client._client.query_url_format.replace(
        "export.arxiv.org", ""
    ), "query_url_format must ONLY reference export.arxiv.org, not the primary host"

    r = _make_result_stub(
        arxiv_id="2005.11401v4",
        title="RAG",
        pdf_url="http://arxiv.org/pdf/2005.11401v4",
        comment=None,
    )
    match = ArxivClient._to_match(r)

    assert match.pdf_url.startswith("https://export.arxiv.org/"), (
        f"pdf_url must be rewritten to https://export.arxiv.org/..., "
        f"got {match.pdf_url!r}"
    )



def test_detects_withdrawn_via_comment(monkeypatch):
    """RET-07 (row 2-03-03): a paper whose comment contains 'withdrawn' -> withdrawn=True.

    The ``arxiv_withdrawn.atom.xml`` cassette documents an Atom entry
    whose ``<arxiv:comment>`` contains the phrase "This paper has been
    withdrawn by the author...". Case-insensitive substring match on
    "withdrawn" flips the flag.

    Belt-and-suspenders: also confirms pdf_url is on export.arxiv.org
    even for withdrawn papers (so if RetrievalAgent double-checks via
    a tiny-PDF download, that download still routes through the
    correct subdomain — P15).
    """
    client = ArxivClient()
    withdrawn_stub = _make_result_stub(
        arxiv_id="1234.5678v1",
        title="Dubious Withdrawn Paper",
        pdf_url="http://arxiv.org/pdf/1234.5678v1",
        comment=(
            "This paper has been withdrawn by the author due to a "
            "critical error in the methodology."
        ),
    )
    monkeypatch.setattr(
        client._client, "results", lambda search: iter([withdrawn_stub])
    )

    match = client.lookup_by_id("1234.5678")

    assert match is not None
    assert match.withdrawn is True, "comment containing 'withdrawn' must set withdrawn=True"
    assert match.pdf_url.startswith("https://export.arxiv.org/"), (
        "Even withdrawn matches must have pdf_url on export.arxiv.org subdomain"
    )


def test_detects_not_withdrawn_on_regular_comment(monkeypatch):
    """RET-07 inverse: an 'Accepted at NeurIPS 2020.' comment does NOT trigger withdrawal.

    Regression guard against over-eager substring matching. The word
    "withdrawn" is specific; generic venue-acceptance comments must
    not flip the flag.
    """
    client = ArxivClient()
    ok_stub = _make_result_stub(
        arxiv_id="2005.11401v4",
        title="RAG",
        pdf_url="http://arxiv.org/pdf/2005.11401v4",
        comment="Accepted at NeurIPS 2020.",
    )
    monkeypatch.setattr(client._client, "results", lambda search: iter([ok_stub]))

    match = client.lookup_by_id("2005.11401")

    assert match is not None
    assert match.withdrawn is False


def test_detects_not_withdrawn_on_none_comment(monkeypatch):
    """RET-07 edge case: a None comment must not crash or flag withdrawal.

    Not all arXiv papers have a comment. ``result.comment`` is None in
    that case; the short-circuit ``bool(None and ...)`` must evaluate
    to False without raising.
    """
    client = ArxivClient()
    nocomment = _make_result_stub(
        arxiv_id="0000.0001v1",
        title="No-comment paper",
        pdf_url="http://arxiv.org/pdf/0000.0001v1",
        comment=None,
    )
    monkeypatch.setattr(client._client, "results", lambda search: iter([nocomment]))

    match = client.lookup_by_id("0000.0001")

    assert match is not None
    assert match.withdrawn is False



def test_lookup_by_id_returns_none_when_no_results(monkeypatch):
    """A malformed / nonexistent arxiv_id produces an empty results iterator -> None.

    The library raises StopIteration on next() of an empty generator;
    the implementation catches this and returns None so RetrievalAgent
    can fall back to abstract-only cleanly.
    """
    client = ArxivClient()
    monkeypatch.setattr(client._client, "results", lambda search: iter([]))

    match = client.lookup_by_id("nonexistent.id")

    assert match is None



def test_jaccard_zero_for_disjoint_titles():
    """Completely disjoint token sets -> Jaccard 0.0 (never a false positive)."""
    assert jaccard("foo bar baz", "qux quux corge") == 0.0


def test_jaccard_one_for_identical_titles():
    """Identical titles (after tokenization) -> Jaccard 1.0."""
    assert jaccard(
        "retrieval augmented generation", "retrieval augmented generation"
    ) == 1.0


def test_jaccard_stopwords_ignored():
    """Stopwords ('for', 'the', 'of') drop out of the token set.

    'Models for Research' and 'Models of Research' share all content
    tokens after stopword removal -> Jaccard 1.0.
    """
    assert jaccard("Models for Research", "Models of Research") == 1.0


def test_jaccard_empty_string_returns_zero():
    """Empty or stopword-only inputs produce Jaccard 0.0, not a divide-by-zero."""
    assert jaccard("", "retrieval augmented generation") == 0.0
    assert jaccard("the a of", "the a of") == 0.0


def test_jaccard_partial_overlap_between_zero_and_one():
    """Partial token overlap lands strictly between 0 and 1 (sanity check).

    "RAG paper" vs "Retrieval Augmented Generation paper":
      tokens_a = {"rag", "paper"}
      tokens_b = {"retrieval", "augmented", "generation", "paper"}
      intersection = {"paper"} = 1; union = 5
      -> 1/5 = 0.2
    """
    score = jaccard("RAG paper", "Retrieval Augmented Generation paper")
    assert 0.0 < score < 1.0
    assert abs(score - 0.2) < 1e-9



def test_cassette_fixtures_are_present():
    """Guard — if someone deletes the fixtures, a test fails loudly.

    The cassettes are canonical documentation of the Atom XML shape
    ``arxiv`` 3.0 parses. Since we stub at the ``results()`` boundary,
    tests don't feed these to the library directly, but they're the
    ground truth for the stub structure and must stay in the tree.
    """
    assert "Retrieval-Augmented Generation" in ATOM_GOOD
    assert "withdrawn by the author" in ATOM_WITHDRAWN
    assert "Quantum Cryptography" in ATOM_NOMATCH
