"""Unit tests for S2Client (RET-01, RET-02, RET-04).

Network-blocked: pytest-socket ``--disable-socket`` is active globally
(set by Phase 1's pyproject.toml addopts). Every external call is mocked
via respx. A plain httpx request would fail loudly.

Tenacity retries sleep 2/4/8/16 seconds in production. To keep tests
fast we monkeypatch ``time.sleep`` to a no-op — retries fire immediately.

Library note (``semanticscholar`` 0.12):
    The library uses ``httpx.AsyncClient`` under the hood. respx intercepts
    both sync and async httpx transports. On HTTP 429 the library raises
    ``ConnectionRefusedError`` (NOT ``httpx.HTTPStatusError``). When all
    outer tenacity attempts are exhausted, ``reraise=True`` propagates that
    original ``ConnectionRefusedError`` — which our test asserts for the
    persistent-429 case via ``pytest.raises(ConnectionRefusedError)``.
"""
from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest
import respx

from ara.services.fetchers import S2Client, S2SearchResult

REPO_ROOT = Path(__file__).resolve().parents[2]
S2_200 = json.loads(
    (REPO_ROOT / "tests/fixtures/http/s2_rag_query_200.json").read_text()
)


@pytest.fixture(autouse=True)
def fast_tenacity_sleep(monkeypatch):
    """Make sleep a no-op so tests don't wait 2+4+8 = 14 seconds.

    tenacity internally calls ``time.sleep`` via its helper. Patching the
    ``time`` module's sleep function is the reliable cross-version approach.
    asyncio.sleep is also patched because the library's async transport
    layer may await it.
    """
    import asyncio
    import time as _time

    monkeypatch.setattr(_time, "sleep", lambda *a, **kw: None)

    async def _no_async_sleep(*_a, **_kw):
        return None

    monkeypatch.setattr(asyncio, "sleep", _no_async_sleep)


def _s2_search_route():
    """The S2 search endpoint — respx pattern matches library URL construction.

    The ``semanticscholar`` 0.12 client hits
    ``https://api.semanticscholar.org/graph/v1/paper/search`` with query-string
    params. A regex route lets us match regardless of trailing query string.
    """
    return respx.get(
        url__regex=r"https://api\.semanticscholar\.org/graph/v1/paper/search.*"
    )


@respx.mock
def test_search_returns_results():
    """RET-01: S2Client.search returns a populated list when S2 responds 200."""
    _s2_search_route().mock(return_value=httpx.Response(200, json=S2_200))
    client = S2Client(timeout=5.0)
    results = client.search("RAG", limit=10)
    assert isinstance(results, list)
    assert len(results) == 1
    assert isinstance(results[0], S2SearchResult)
    assert results[0].paper_id == "df2b0e26d0599ce3e70df8a9da02e51594e0e992"
    assert results[0].title.startswith("Retrieval-Augmented Generation")


@respx.mock
def test_response_has_arxiv_id_and_pdf_url():
    """RET-02: externalIds.ArXiv + openAccessPdf.url land on the normalized result."""
    _s2_search_route().mock(return_value=httpx.Response(200, json=S2_200))
    client = S2Client(timeout=5.0)
    results = client.search("RAG", limit=10)
    assert len(results) == 1
    r = results[0]
    assert r.arxiv_id == "2005.11401"
    assert r.pdf_url is not None
    assert r.pdf_url.startswith("https://arxiv.org/")
    assert r.year == 2020
    assert r.venue == "NeurIPS"
    assert "Patrick Lewis" in r.authors
    assert r.doi == "10.48550/arXiv.2005.11401"


@respx.mock
def test_429_then_200_retry_succeeds():
    """RET-04: 2x HTTP 429 then 200 -> tenacity retries, final success, call_count == 3."""
    route = _s2_search_route().mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(200, json=S2_200),
        ]
    )
    client = S2Client(timeout=5.0)
    results = client.search("RAG", limit=10)
    assert route.call_count == 3, (
        f"expected 3 calls (2 retries + 1 success), got {route.call_count}"
    )
    assert len(results) == 1
    assert results[0].arxiv_id == "2005.11401"


@respx.mock
def test_persistent_429_raises_after_four_attempts():
    """RET-04: 4x consecutive 429 -> the last exception reraises; no 5th call.

    The ``semanticscholar`` 0.12 library surfaces HTTP 429 as
    ``ConnectionRefusedError`` (see ApiRequester.py lines 139-140).
    tenacity's ``reraise=True`` propagates that original exception type.
    """
    route = _s2_search_route().mock(
        side_effect=[
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(429, headers={"Retry-After": "0.01"}),
            httpx.Response(429, headers={"Retry-After": "0.01"}),
        ]
    )
    client = S2Client(timeout=5.0)
    with pytest.raises(ConnectionRefusedError):
        client.search("RAG", limit=10)
    assert route.call_count == 4, (
        f"expected 4 attempts (stop_after_attempt(4)), got {route.call_count}"
    )
