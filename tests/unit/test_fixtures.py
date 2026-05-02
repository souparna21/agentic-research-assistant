"""Fixture validation tests (FND-11).

Guarantees the offline fixture corpus committed under ``tests/fixtures/`` is
structurally valid against the frozen :class:`ara.state.Paper` and
:class:`ara.state.PipelineState` schemas, loadable by :class:`ara.services.llm.FakeLLM`,
and small enough to live in git (each file under 500KB per pre-commit's
``check-added-large-files --maxkb=500`` hook).

These tests are the canary for schema drift: any change to ``src/ara/state.py``
that breaks fixture deserialization will flip one of these tests red before
the smoke test (plan 09) even runs.
"""

from __future__ import annotations

import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FIXTURES = REPO / "tests" / "fixtures"


def test_two_paper_corpus_loads():
    """Both paper JSONs deserialize into Paper and cover both pdf + abstract-only paths."""
    from ara.state import Paper

    with_pdf = Paper.model_validate_json((FIXTURES / "papers" / "paper_with_pdf.json").read_text())
    abstract_only = Paper.model_validate_json(
        (FIXTURES / "papers" / "paper_abstract_only.json").read_text()
    )

    assert with_pdf.pdf_source == "arxiv"
    assert with_pdf.full_text_available is True
    assert with_pdf.pdf_path is not None
    assert (REPO / with_pdf.pdf_path).exists(), "Referenced PDF file missing"

    assert abstract_only.pdf_source == "abstract-only"
    assert abstract_only.full_text_available is False
    assert abstract_only.pdf_path is None
    assert abstract_only.fallback_to_abstract is True


def test_pdf_fixture_under_size_budget():
    """pre-commit's check-added-large-files --maxkb=500 enforces this at commit time,
    but we also assert it in-test so a developer who forgot to run pre-commit still catches it."""
    pdf = FIXTURES / "papers" / "paper_with_pdf.pdf"
    size = pdf.stat().st_size
    assert size > 0, "paper_with_pdf.pdf is empty"
    assert size < 500 * 1024, f"paper_with_pdf.pdf is {size} bytes (budget: 500KB)"


def test_fake_llm_fixture_is_valid_json():
    from ara.services.llm import FakeLLM

    fixture = FIXTURES / "fake_llm" / "smoke.json"
    assert fixture.exists()
    data = json.loads(fixture.read_text())
    assert isinstance(data, dict), "FakeLLM fixture must be a JSON object"
    fake = FakeLLM(fixture)
    assert fake is not None


def test_golden_state_validates_pipelinestate():
    from ara.state import PipelineState

    golden = json.loads((FIXTURES / "golden_state.json").read_text())
    state = PipelineState.model_validate(golden)
    assert state.run_id
    assert state.question




def test_adversarial_pdf_exists_and_parseable():
    """02-00: adversarial_prompt_injection.pdf exists, loads via pymupdf,
    contains 'PWN' pre-redaction (EXT-02 is what strips it, plan 02-05).
    """
    import pymupdf

    pdf = FIXTURES / "papers" / "adversarial_prompt_injection.pdf"
    assert pdf.exists(), f"Missing: {pdf}"
    assert pdf.stat().st_size < 100_000, "Adversarial PDF should be < 100KB"
    doc = pymupdf.open(str(pdf))
    assert len(doc) == 1
    raw = doc[0].get_text()
    assert "PWN" in raw, (
        "Adversarial fixture must contain 'PWN' pre-redaction — the defense "
        "in plan 02-05 is what strips it; the fixture only sets up the attack."
    )
    doc.close()


def test_s2_rag_query_200_json_is_valid():
    """02-00: S2 happy-path cassette parses and carries expected arXiv id."""
    j = json.loads((FIXTURES / "http" / "s2_rag_query_200.json").read_text())
    assert (
        j["data"][0]["paperId"]
        == "df2b0e26d0599ce3e70df8a9da02e51594e0e992"  # pragma: allowlist secret
    )
    assert j["data"][0]["externalIds"]["ArXiv"] == "2005.11401"
    assert j["data"][0]["openAccessPdf"]["url"].startswith("https://arxiv.org/")


def test_s2_zero_results_json_is_valid():
    """02-00: S2 zero-results cassette parses with empty data list."""
    j = json.loads((FIXTURES / "http" / "s2_zero_results.json").read_text())
    assert j["total"] == 0
    assert j["data"] == []


def test_arxiv_cassettes_are_valid_xml():
    """02-00: All three arXiv Atom cassettes parse as well-formed XML."""
    import xml.etree.ElementTree as ET

    for name in (
        "arxiv_rag_result.atom.xml",
        "arxiv_withdrawn.atom.xml",
        "arxiv_title_search_no_match.atom.xml",
    ):
        ET.fromstring((FIXTURES / "http" / name).read_text())


def test_arxiv_withdrawn_cassette_has_withdrawn_comment():
    """02-00: arxiv_withdrawn.atom.xml's arxiv:comment contains 'withdrawn'
    (RetrievalAgent filter key, plan 02-04)."""
    xml = (FIXTURES / "http" / "arxiv_withdrawn.atom.xml").read_text()
    assert "withdrawn" in xml.lower()


def test_query_scenarios_is_valid_json():
    """02-00: query_scenarios.json parses as a JSON object. Plan 02-01
    populates it with prompt-hash-keyed canned responses during its TDD cycle.
    """
    j = json.loads((FIXTURES / "fake_llm" / "query_scenarios.json").read_text())
    assert isinstance(j, dict)
