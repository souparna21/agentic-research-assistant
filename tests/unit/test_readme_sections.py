"""Grep tests over README.md — PKG-01 structural gate.

These seven grep tests are the single enforcement layer for the README's
authoritative structure (Overview / Setup / Usage / Architecture / Demo /
Reproducibility / Troubleshooting) and for the required substrings that
document install (`uv sync` + `.env.example` + `ARA_GEMINI_API_KEY`),
reproducibility (`uv.lock`, `.python-version`, `temperature=0`,
`gemini/gemini-2.5-flash`, `run_id`, `git_sha`), architecture (six module
paths under `ara.agents.*`), offline demo (`--cached-only` + the committed
`demo-cache.tar.gz`), and the two documented troubleshooting pain points
(Gemini quota fallback via `--cached-only`, missing `latexmk`).

Plan 04-03 (Phase 4, Wave 2) replaces the Wave-0 placeholder scaffold.
"""

from __future__ import annotations

import re
from pathlib import Path

README = Path(__file__).resolve().parents[2] / "README.md"


def _read() -> str:
    assert README.exists(), f"README.md missing at {README}"
    return README.read_text(encoding="utf-8")


def test_all_seven_sections_present() -> None:
    body = _read()
    required = [
        r"^##\s+Overview\b",
        r"^##\s+Setup\b",
        r"^##\s+Usage\b",
        r"^##\s+Architecture\b",
        r"^##\s+Demo\b",
        r"^##\s+Reproducibility\b",
        r"^##\s+Troubleshooting\b",
    ]
    for pat in required:
        assert re.search(pat, body, re.MULTILINE), f"README missing section: {pat}"


def test_five_cli_examples() -> None:
    body = _read()
    assert re.search(r"\bara\s+run\s+", body), "missing `ara run` example"
    assert re.search(r"\bara\s+stage\s+", body), "missing `ara stage` example"
    assert re.search(r"\bara\s+show\s+", body), "missing `ara show` example"
    assert re.search(r"\bara\s+clean\s+", body), "missing `ara clean` example"
    assert "--cached-only" in body, "missing --cached-only example"


def test_install_instructions_present() -> None:
    body = _read()
    for snippet in (
        "curl -LsSf https://astral.sh/uv/install.sh",
        "uv sync",
        "cp .env.example .env",
        "ARA_GEMINI_API_KEY",
    ):
        assert snippet in body, f"README missing install snippet: {snippet!r}"


def test_reproducibility_substrings() -> None:
    body = _read()
    for snippet in (
        "uv.lock",
        ".python-version",
        "temperature=0",
        "gemini/gemini-2.5-flash",
        "run_id",
        "git_sha",
    ):
        assert snippet in body, f"README missing reproducibility substring: {snippet!r}"


def test_architecture_has_six_module_paths() -> None:
    body = _read()
    for module in (
        "ara.agents.query",
        "ara.agents.retrieval",
        "ara.agents.extraction",
        "ara.agents.indexing",
        "ara.agents.analysis",
        "ara.agents.report",
    ):
        assert module in body, f"README Architecture missing module path: {module!r}"


def test_demo_mode_references_tarball() -> None:
    body = _read()
    assert "demo-cache.tar.gz" in body, "README missing demo-cache.tar.gz reference"
    assert "--cached-only" in body, "README missing --cached-only flag"


def test_troubleshooting_covers_quota_and_latexmk() -> None:
    body = _read()
    match = re.search(
        r"^##\s+Troubleshooting\b(.*?)(?:^##\s|\Z)",
        body,
        re.MULTILINE | re.DOTALL,
    )
    assert match, "Troubleshooting section not found"
    section = match.group(1)
    assert "--cached-only" in section, (
        "Troubleshooting missing --cached-only fallback for quota"
    )
    assert "latexmk" in section, "Troubleshooting missing latexmk guidance"
