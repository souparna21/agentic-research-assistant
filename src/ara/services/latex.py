"""LaTeX utilities — escape filter, BibTeX key helpers, latexmk driver, [?] scan.

Composed by ReportAgent (plan 03-08):
  - latex_escape: Jinja filter over every << paper.* >> / << state.* >> substitution (RPT-03).
  - make_bibtex_key + resolve_key_collisions: deterministic unique BibTeX keys (RPT-04/05).
  - compile_latex: latexmk -pdf -halt-on-error subprocess driver (RPT-06).
  - count_unresolved_citations: pymupdf-based [?] scan on compiled PDF (RPT-07).
"""
from __future__ import annotations

import re
import shutil
import subprocess
from collections import Counter
from pathlib import Path

import pymupdf
from unidecode import unidecode

from ara.errors import LatexCompileError, UnresolvedCitationError


_SENTINEL_BACKSLASH = "\uE000"
_SENTINEL_TILDE = "\uE001"
_SENTINEL_CARET = "\uE002"

_LATEX_REPLACEMENTS: list[tuple[str, str]] = [
    ("\\", _SENTINEL_BACKSLASH),
    ("&", r"\&"),
    ("%", r"\%"),
    ("$", r"\$"),
    ("#", r"\#"),
    ("_", r"\_"),
    ("{", r"\{"),
    ("}", r"\}"),
    ("~", _SENTINEL_TILDE),
    ("^", _SENTINEL_CARET),
]

_SENTINEL_EXPANSIONS: list[tuple[str, str]] = [
    (_SENTINEL_BACKSLASH, r"\textbackslash{}"),
    (_SENTINEL_TILDE, r"\textasciitilde{}"),
    (_SENTINEL_CARET, r"\textasciicircum{}"),
]


def latex_escape(value: object) -> str:
    """Escape LaTeX specials in ``value``. Non-strings are stringified first.

    Handles all 10 LaTeX specials in a two-pass order: first a sentinel pass for
    ``\\``, ``~``, and ``^`` (whose replacement output contains literal ``{}``),
    then the ``{`` / ``}`` rules, then the sentinels are expanded to their final
    LaTeX command form. This preserves the backslash-first invariant while
    preventing the ``\\textbackslash{}`` output from being re-escaped into
    ``\\textbackslash\\{\\}``.

    Reference: https://en.wikibooks.org/wiki/LaTeX/Basics#Reserved_Characters
    """
    text = str(value)
    for src, dst in _LATEX_REPLACEMENTS:
        text = text.replace(src, dst)
    for src, dst in _SENTINEL_EXPANSIONS:
        text = text.replace(src, dst)
    return text



_STOPWORDS: frozenset[str] = frozenset(
    {"a", "an", "the", "of", "for", "and", "or", "on", "in", "to", "with", "via", "using"}
)
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


def _normalize_component(s: str) -> str:
    """Lowercase, ASCII-transliterate, strip non-alphanumerics."""
    return _NON_ALNUM.sub("", unidecode(s).lower())


def _first_significant_title_word(title: str) -> str:
    """First non-stopword word of the title (lowercased, ASCII).

    Fallback to the first raw word when every word is a stopword; final
    fallback to ``"untitled"`` when the title has no alphanumerics.
    """
    words = re.findall(r"[A-Za-z0-9]+", unidecode(title).lower())
    for w in words:
        if w not in _STOPWORDS and len(w) >= 2:
            return w
    return words[0] if words else "untitled"


def make_bibtex_key(first_author_surname: str, year: int | None, title: str) -> str:
    """Canonical key: ``<author_surname>_<year>_<first_word>`` — ASCII + lowercase + alnum only."""
    surname_part = _normalize_component(first_author_surname) or "unknown"
    year_part = str(year) if year else "nd"
    word_part = _normalize_component(_first_significant_title_word(title)) or "untitled"
    return f"{surname_part}_{year_part}_{word_part}"


def resolve_key_collisions(base_keys: list[str]) -> list[str]:
    """Deduplicate via ``_2``, ``_3``, ... suffixes in first-seen order.

    Invariant: ``len(set(return_value)) == len(return_value)``. Raises
    ValueError as a defensive tripwire if the output still has duplicates
    (should be impossible).
    """
    resolved: list[str] = []
    seen: Counter[str] = Counter()
    for key in base_keys:
        n = seen[key]
        seen[key] += 1
        resolved.append(key if n == 0 else f"{key}_{n + 1}")
    if len(set(resolved)) != len(resolved):
        raise ValueError(
            "resolve_key_collisions produced duplicates; this should be impossible"
        )
    return resolved




def latexmk_available() -> bool:
    """True if ``latexmk`` is on PATH. @pytest.mark.latex tests skip when False."""
    return shutil.which("latexmk") is not None


def compile_latex(
    tex_path: Path,
    out_dir: Path,
    *,
    timeout_s: int = 120,
) -> Path:
    """Compile ``tex_path`` to PDF inside ``out_dir`` via latexmk.

    Returns the absolute path to the resulting ``.pdf``. Raises LatexCompileError
    on any failure (non-zero exit, timeout, or binary missing).
    """
    if not latexmk_available():
        raise LatexCompileError(
            "latexmk not found on PATH; install `texlive-latex-recommended latexmk`."
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        "latexmk",
        "-pdf",
        "-halt-on-error",
        "-interaction=nonstopmode",
        f"-output-directory={out_dir}",
        str(tex_path),
    ]
    try:
        result = subprocess.run(
            cmd,
            cwd=tex_path.parent,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        raise LatexCompileError(
            f"latexmk exceeded {timeout_s}s timeout on {tex_path.name}"
        ) from exc

    if result.returncode != 0:
        tail = (result.stdout or "")[-2000:] + "\n---\n" + (result.stderr or "")[-2000:]
        raise LatexCompileError(
            f"latexmk failed (exit {result.returncode}) for {tex_path.name}:\n{tail}"
        )

    pdf_path = out_dir / (tex_path.stem + ".pdf")
    if not pdf_path.exists():
        raise LatexCompileError(f"latexmk returned 0 but produced no PDF at {pdf_path}")
    return pdf_path




def count_unresolved_citations(pdf_path: Path) -> int:
    """Return count of ``[?]`` occurrences across all pages of ``pdf_path``.

    LaTeX renders an undefined ``\\cite`` key as ``[?]`` — scanning for this
    substring catches BibTeX errors that latexmk itself did not surface as a
    non-zero exit.
    """
    doc = pymupdf.open(str(pdf_path))
    try:
        total = 0
        for page in doc:
            total += page.get_text().count("[?]")
        return total
    finally:
        doc.close()


def assert_no_unresolved_citations(pdf_path: Path) -> None:
    """Raise ``UnresolvedCitationError`` when ``pdf_path`` contains any ``[?]`` markers."""
    n = count_unresolved_citations(pdf_path)
    if n > 0:
        raise UnresolvedCitationError(
            f"Compiled PDF {pdf_path.name} has {n} unresolved-citation markers ([?]). "
            f"Check references.bib for missing keys."
        )


__all__ = [
    "LatexCompileError",
    "UnresolvedCitationError",
    "assert_no_unresolved_citations",
    "compile_latex",
    "count_unresolved_citations",
    "latex_escape",
    "latexmk_available",
    "make_bibtex_key",
    "resolve_key_collisions",
]
