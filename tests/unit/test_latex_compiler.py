"""RPT-06/07 compile_latex + count_unresolved_citations (VALIDATION tasks 3-07-04..05).

@pytest.mark.latex — skipped when latexmk binary is missing.
"""
from __future__ import annotations

from pathlib import Path

import pytest


pytestmark = pytest.mark.latex


@pytest.fixture(autouse=True)
def _require_latexmk() -> None:
    from ara.services.latex import latexmk_available

    if not latexmk_available():
        pytest.skip("latexmk not installed")


def test_minimal_latex_compiles(tmp_path: Path) -> None:
    from ara.services.latex import compile_latex, count_unresolved_citations

    tex = tmp_path / "hello.tex"
    tex.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "Hello \\& World 100\\%  % comment\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    pdf = compile_latex(tex, out_dir=tmp_path, timeout_s=60)
    assert pdf.exists() and pdf.suffix == ".pdf"
    assert count_unresolved_citations(pdf) == 0


def test_unresolved_cite_is_caught(tmp_path: Path) -> None:
    """VALIDATION task 3-07-05 — [?] count > 0 for undefined \\cite."""
    from ara.services.latex import compile_latex, count_unresolved_citations

    tex = tmp_path / "broken.tex"
    tex.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\cite{does_not_exist}\n"
        "\\bibliographystyle{plain}\n"
        "\\bibliography{empty}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    (tmp_path / "empty.bib").write_text("", encoding="utf-8")
    pdf = compile_latex(tex, out_dir=tmp_path, timeout_s=60)
    assert count_unresolved_citations(pdf) >= 1


def test_nonzero_exit_raises_with_tail(tmp_path: Path) -> None:
    from ara.services.latex import LatexCompileError, compile_latex

    tex = tmp_path / "broken_syntax.tex"
    tex.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "\\undefinedcommand\n",
        encoding="utf-8",
    )
    with pytest.raises(LatexCompileError):
        compile_latex(tex, out_dir=tmp_path, timeout_s=60)


def test_timeout_raises(tmp_path: Path) -> None:
    """Simulate timeout via subprocess monkeypatch — no real sleep."""
    import subprocess as _sp

    import ara.services.latex as lat
    from ara.services.latex import LatexCompileError

    def raise_timeout(*a, **kw):
        raise _sp.TimeoutExpired(cmd=a[0] if a else "latexmk", timeout=1)

    tex = tmp_path / "x.tex"
    tex.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8"
    )
    mp = pytest.MonkeyPatch()
    try:
        mp.setattr(_sp, "run", raise_timeout)
        with pytest.raises(LatexCompileError, match="timeout"):
            lat.compile_latex(tex, out_dir=tmp_path, timeout_s=1)
    finally:
        mp.undo()


def test_missing_latexmk_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Simulate missing binary via latexmk_available monkeypatch."""
    import ara.services.latex as lat
    from ara.services.latex import LatexCompileError, compile_latex

    monkeypatch.setattr(lat, "latexmk_available", lambda: False)
    tex = tmp_path / "x.tex"
    tex.write_text(
        "\\documentclass{article}\\begin{document}x\\end{document}\n", encoding="utf-8"
    )
    with pytest.raises(LatexCompileError, match="not found"):
        compile_latex(tex, out_dir=tmp_path, timeout_s=10)
