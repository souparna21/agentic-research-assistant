"""RPT-03 latex_escape unit tests (VALIDATION task 3-07-01)."""
from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("", ""),
        ("plain text", "plain text"),
        ("R&D 100%", r"R\&D 100\%"),
        ("$cost$", r"\$cost\$"),
        ("#tag", r"\#tag"),
        ("under_score", r"under\_score"),
        ("~home~", r"\textasciitilde{}home\textasciitilde{}"),
        ("a^b", r"a\textasciicircum{}b"),
        ("{a}", r"\{a\}"),
        ("\\", r"\textbackslash{}"),
        (
            "& % $ _ # ^ ~ \\ { }",
            r"\& \% \$ \_ \# \textasciicircum{} \textasciitilde{} \textbackslash{} \{ \}",
        ),
    ],
)
def test_latex_escape_handles_all_specials(raw: str, expected: str) -> None:
    from ara.services.latex import latex_escape

    assert latex_escape(raw) == expected


def test_latex_escape_backslash_then_brace_composition() -> None:
    """Backslash substituted FIRST so its own { / } are NOT double-escaped."""
    from ara.services.latex import latex_escape

    assert latex_escape("\\") == r"\textbackslash{}"
    assert "\\{" not in latex_escape("\\")


def test_latex_escape_accepts_non_string_input() -> None:
    from ara.services.latex import latex_escape

    assert latex_escape(42) == "42"
    assert latex_escape(3.14) == "3.14"
    assert latex_escape(None) == "None"
