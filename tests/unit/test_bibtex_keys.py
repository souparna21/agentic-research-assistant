"""RPT-04/05 make_bibtex_key + resolve_key_collisions (VALIDATION tasks 3-07-02..03)."""
from __future__ import annotations


def test_make_bibtex_key_unidecodes_author() -> None:
    from ara.services.latex import make_bibtex_key

    assert (
        make_bibtex_key("Müller", 2024, "Attention Is All You Need")
        == "muller_2024_attention"
    )


def test_make_bibtex_key_strips_stopwords_from_title() -> None:
    from ara.services.latex import make_bibtex_key

    out = make_bibtex_key("Smith", 2024, "The of an a")
    assert out in {"smith_2024_the", "smith_2024_untitled"}


def test_make_bibtex_key_handles_no_year() -> None:
    from ara.services.latex import make_bibtex_key

    assert make_bibtex_key("Smith", None, "Attention") == "smith_nd_attention"


def test_make_bibtex_key_skips_short_stopwords_picks_significant() -> None:
    from ara.services.latex import make_bibtex_key

    assert (
        make_bibtex_key("Vaswani", 2017, "The Attention Mechanism")
        == "vaswani_2017_attention"
    )


def test_make_bibtex_key_unidecodes_cjk() -> None:
    """Defense-in-depth — non-Latin surnames must produce ASCII output."""
    from ara.services.latex import make_bibtex_key

    key = make_bibtex_key("李", 2024, "Example Title")
    assert key == "li_2024_example"


def test_resolve_key_collisions_suffixes_duplicates() -> None:
    """VALIDATION task 3-07-03 — deterministic _2, _3 suffix chain."""
    from ara.services.latex import resolve_key_collisions

    base = [
        "smith_2024_attention",
        "smith_2024_attention",
        "smith_2024_attention",
        "jones_2023_rag",
    ]
    assert resolve_key_collisions(base) == [
        "smith_2024_attention",
        "smith_2024_attention_2",
        "smith_2024_attention_3",
        "jones_2023_rag",
    ]


def test_resolve_key_collisions_asserts_uniqueness() -> None:
    from ara.services.latex import resolve_key_collisions

    out = resolve_key_collisions(["a", "a", "a", "b", "b", "c"])
    assert len(set(out)) == len(out)


def test_resolve_key_collisions_no_duplicates_passthrough() -> None:
    from ara.services.latex import resolve_key_collisions

    out = resolve_key_collisions(["alpha", "beta", "gamma"])
    assert out == ["alpha", "beta", "gamma"]
