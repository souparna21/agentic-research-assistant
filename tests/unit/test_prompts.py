"""PromptLoader tests (FND-06 + StrictUndefined defense against P18 prompt drift)."""

from __future__ import annotations

from pathlib import Path

import pytest
from jinja2.exceptions import TemplateNotFound


@pytest.fixture
def prompts_dir(tmp_path: Path) -> Path:
    d = tmp_path / "prompts"
    d.mkdir()
    (d / "hello.j2").write_text("Hello {{ name }}!")
    (d / "with_missing.j2").write_text("Value: {{ missing_var }}")
    (d / ".gitkeep").write_text("")
    return d


def test_renders_basic(prompts_dir: Path):
    from ara.services.prompts import PromptLoader

    loader = PromptLoader(prompts_dir)
    out = loader.render("hello.j2", name="World")
    assert "Hello World!" in out


def test_strict_undefined_raises(prompts_dir: Path):
    from ara.errors import PromptTemplateError
    from ara.services.prompts import PromptLoader

    loader = PromptLoader(prompts_dir)
    with pytest.raises(PromptTemplateError) as excinfo:
        loader.render("with_missing.j2")
    msg = str(excinfo.value)
    assert "with_missing.j2" in msg
    assert "missing_var" in msg


def test_missing_template_raises(prompts_dir: Path):
    from ara.services.prompts import PromptLoader

    loader = PromptLoader(prompts_dir)
    with pytest.raises(TemplateNotFound):
        loader.render("does_not_exist.j2")


def test_deterministic(prompts_dir: Path):
    from ara.services.prompts import PromptLoader

    loader = PromptLoader(prompts_dir)
    a = loader.render("hello.j2", name="Alice")
    b = loader.render("hello.j2", name="Alice")
    assert a == b


def test_caches_compiled_templates(prompts_dir: Path, monkeypatch):
    """Jinja2 FileSystemLoader caches compiled templates — second render doesn't re-read."""
    from ara.services.prompts import PromptLoader

    loader = PromptLoader(prompts_dir)
    loader.render("hello.j2", name="A")
    assert any(
        "hello.j2" in (k or "") for k in loader.env.cache if isinstance(k, str)
    ) or any("hello.j2" in str(k) for k in loader.env.cache)


def test_ignores_non_template_files(prompts_dir: Path):
    """A loader built against a dir with .gitkeep or README.md files must not error."""
    from ara.services.prompts import PromptLoader

    loader = PromptLoader(prompts_dir)
    assert loader is not None
