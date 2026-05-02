"""Jinja2 PromptLoader with StrictUndefined.

Templates live at `prompts/*.j2` (or the configured directory). Rendering
with a missing variable raises PromptTemplateError — never a silent blank.

The Environment cache is on by default; re-rendering the same template
does not re-read from disk.

Two separate loaders live here:
  - :class:`PromptLoader` uses STANDARD Jinja2 delimiters (``{% %}`` / ``{{ }}``)
    for LLM prompts.
  - :class:`LatexTemplateLoader` uses LaTeX-safe custom delimiters
    (``<% %>`` / ``<< >>`` / ``<# #>``) for the report skeleton (plan 03-08).

Two Environment instances keep the delimiter settings isolated — neither
loader can leak its delimiter configuration into the other (pitfall P10
defense).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined
from jinja2.exceptions import UndefinedError

from ara.errors import PromptTemplateError
from ara.services.latex import latex_escape


class PromptLoader:
    """Load and render Jinja2 prompt templates from a directory.

    Use STANDARD Jinja2 delimiters (`{% %}` / `{{ }}`) for LLM prompts.
    LaTeX-safe delimiters (`<% %>` / `<< >>`) are reserved for the report
    template in Phase 3 — those land in a separate loader when needed.
    """

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = Path(prompts_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            undefined=StrictUndefined,
            keep_trailing_newline=True,
            autoescape=False,
        )

    def render(self, template_name: str, **ctx: Any) -> str:
        """Render a template. Raises PromptTemplateError on missing vars."""
        tmpl = self.env.get_template(template_name)
        try:
            return tmpl.render(**ctx)
        except UndefinedError as e:
            raise PromptTemplateError(
                f"Template {template_name!r} failed: {e}. "
                f"Hint: pass the missing variable as a kwarg to PromptLoader.render()."
            ) from e


class LatexTemplateLoader:
    """Jinja2 loader with LaTeX-safe delimiters (``<% %>``, ``<< >>``, ``<# #>``).

    Pairs with :class:`PromptLoader` (standard delimiters for LLM prompts).
    Separate Environment instances avoid delimiter bleed. Reference pattern:
    the community-verified LaTeX-Jinja2 recipe (mbr/latex/blob/master/latex/jinja2.py).

    The ``latex_escape`` filter from :mod:`ara.services.latex` is registered
    on ``env.filters`` so every ``<< paper.* | latex_escape >>`` substitution
    in ``report_skeleton.tex.j2`` handles the 10 LaTeX specials (RPT-03).
    """

    def __init__(self, prompts_dir: Path) -> None:
        self.prompts_dir = Path(prompts_dir)
        self.env = Environment(
            loader=FileSystemLoader(str(self.prompts_dir)),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
            keep_trailing_newline=True,
            block_start_string="<%",
            block_end_string="%>",
            variable_start_string="<<",
            variable_end_string=">>",
            comment_start_string="<#",
            comment_end_string="#>",
        )
        self.env.filters["latex_escape"] = latex_escape

    def render(self, template_name: str, **ctx: Any) -> str:
        """Render a LaTeX template. Raises PromptTemplateError on missing vars."""
        tmpl = self.env.get_template(template_name)
        try:
            return tmpl.render(**ctx)
        except UndefinedError as e:
            raise PromptTemplateError(
                f"LaTeX template {template_name!r} failed: {e}. "
                f"Hint: pass the missing variable as a kwarg to "
                f"LatexTemplateLoader.render()."
            ) from e
