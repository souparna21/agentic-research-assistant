"""Typer CLI: `ara run` / `ara stage` / `ara show` / `ara clean`.

CRITICAL: This module must be fast to import. NO top-level imports of
torch, sentence_transformers, faiss, pymupdf, litellm, or ara.orchestrator.
All heavy imports happen inside command function bodies (lazy-load).

Target: `ara --help` returns in < 500ms of pure Python startup after
bytecode warm-up. The `uv run` wrapper adds its own ~200-400ms on top;
budget for the full `uv run ara --help` latency is < 1.5s wall-clock.

Why this matters: `ara --help` being slow is the earliest, loudest signal
that a heavy dependency (torch, sentence-transformers, litellm) leaked
into the module-level imports of `cli.py`. Enforced by
`tests/unit/test_cli.py::test_cli_has_no_heavy_top_level_imports`
(AST scan) and `test_help_is_fast` (wall-clock timing).
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import typer

from ara.config import load_settings
from ara.runids import make_run_id, read_git_sha

app = typer.Typer(
    add_completion=False,
    rich_markup_mode="rich",
    help="Agentic Research Assistant — multi-agent RAG literature review pipeline.",
)


@app.callback()
def main(
    ctx: typer.Context,
    log_level: str = typer.Option(
        "INFO", "--log-level", help="DEBUG / INFO / WARN / ERROR"
    ),
    cached_only: bool = typer.Option(
        False, "--cached-only", help="Offline demo mode (honored in Phase 4)"
    ),
) -> None:
    """Global options; subcommands read these via ctx.obj."""
    settings = load_settings()
    settings = settings.model_copy(
        update={"log_level": log_level, "cached_only": cached_only}
    )
    ctx.obj = settings


@app.command()
def run(
    ctx: typer.Context,
    question: str = typer.Argument(..., help="Research question to run the full pipeline on"),
    run_id: str | None = typer.Option(None, "--run-id", help="Resume an existing run"),
    year_since: int | None = typer.Option(
        None,
        "--year-since",
        help="Filter Semantic Scholar results to papers from this year or later",
    ),
    min_citations: int | None = typer.Option(
        None,
        "--min-citations",
        help="Filter Semantic Scholar results to papers with at least N citations",
    ),
) -> None:
    """Run the full pipeline (all 6 stages)."""
    from ara.orchestrator import Orchestrator  # noqa: PLC0415

    settings = ctx.obj
    overrides: dict = {}
    if year_since is not None:
        overrides["retrieval_year_since"] = year_since
    if min_citations is not None:
        overrides["retrieval_min_citations"] = min_citations
    if overrides:
        settings = settings.model_copy(update=overrides)

    if settings.cached_only:
        from ara.demo_cache import extract_demo_cache  # noqa: PLC0415

        if run_id is not None:
            typer.secho(
                f"--cached-only: ignoring --run-id {run_id!r} "
                f"(cached demo run_id wins).",
                fg=typer.colors.YELLOW,
                err=True,
            )
        rid = extract_demo_cache(Path(settings.runs_dir))
        typer.echo(f"Using cached demo run {rid}")
    else:
        rid = run_id or make_run_id(question)

    stages = _build_live_stages(settings) if not settings.cached_only else None
    orch = Orchestrator(
        settings, rid, git_sha=read_git_sha(), question=question, stages=stages
    )
    orch.run()


def _build_live_stages(settings):  # noqa: ANN001, ANN202 — internal CLI helper
    """Construct the six real agents wired to live services.

    Lazy-imports each agent + service module so that `ara --help` and
    `ara --cached-only` stay fast and don't pull torch / faiss / litellm
    at import time.
    """
    from ara.agents.analysis import AnalysisAgent  # noqa: PLC0415
    from ara.agents.extraction import ExtractionAgent  # noqa: PLC0415
    from ara.agents.indexing import IndexingAgent  # noqa: PLC0415
    from ara.agents.query import QueryAgent  # noqa: PLC0415
    from ara.agents.report import ReportAgent  # noqa: PLC0415
    from ara.agents.retrieval import RetrievalAgent  # noqa: PLC0415
    from ara.services.embed import MiniLMEmbedder  # noqa: PLC0415
    from ara.services.fetchers import ArxivClient, PdfDownloader, S2Client  # noqa: PLC0415
    from ara.services.llm import LiteLLMClient  # noqa: PLC0415
    from ara.services.pdf import PdfExtractor  # noqa: PLC0415
    from ara.services.prompts import LatexTemplateLoader, PromptLoader  # noqa: PLC0415

    llm = LiteLLMClient(model=settings.llm_model, max_tokens=settings.llm_max_tokens)
    prompts_dir = Path(settings.prompts_dir)
    prompts = PromptLoader(prompts_dir=prompts_dir)
    latex_templates = LatexTemplateLoader(prompts_dir=prompts_dir)
    embedder = MiniLMEmbedder()

    query = QueryAgent(llm=llm, prompts=prompts)
    retrieval = RetrievalAgent(
        s2=S2Client(),
        arxiv=ArxivClient(),
        downloader=PdfDownloader(),
        year_since=settings.retrieval_year_since,
        min_citation_count=settings.retrieval_min_citations,
    )
    extraction = ExtractionAgent(pdf_extractor=PdfExtractor())
    indexing = IndexingAgent(embedder=embedder)
    analysis = AnalysisAgent(llm=llm, prompts=prompts)
    report = ReportAgent(latex_templates=latex_templates)

    return [query, retrieval, extraction, indexing, analysis, report]


@app.command()
def stage(
    ctx: typer.Context,
    name: str = typer.Argument(
        ..., help="query | retrieval | extraction | indexing | analysis | report"
    ),
    run_id: str = typer.Option(..., "--run-id", help="Target run id"),
    year_since: int | None = typer.Option(
        None,
        "--year-since",
        help="Filter Semantic Scholar results to papers from this year or later",
    ),
    min_citations: int | None = typer.Option(
        None,
        "--min-citations",
        help="Filter Semantic Scholar results to papers with at least N citations",
    ),
) -> None:
    """Run a single named stage against an existing run."""
    from ara.orchestrator import Orchestrator  # noqa: PLC0415

    settings = ctx.obj
    overrides: dict = {}
    if year_since is not None:
        overrides["retrieval_year_since"] = year_since
    if min_citations is not None:
        overrides["retrieval_min_citations"] = min_citations
    if overrides:
        settings = settings.model_copy(update=overrides)

    orch = Orchestrator(settings, run_id, git_sha=read_git_sha())
    orch.run_single_stage(name)


@app.command()
def show(
    ctx: typer.Context,
    run_id: str = typer.Argument(..., help="Run id to inspect"),
) -> None:
    """Print a summary of `runs/<run_id>/state.json`."""
    settings = ctx.obj
    path = Path(settings.runs_dir) / run_id / "state.json"
    if not path.exists():
        typer.secho(f"Run not found: {path}", fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2)
    data = json.loads(path.read_text(encoding="utf-8"))
    typer.echo(f"run_id:   {data.get('run_id')}")
    typer.echo(f"question: {data.get('question')}")
    typer.echo(f"config:   {json.dumps(data.get('config_snapshot', {}), indent=2)}")
    typer.echo(f"papers:   {len(data.get('papers', []))}")
    typer.echo(f"stages:   {len(data.get('stage_reports', []))}")


@app.command()
def clean(
    ctx: typer.Context,
    run_id: str | None = typer.Option(None, "--run-id", help="Delete a specific run"),
    all_runs: bool = typer.Option(
        False, "--all", help="Delete ALL runs (requires confirmation)"
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt"),
) -> None:
    """Delete pipeline runs. Always prompts unless --yes."""
    settings = ctx.obj
    runs_dir = Path(settings.runs_dir)

    if not run_id and not all_runs:
        typer.secho(
            "Specify either --run-id <id> or --all.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)

    if run_id and all_runs:
        typer.secho(
            "--run-id and --all are mutually exclusive.", fg=typer.colors.RED, err=True
        )
        raise typer.Exit(code=2)

    if all_runs:
        if not yes:
            confirm = typer.prompt(
                f"Delete ALL runs under {runs_dir}? Type 'yes' to confirm",
                default="n",
            )
            if confirm.strip().lower() not in ("y", "yes"):
                typer.echo("Cancelled.")
                return
        if runs_dir.exists():
            for entry in runs_dir.iterdir():
                if entry.is_dir():
                    shutil.rmtree(entry)
        typer.echo(f"Cleaned all runs under {runs_dir}.")
        return

    target = runs_dir / run_id  # type: ignore[arg-type]
    if not target.exists():
        typer.secho(f"No such run: {target}", fg=typer.colors.YELLOW, err=True)
        raise typer.Exit(code=1)
    if not yes:
        confirm = typer.prompt(
            f"Delete {target}? Type 'yes' to confirm", default="n"
        )
        if confirm.strip().lower() not in ("y", "yes"):
            typer.echo("Cancelled.")
            return
    shutil.rmtree(target)
    typer.echo(f"Deleted {target}.")


if __name__ == "__main__":
    app()
