"""Six stub stages for Phase 1's end-to-end smoke test.

Each stub:
    - Has a canonical `name` (query/retrieval/extraction/indexing/analysis/report).
    - Writes ONLY to fields it owns per FIELD_OWNERS (enforced by the orchestrator
      at run time — any cross-ownership write raises FieldOwnershipViolation).
    - Has an `output_artifact_path(state)` used for delete-to-invalidate idempotency.
    - Uses small fixed data — the whole smoke pipeline must run in <5s of stub work.

In Phase 2, StubQueryStage / StubRetrievalStage / StubExtractionStage are replaced
by agents/query.py / retrieval.py / extraction.py. Same for Phase 3 and the last
three stubs. This file is DELETED at the end of Phase 3.

Note on FIELD_OWNERS and the `papers` list (Pattern 7 in 01-RESEARCH.md):
`StubExtractionStage` does NOT mutate the `papers` list in place. The
orchestrator's snapshot-diff single-writer check treats `papers` as
retrieval-owned at the top level, so any post-snapshot mutation (even of
nested Paper fields) would be detected as a cross-ownership write. Phase 1's
extraction stage therefore only writes `extraction_stats`; nested per-Paper
extraction metadata is reserved for Phase 2 extractors, which will extend
FIELD_OWNERS to sub-object fields.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ara.agents.base import Stage
from ara.state import ExtractionStats, Paper, PipelineState, RetrievalStats, StageReport

FIXTURE_DIR = Path(__file__).resolve().parents[3] / "tests" / "fixtures"




class StubQueryStage:
    """Stage 1 stub — writes canned search_terms."""

    name = "query"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "query.marker"

    def run(self, state: PipelineState) -> PipelineState:
        state.search_terms = [
            "retrieval-augmented generation",
            "RAG",
            "vector retrieval",
        ]
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=1,
                outputs_count=len(state.search_terms),
            )
        )
        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state




class StubRetrievalStage:
    """Stage 2 stub — loads paper_with_pdf.json + paper_abstract_only.json."""

    name = "retrieval"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "retrieval.marker"

    def run(self, state: PipelineState) -> PipelineState:
        papers_dir = FIXTURE_DIR / "papers"
        with_pdf = Paper.model_validate_json(
            (papers_dir / "paper_with_pdf.json").read_text(encoding="utf-8")
        )
        abstract_only = Paper.model_validate_json(
            (papers_dir / "paper_abstract_only.json").read_text(encoding="utf-8")
        )
        state.papers = [with_pdf, abstract_only]
        state.retrieval_stats = RetrievalStats(
            search_terms_submitted=len(state.search_terms),
            s2_total_results=2,
            arxiv_fallback_hits=1,
            abstract_only_count=1,
        )
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=len(state.search_terms),
                outputs_count=len(state.papers),
            )
        )
        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state




class StubExtractionStage:
    """Stage 3 stub — writes only extraction_stats; Paper records untouched."""

    name = "extraction"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "extraction.marker"

    def run(self, state: PipelineState) -> PipelineState:
        ok = sum(1 for p in state.papers if p.pdf_path)
        state.extraction_stats = ExtractionStats(
            pdfs_attempted=len(state.papers),
            pdfs_extracted_ok=ok,
        )
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=len(state.papers),
                outputs_count=ok,
            )
        )
        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state




class StubIndexingStage:
    """Stage 4 stub — sets chunks_count, index_path, index_meta."""

    name = "indexing"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "indexing.marker"

    def run(self, state: PipelineState) -> PipelineState:
        state.chunks_count = 10
        state.index_path = f"runs/{state.run_id}/index/faiss.index"
        state.index_meta = {
            "dim": 384,
            "model": "all-MiniLM-L6-v2",
            "built_at": "GOLDEN-TIMESTAMP",
        }
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=len(state.papers),
                outputs_count=state.chunks_count,
            )
        )
        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state




class StubAnalysisStage:
    """Stage 5 stub — sets summaries_dir, comparison_path, gaps_path."""

    name = "analysis"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "analysis.marker"

    def run(self, state: PipelineState) -> PipelineState:
        state.summaries_dir = f"runs/{state.run_id}/summaries"
        state.comparison_path = f"runs/{state.run_id}/comparison.json"
        state.gaps_path = f"runs/{state.run_id}/gaps.json"
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=len(state.papers),
                outputs_count=3,
            )
        )
        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state




class StubReportStage:
    """Stage 6 stub — sets report_path."""

    name = "report"

    def output_artifact_path(self, state: PipelineState) -> Path:
        return Path(state.runs_dir) / state.run_id / "report.marker"

    def run(self, state: PipelineState) -> PipelineState:
        state.report_path = f"runs/{state.run_id}/report.tex"
        state.stage_reports.append(
            StageReport(
                stage=self.name,
                inputs_count=1,
                outputs_count=1,
            )
        )
        marker = self.output_artifact_path(state)
        marker.parent.mkdir(parents=True, exist_ok=True)
        marker.write_text("ok")
        return state




def build_pure_stub_stages() -> list[Stage]:
    """All six stages are Phase-1 stubs. Legacy smoke-test entry point.

    Useful for tests that need purely deterministic, zero-LLM, zero-disk-IO
    stage behavior (the original Phase-1 smoke shape).
    """
    return [
        StubQueryStage(),
        StubRetrievalStage(),
        StubExtractionStage(),
        StubIndexingStage(),
        StubAnalysisStage(),
        StubReportStage(),
    ]


def build_stub_stages(*, fake_llm_fixture: Path | None = None) -> list[Stage]:
    """Smoke-test factory — all six stages wired as REAL agents as of Phase 3 close.

    - query / retrieval / extraction: Phase 2 real agents (plans 02-03..06),
      with RetrievalAgent given in-memory fakes for S2 / Arxiv / PdfDownloader
      so the Phase-1 2-paper corpus drives the smoke (paper_with_pdf with a
      readable PDF on disk; paper_abstract_only with no PDF). ExtractionAgent
      runs against the real PdfExtractor.
    - indexing / analysis / report: Phase 3 real agents (plans 03-04, 03-06,
      03-08). AnalysisAgent + QueryAgent share the same ``FakeLLM`` instance
      loaded from ``fake_llm_fixture``; ReportAgent uses ``LatexTemplateLoader``.

    The ``Stub*Stage`` classes remain in this module ONLY for
    :func:`build_pure_stub_stages` (Phase-1 legacy smoke path — all six
    stages stubbed, zero LLM / zero disk-IO beyond markers).

    Defaults to ``fake_llm_fixture = tests/fixtures/fake_llm/smoke.json`` so
    the existing smoke test keeps passing without code changes.
    """
    from ara.agents.analysis import AnalysisAgent
    from ara.agents.extraction import ExtractionAgent
    from ara.agents.indexing import IndexingAgent
    from ara.agents.query import QueryAgent
    from ara.agents.report import ReportAgent
    from ara.agents.retrieval import RetrievalAgent
    from ara.services.embed import MiniLMEmbedder
    from ara.services.llm import FakeLLM
    from ara.services.pdf import PdfExtractor
    from ara.services.prompts import LatexTemplateLoader, PromptLoader

    if fake_llm_fixture is None:
        fake_llm_fixture = FIXTURE_DIR / "fake_llm" / "smoke.json"
    prompts_dir = Path(__file__).resolve().parents[3] / "prompts"
    papers_dir = FIXTURE_DIR / "papers"

    llm = FakeLLM(fixture_path=fake_llm_fixture)
    prompts = PromptLoader(prompts_dir=prompts_dir)
    latex_templates = LatexTemplateLoader(prompts_dir=prompts_dir)
    embedder = MiniLMEmbedder()

    query = QueryAgent(llm=llm, prompts=prompts)
    retrieval = RetrievalAgent(
        s2=_SmokeS2(papers_dir=papers_dir),  # type: ignore[arg-type]
        arxiv=_SmokeArxiv(),  # type: ignore[arg-type]
        downloader=_SmokePdfDownloader(papers_dir=papers_dir),  # type: ignore[arg-type]
    )
    extraction = ExtractionAgent(pdf_extractor=PdfExtractor())
    indexing = IndexingAgent(embedder=embedder)
    analysis = AnalysisAgent(llm=llm, prompts=prompts)
    report = ReportAgent(latex_templates=latex_templates)

    return [query, retrieval, extraction, indexing, analysis, report]




@dataclass
class _SmokeS2:
    """Returns the Phase-1 2-paper corpus for any query.

    Duck-typed match for ``ara.services.fetchers.S2Client.search``. Reads the
    committed fixture JSONs and adapts them to ``S2SearchResult`` shape.
    """

    papers_dir: Path

    def search(
        self,
        query: str,
        *,
        limit: int = 10,
        year_since: int | None = None,
        min_citation_count: int | None = None,
    ) -> list[Any]:
        import json as _json

        from ara.services.fetchers import S2SearchResult

        with_pdf = _json.loads(
            (self.papers_dir / "paper_with_pdf.json").read_text(encoding="utf-8")
        )
        abs_only = _json.loads(
            (self.papers_dir / "paper_abstract_only.json").read_text(encoding="utf-8")
        )

        def _to_hit(raw: dict[str, Any], pdf_url: str | None) -> S2SearchResult:
            return S2SearchResult(
                paper_id=raw["paper_id"],
                title=raw["title"],
                authors=raw.get("authors", []),
                year=raw.get("year"),
                venue=raw.get("venue"),
                citation_count=raw.get("citation_count"),
                abstract=raw.get("abstract"),
                pdf_url=pdf_url,
                arxiv_id=None,
                doi=None,
            )

        return [
            _to_hit(with_pdf, pdf_url=str(self.papers_dir / "paper_with_pdf.pdf")),
            _to_hit(abs_only, pdf_url=None),
        ]


@dataclass
class _SmokeArxiv:
    """Always returns no match — the smoke corpus covers S2 + abstract-only only.

    Duck-typed match for ``ara.services.fetchers.ArxivClient``.
    """

    def lookup_by_id(self, arxiv_id: str) -> None:
        return None

    def lookup_by_title(
        self, title: str, year: int | None = None, accept_threshold: float = 0.8
    ) -> None:
        return None


@dataclass
class _SmokePdfDownloader:
    """Copies the Phase-1 ``paper_with_pdf.pdf`` into ``dest`` (simulating a download).

    Duck-typed match for ``ara.services.fetchers.PdfDownloader``.
    """

    papers_dir: Path

    def download(self, url: str, dest: Path) -> int:
        src = Path(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = src.read_bytes()
        dest.write_bytes(data)
        return len(data)

    def close(self) -> None:
        pass
