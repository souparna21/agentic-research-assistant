"""Stage implementations.

`base.py` defines the Stage protocol; individual stage files (query.py,
retrieval.py, extraction.py, ...) are added in Phases 2-3. Phase 1 stubs
live in `stubs.py` (created in plan 09).

Convenience re-exports for downstream factory wiring (plan 02-07).
"""

from ara.agents.analysis import AnalysisAgent
from ara.agents.extraction import ExtractionAgent
from ara.agents.indexing import IndexingAgent, IndexingIntegrityError
from ara.agents.query import QueryAgent
from ara.agents.report import ReportAgent
from ara.agents.retrieval import RetrievalAgent

__all__ = [
    "AnalysisAgent",
    "ExtractionAgent",
    "IndexingAgent",
    "IndexingIntegrityError",
    "QueryAgent",
    "ReportAgent",
    "RetrievalAgent",
]
