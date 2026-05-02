# Agentic Research Assistant (`ara`)

> Turn a natural-language research question into a citation-backed LaTeX literature review using only free, open-source tools — six sequential agents, one deterministic pipeline.

**Course:** IE624, IIT Bombay (April 2026)
**Team:** Souparna Bhowmik (25D1386), Damor Jaydipkumar (22B4221), Akash Sansugu Palaniswami (21D171001)

---

## Table of contents

1. [What this project does](#what-this-project-does)
2. [Repository layout](#repository-layout)
3. [Architecture](#architecture)
4. [Setup](#setup)
5. [Running the project](#running-the-project)
6. [Running the live demo](#running-the-live-demo)
7. [Tests](#tests)
8. [Reproducibility guarantees](#reproducibility-guarantees)
9. [Troubleshooting](#troubleshooting)
10. [License](#license)

---

## What this project does

`ara` automates the most tedious half of a literature review:

1. **You give it a research question** in natural English.
2. It **expands the question** into 3–5 search terms via Gemini.
3. It **discovers papers** through Semantic Scholar (with arXiv fallback) and **downloads PDFs**.
4. It **extracts the text** from each PDF — with a span-level prompt-injection defense that strips white-on-white, sub-half-point, off-mediabox, and zero-bbox spans.
5. It **chunks and indexes** the text with FAISS (`IndexFlatIP` on L2-normalized MiniLM embeddings — equivalent to cosine similarity).
6. It **runs hierarchical RAG analysis**: per-paper summaries with `[P<paper-id>-N]` citations, methodology extraction, limitations extraction, and a per-claim verifier — all schema-validated against strict Pydantic models.
7. It **synthesizes across papers**: a methodology comparison matrix and a research-gaps map.
8. It **assembles a LaTeX report** with proper BibTeX citations, runs the collision-safe BibTeX-key resolver, and (if `latexmk` is available) compiles `report.pdf`.

Every claim in the output traces back to a specific passage in a specific paper. The verifier re-checks every claim post-hoc and prepends `[UNVERIFIED]` to anything it can't ground in the source passages.

---

## Repository layout

```
.
├── README.md                       This file.
├── PRESENTATION_SCRIPT.md          20-minute presentation script with speaker assignments.
├── report.tex                      IE624 final project report (architecture, results, evaluation).
├── references.bib                  Bibliography for report.tex (16 entries).
├── ie624_presentation.pptx         Slide deck (20 slides) — open in Google Slides or PowerPoint.
├── pyproject.toml                  Project metadata + dependency declarations.
├── uv.lock                         Pinned dependency closure (SHA-256 checksums).
├── .python-version                 Python version pin (3.12).
├── .env.example                    Environment template — copy to .env and fill in your API key.
├── demo-cache.tar.gz               Committed offline-demo artifact (~8 KB).
│
├── src/ara/                        Source code.
│   ├── cli.py                      Typer CLI: `ara run | stage | show | clean`.
│   ├── orchestrator.py             Stage runner; enforces FIELD_OWNERS invariant.
│   ├── state.py                    PipelineState (Pydantic) + FIELD_OWNERS + StageReport.
│   ├── config.py                   Settings loader (reads .env via pydantic-settings).
│   ├── analysis_schemas.py         Pydantic schemas for analysis prompt outputs.
│   ├── persistence.py              Atomic state.json read/write helpers.
│   ├── logging.py                  structlog setup with run_id + git_sha context-binding.
│   ├── demo_cache.py               Offline-demo extraction (tarfile filter='data').
│   ├── runids.py                   Run-id generation + git SHA discovery.
│   ├── errors.py                   Domain exception hierarchy.
│   │
│   ├── agents/                     The six pipeline agents.
│   │   ├── query.py                Stage 1 — Gemini query expansion + topic-drift judge.
│   │   ├── retrieval.py            Stage 2 — S2 + arXiv discovery, PDF download, dedup.
│   │   ├── extraction.py           Stage 3 — PyMuPDF extraction + injection defense.
│   │   ├── indexing.py             Stage 4 — section-aware chunking, MiniLM embed, FAISS.
│   │   ├── analysis.py             Stage 5 — hierarchical RAG (4N+2 LLM calls) + verifier.
│   │   ├── report.py               Stage 6 — Jinja2 LaTeX, BibTeX collision resolver, latexmk.
│   │   ├── base.py                 Stage Protocol definition.
│   │   └── stubs.py                Test factory: build_stub_stages() for offline smoke tests.
│   │
│   └── services/                   Reusable infrastructure.
│       ├── llm.py                  LiteLLMClient + FakeLLM. Provider-agnostic via config string.
│       ├── fetchers.py             S2Client, ArxivClient, PdfDownloader.
│       ├── pdf.py                  PyMuPDF wrapper with span-level redaction.
│       ├── chunker.py              SectionAwareChunker (~500 token chunks, 50-token overlap).
│       ├── embed.py                MiniLMEmbedder (sentence-transformers, normalize=True).
│       ├── vector_store.py         FaissVectorStore (IndexFlatIP + chunks.json sidecar).
│       ├── prompts.py              PromptLoader + LatexTemplateLoader (Jinja2).
│       └── latex.py                latex_escape filter, BibTeX collision resolver.
│
├── prompts/                        Jinja2 templates — prompts AND LaTeX skeleton.
│   ├── query_expansion.j2
│   ├── query_judge.j2
│   ├── per_paper_summary.j2
│   ├── methodology_extraction.j2
│   ├── limitations_extraction.j2
│   ├── methodology_matrix_synthesis.j2
│   ├── gap_synthesis.j2
│   ├── verifier.j2
│   └── report_skeleton.tex.j2
│
├── tests/                          263 tests (pytest, offline, deterministic).
│   ├── unit/                       Per-module unit tests.
│   ├── integration/                Cross-module flow tests.
│   ├── e2e/                        Smoke tests including --cached-only end-to-end.
│   └── fixtures/                   Test corpus + golden state + FakeLLM JSON fixtures.
│
└── scripts/
    ├── dress_rehearsal.sh          One-command verification: clone fresh → uv sync → cached-only → assert.
    ├── create_demo_cache.py        Regenerate demo-cache.tar.gz from current fixtures.
    └── regen_smoke_fixtures.py     Re-record FakeLLM responses against a known-good run.
```

---

## Architecture

### High-level data flow

```
                                                      cited LaTeX review
   Question                                            (with BibTeX bibliography)
      │                                                             ▲
      ▼                                                             │
 ┌─────────┐    ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌─────────┐
 │ Query   │───▶│ Retrieval│───▶│Extraction│───▶│ Indexing │───▶│Analysis │───┐
 │ Expand  │    │  S2 +    │    │ PyMuPDF  │    │  Chunk + │    │ RAG +   │   │
 │ (LLM)   │    │  arXiv   │    │ + scrub  │    │  FAISS   │    │ Verifier│   │
 └─────────┘    └──────────┘    └──────────┘    └──────────┘    └─────────┘   │
                                                                              │
                                                                              ▼
                                                                      ┌────────────┐
                                                                      │  Report    │
                                                                      │  latexmk + │
                                                                      │  BibTeX    │
                                                                      └────────────┘

  ┌───────────────────────────────────────────────────────────────────────────────┐
  │ Shared PipelineState (typed Pydantic; FIELD_OWNERS single-writer invariant)   │
  └───────────────────────────────────────────────────────────────────────────────┘
```

### Single-writer invariant

The six agents communicate **only** through one `PipelineState` Pydantic model. A frozen `FIELD_OWNERS` map registers which fields each stage may write:

```python
FIELD_OWNERS = {
    "query":      {"search_terms", "stage_reports"},
    "retrieval":  {"papers", "retrieval_stats", "stage_reports"},
    "extraction": {"papers", "extraction_stats", "stage_reports"},
    "indexing":   {"chunks_count", "index_path", "indexing_stats", "stage_reports"},
    "analysis":   {"papers", "summaries_dir", "comparison_path", "gaps_path", ...},
    "report":     {"papers", "report_path", "report_stats", "stage_reports"},
}
```

The `Orchestrator` snapshot-diffs non-owned fields before and after each `stage.run()` and raises `FieldOwnershipViolation` on cross-writes. This mitigates Cemri et al. 2025's MAST finding that **42% of multi-agent LLM failures are specification errors**.

### Stage details

| # | Agent | Input | Output | Notes |
|---|---|---|---|---|
| 1 | `QueryAgent` | natural-language question | `state.search_terms` (3–5 terms) | 1 LLM call; LLM-as-judge filters off-topic expansions |
| 2 | `RetrievalAgent` | search_terms | `state.papers`, downloaded PDFs | S2 → arXiv fallback; tenacity backoff on 429; withdrawal detection |
| 3 | `ExtractionAgent` | PDFs | extracted Markdown per paper | PyMuPDF + pymupdf4llm; strips hidden-text injection vectors |
| 4 | `IndexingAgent` | extracted text | FAISS index + `chunks.json` | section-aware ~500 token chunks; MiniLM-L6-v2 normalised embeddings |
| 5 | `AnalysisAgent` | papers + index | per-paper summaries, methodology matrix, gap analysis | hierarchical RAG: 4N+2 LLM calls (4 per paper + 2 cross-paper); per-claim verifier |
| 6 | `ReportAgent` | all of the above | `report.tex` (+ `references.bib`, optional `report.pdf`) | Jinja2 LaTeX with `<% %>` / `<< >>` delimiters; collision-safe BibTeX keys; latexmk gate |

### Hierarchical RAG (Stage 5 detail)

To mitigate Belem et al. 2025's up-to-75% multi-doc hallucination rate, analysis runs in two layers:

```
Per-paper layer (4 calls × N papers):
  ┌──────────┐  ┌──────────────┐  ┌─────────────┐  ┌──────────┐
  │ Summary  │  │ Methodology  │  │ Limitations │  │ Verifier │
  │ obj+find │  │ datasets+    │  │ + future    │  │ fact-    │
  │ +limits  │  │ techniques+  │  │ work        │  │ check    │
  │          │  │ metrics      │  │             │  │ vs pass. │
  └──────────┘  └──────────────┘  └─────────────┘  └──────────┘

Cross-paper layer (2 calls, runs once):
  ┌──────────────────────┐    ┌──────────────────────┐
  │ Methodology Matrix   │    │ Gap Analysis         │
  │ rows = papers,       │    │ synthesises per-paper│
  │ cols = datasets/     │    │ limitations into     │
  │ techniques/metrics   │    │ research gaps        │
  └──────────────────────┘    └──────────────────────┘
```

Total: **4N+2** LLM calls. For 10 papers: 42 calls.

### Citation format

Every generated claim carries a citation in the format `[P<paper_id>-N]` where `paper_id` is pinned inside the marker (e.g. `[Parxiv:2401.15391-3]`). This format is enforced by a Pydantic `field_validator` on `Claim.citations`, so attribution drift between papers is impossible.

---

## Setup

### Prerequisites

- **Python 3.12+** (exact version pinned in `.python-version`)
- **[uv](https://docs.astral.sh/uv/)** — the dependency manager
- **Optional: LaTeX toolchain** (`latexmk` + `texlive-latex-recommended`) if you want to compile `report.pdf` locally instead of using Overleaf

### One-time install

```bash
# 1. Install uv (if not already installed)
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.bashrc   # or ~/.zshrc — whichever your shell uses

# 2. Clone and enter the repo
git clone https://github.com/souparna21/<repo>.git
cd <repo>

# 3. Create the hermetic virtual environment (uv reads uv.lock)
uv sync

# 4. Configure your Gemini API key
cp .env.example .env
# Edit .env and set:
#   ARA_GEMINI_API_KEY=<your-key>
#   GEMINI_API_KEY=<same-key>           # LiteLLM auto-discovers this name
#   ARA_LLM_MODEL=gemini/gemini-2.5-flash-lite
# Get a free API key at https://aistudio.google.com/apikey
```

### Optional: LaTeX toolchain

If you want a local PDF (otherwise Overleaf works):

```bash
# Debian / Ubuntu
sudo apt-get install -y texlive-latex-recommended texlive-latex-extra latexmk

# macOS
brew install --cask mactex-no-gui
```

Without `latexmk`, the report agent still writes `report.tex` + `references.bib` — you can upload them to Overleaf to compile.

---

## Running the project

All commands run under `uv run` so they pick up the hermetic `.venv/` and the `ara` console entry point.

### Full pipeline (live)

```bash
uv run ara run "How does retrieval-augmented generation reduce hallucination?"
```

Runs all six stages end-to-end. Wall-clock: roughly 2–3 minutes for a 10-paper corpus on CPU. Output lands in `runs/<run_id>/`.

### With retrieval filters

```bash
uv run ara run "your question" --year-since 2020 --min-citations 50
```

Restricts Semantic Scholar results to recent, well-cited papers — useful when API quota is tight.

### Resume a single stage

The `Orchestrator` skips stages whose output artifact already exists (delete-to-invalidate idempotency). To re-run a single stage:

```bash
# Delete the artifact you want to invalidate, e.g.:
rm runs/<run-id>/index/faiss.index

# Then re-run the named stage
uv run ara stage indexing --run-id <run-id>
```

Valid stage names: `query | retrieval | extraction | indexing | analysis | report`.

### Inspect a finished run

```bash
uv run ara show <run-id>
```

Prints the question, config snapshot, paper count, and per-stage report list from `runs/<run-id>/state.json`.

### Clean up runs

```bash
uv run ara clean --run-id <run-id>     # delete one
uv run ara clean --all --yes           # delete all (no prompt with --yes)
```

---

## Running the live demo

There are **three demo paths**. Combine them for safety.

### Path 1 — Offline demo (4 seconds, zero network, bulletproof)

```bash
uv run ara run "anything" --cached-only
```

What happens:
- Extracts the committed `demo-cache.tar.gz` into `runs/demo-v1/`.
- The orchestrator runs against the extracted run-id; every stage finds its artifact on disk and emits `stage_skipped`.
- Zero outbound API calls — verified in CI by `pytest-socket` globally blocking sockets.

This is the **safest demo path**. If your demo-room WiFi dies, this still works.

### Path 2 — Live end-to-end demo (2–3 minutes)

```bash
uv run ara run "<your question>"
```

You can narrate the architecture as the stage logs scroll by:
- `stage_start: query` — Gemini expanding the question
- `stage_start: retrieval` — arXiv calls + PDF downloads
- `stage_start: extraction` — PyMuPDF + injection-defense firing
- `stage_start: indexing` — embeddings + FAISS index build
- `stage_start: analysis` — 4N+2 hierarchical RAG calls
- `stage_start: report` — LaTeX assembly

### Path 3 — Show prior live-run artifacts

If you have a previous live run preserved in `runs/<id>/`, open these files:
- `state.json` — the typed `PipelineState` with all measurements
- `report.tex` — the auto-generated literature review
- `comparison.json` — cross-paper methodology matrix
- `gaps.json` — research gap analysis with citations
- `summaries/arxiv:*.json` — per-paper summaries with citation tags

### Pre-demo checklist (run the day before, not the day of)

```bash
git clone https://github.com/souparna21/<repo>.git && cd <repo>
uv sync                                       # ~30 s
cp .env.example .env                          # edit and add your API key
uv run pytest -q                              # expect 263 passed in ~75 s
./scripts/dress_rehearsal.sh                  # 4-second offline smoke
uv run ara run "What is RAG?"                 # 2-3 min live (uses some quota)
```

If all four pass on the demo machine, you're ready.

### Suggested 20-minute demo flow

1. **(2 min) Architecture overview** — slide 4 of the deck (six agents + shared state)
2. **(1 min) Tests green** — `uv run pytest -q` shows 263 passed
3. **(1 min) Offline demo** — `uv run ara run "anything" --cached-only`
4. **(4 min) Live run** — `ara run "<professor's question>"`, narrate the stages
5. **(3 min) Walk through the auto-generated literature review** — open the produced `report.tex`, point at `[Pqi_2025_ar-3]` style citations, the methodology comparison, the research gaps
6. **(remaining)** Q&A — pull up the IE624 `report.tex` for measured numbers

See `PRESENTATION_SCRIPT.md` for the full speaker-by-speaker script.

---

## Tests

```bash
uv run pytest -q                  # 263 passed, 9 latex-skipped, 0 flakes
uv run pytest -q -m smoke         # just the e2e smoke tests
uv run pytest -q -m latex         # latexmk-gated tests (only with TeX installed)
```

Tests are offline by design: `pytest-socket --disable-socket` is in `pyproject.toml`'s `addopts`, so any accidental network call fails loudly with `SocketBlockedError`. `FakeLLM` keys responses by `sha256(prompt)[:16]` and raises `UnknownPromptError` on miss — so a whitespace change in any Jinja prompt template fails the test loudly instead of silently producing degraded output.

---

## Reproducibility guarantees

Every reproducibility claim is mechanised — no "trust us":

- **Pinned dependencies:** `uv.lock` contains the full transitive closure with SHA-256 checksums. `uv sync` is bit-reproducible across machines.
- **Pinned Python version:** `.python-version` (3.12) honoured by `uv` and `pyproject.toml`.
- **Pinned LLM model:** `ARA_LLM_MODEL=gemini/gemini-2.5-flash-lite` in `.env.example`. Override only via `.env`.
- **Deterministic output:** every analysis and verifier LLM call passes `temperature=0`. `FakeLLM` is keyed by prompt hash and raises on miss.
- **Traceable logs:** every line in `runs/<run_id>/run.log` (NDJSON) carries `run_id` + `git_sha`, bound once per `Orchestrator.__init__` via `structlog.contextvars`. Secrets are redacted at log-time.
- **Demo tarball:** `demo-cache.tar.gz` is a committed artifact — auditable via `git diff`, regenerable via `scripts/create_demo_cache.py`.

---

## Troubleshooting

### Gemini quota exhausted

Free tier is **20 requests/day per model** (not 500 — the documentation is misleading). A full live run uses about 39 calls.

**Options:**
1. Enable Gemini billing — Tier 1 is 1500 RPD and a full run costs ~$0.03.
2. Switch to a different free model (`gemini-flash-latest`, `gemini-2.5-flash`, `gemini-2.5-flash-lite` each have separate 20-RPD buckets).
3. Use `--cached-only` — zero API calls.

### `latexmk: command not found`

Two options:
1. Install the LaTeX toolchain (commands above).
2. Skip local PDF compile — upload `report.tex` + `references.bib` to Overleaf instead.

The `ReportAgent` detects missing `latexmk` and degrades gracefully: it writes `report.tex` + `references.bib` and sets `pdf_bytes=0` in `ReportStats`.

### Semantic Scholar IP rate-limiting

The unauthenticated S2 endpoint hard-blocks aggressive callers with sticky 429s. Two mitigations:
1. **Get a free S2 API key** — 1 RPS sustained, no IP-block sensitivity. One-line config change in `S2Client(api_key=...)`.
2. **Wait** — the cooldown window is unpredictable but typically clears within an hour.

### `pytest-socket.SocketBlockedError`

Expected. The test suite blocks all sockets via `pyproject.toml addopts: --disable-socket`. Real-API tests are tagged `@pytest.mark.live` and deselected by default; run them with `uv run pytest -m live`.

### `SentenceTransformer` tries to download from HuggingFace

The smoke tests set `HF_HUB_OFFLINE=1` + `TRANSFORMERS_OFFLINE=1` to force the local cache. Populate the MiniLM cache once with:

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"
```

---

## License

MIT — see `pyproject.toml`.
