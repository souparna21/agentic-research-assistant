# Instructions for Claude — Build the IE624 Presentation Deck

You are an expert presentation designer. Your task is to generate a polished
30‑slide `.pptx` file for the IE624 final project at IIT Bombay using
`python-pptx`. Output the script as `scripts/build_presentation_v2.py` and
save the deck to `ie624_presentation.pptx` (overwrite the existing file).

This document is **self-contained** — every fact, number, design rule, and
slide blueprint you need is here. Do not search the codebase for content;
use the project context in §1 and the slide blueprint in §6 verbatim. Use
the codebase only to verify file paths cited in the demo slides.

The reference style file is `seminar.pptx` (already inspected; the design
system in §2 reproduces it). Do not copy `seminar.pptx`'s content — only
its visual style.

---

## §1. Full Project Context

### 1.1 What the project is

**Title:** Agentic Research Assistant (ARA) — A Multi-Agent RAG Pipeline for
Automated Research Surveys

**Course:** IE624 — Sustainable Technology, Innovation, and Business
**Institution:** IIT Bombay
**Team:** Souparna Bhowmik, Damor (first name), Akash (first name)
*(use the surnames if you find them in `report.tex`; otherwise leave as-is)*

**One-sentence pitch:** ARA takes a natural-language research topic and
returns a verifier-checked LaTeX literature survey end-to-end — discovering
papers from arXiv and Semantic Scholar, parsing PDFs with prompt-injection
defence, indexing them in FAISS, synthesising a methodology matrix and gap
analysis with hierarchical RAG, and rendering a compiled PDF report — in
about 3 minutes for $0.03.

### 1.2 The problem we are solving

Literature reviews are the highest-leverage step in any research project,
yet they are crushingly manual: discovering papers, downloading PDFs,
reading them, extracting comparable methodology details, identifying gaps,
writing a coherent narrative, and citing every claim correctly. A PhD
student spends weeks on this; a course project usually skips it.

Naïvely throwing an LLM at the problem fails: Belem et al. (2025) showed
that single-call multi-document summarisation hallucinates **75 % of the
time** when the model has to reason across more than three papers. ARA's
core contribution is a *hierarchical* pipeline that isolates per-paper
reasoning before any cross-paper synthesis, and verifies every generated
claim against its cited passage.

### 1.3 The six agents

Each agent is a pure function `state -> state`. They run in a fixed
sequence under an immutable Orchestrator that enforces a
single-writer-per-field invariant: each field of `PipelineState` has
exactly one agent allowed to write it (declared in `FIELD_OWNERS`). After
each stage, the orchestrator diffs the output against the snapshot it took
before the call and raises `FieldOwnershipViolation` if a non-owner wrote.

| # | Agent | Reads | Writes | Key technique |
|---|---|---|---|---|
| 1 | **Query** | `topic` | `expanded_terms`, `query_decision` | LLM expands the topic into 6 search terms; an LLM-judge accepts or rejects the expansion |
| 2 | **Retrieval** | `expanded_terms` | `papers` | arXiv Atom + Semantic Scholar JSON, dedup by DOI/arXiv-id, 2 s inter-term sleep to respect S2's 1 RPS, tenacity backoff on 429 |
| 3 | **Extraction** | `papers` | `extracted_texts` | PyMuPDF span-level extraction with three injection defences: white-on-white spans, sub-0.5pt fonts, off-mediabox spans — all stripped before LLM ever sees the text |
| 4 | **Indexing** | `extracted_texts` | `chunks`, `vector_index_path` | Section-aware chunking (500 / 50 token overlap), all-MiniLM-L6-v2 embeddings, FAISS `IndexFlatIP` on L2-normalised vectors (cosine equivalent) |
| 5 | **Analysis** | `chunks`, `vector_index_path` | `per_paper_summaries`, `methodology_matrix`, `gap_analysis`, `verifier_verdicts` | Hierarchical RAG: 4N + 2 LLM calls — for each of N papers do {summary, methodology, limitations, verifier} = 4 calls, then 2 cross-paper calls (matrix + gaps). Pydantic schema-strict (`extra="forbid"`) |
| 6 | **Report** | everything above | `report_tex_path`, `report_pdf_path` | Jinja2 LaTeX template, BibTeX collision-resolved keys, `latexmk` compile gate (skipped cleanly when `latexmk` is absent) |

### 1.4 Data model — `PipelineState`

A frozen Pydantic model. Fields filled in order; every field has exactly
one owner. The orchestrator snapshot-diffs after each stage and crashes
loudly if a non-owner wrote a field. This is the core invariant that makes
the pipeline auditable and re-runnable from any stage.

### 1.5 Tooling stack

- **LLM gateway:** LiteLLM (provider-agnostic). Default model
  `gemini/gemini-2.5-flash-lite` (no thinking-token tax). Swappable to
  OpenAI / Anthropic via env-var only. Retries 503 / RateLimit /
  InternalServerError / APIConnectionError with 4 / 8 / 16 s backoff.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` (80 MB, CPU
  fast, ~5× faster than mpnet-base). `normalize_embeddings=True` so
  FAISS inner-product == cosine similarity.
- **Vector store:** FAISS `IndexFlatIP` — exact search; small corpus
  (~700 chunks) so HNSW would be over-engineering.
- **PDF parsing:** PyMuPDF span-level (not page-level) — enables
  injection-vector detection.
- **Validation:** Pydantic v2 with `extra="forbid"`; LiteLLM
  `response_format={"type": "json_schema", "strict": true}` pins the
  Gemini output to our schema.
- **CLI:** Typer with subcommands `run`, `resume`, `inspect`, `clean`.
- **Logging:** `structlog` dual-sink (console + JSON file).
- **Test double:** `FakeLLM` keyed on SHA-256[:16] of the prompt — raises
  `UnknownPromptError` on prompt drift instead of returning empty text.
- **Tests:** 266 unit + e2e, all hermetic via `pytest-socket`.

### 1.6 Live run results — headline run

Run-id `20260502-190241-what-is-rag` on 2026-05-02 against authenticated
Semantic Scholar (free-tier API key). These are the **actual measured
numbers** to put on the slides — do not invent.

| Metric | Value |
|---|---|
| Topic | "What is RAG?" |
| Search terms | retrieval-augmented generation, RAG explained, large language models RAG |
| Raw S2 hits | **30** (3 requests, 0 retries, 0 IP-blocks) |
| Papers after dedup | **29** |
| PDFs downloaded | **10** (8 via arXiv-fallback, 2 direct from S2) |
| Aggregate PDF size | **13.55 MB** |
| Abstract-only fallbacks | **19** (graceful degradation when no PDF) |
| Hidden injection spans stripped | **31** (defense fired on real PDFs) |
| Chunks indexed | **551** across 10 full-text papers |
| Papers analysed successfully | **10 / 10** (citation-validator hardening) |
| Total LLM calls | **42** = 1 query + 30 per-paper + 2 synthesis + 10 verifier (rounding) |
| Verifier `unsupported` verdicts | **0 / 10** |
| Wall-clock (sum of stages) | **331 s** (retrieval 162.7 + extraction 47.3 + indexing 10.5 + analysis 110.7 + report 0.02) |
| Output report size | **29.69 KB** LaTeX |
| BibTeX keys | **10** (0 collisions, 0 unresolved citations) |
| Cost on Tier-1 billing | **≈ US$0.03** |

For broader retrieval-quality numbers (top-k probe scores), the earlier
April-28 reference run on the same query family produced an inner-product
range of **0.432 – 0.682**, mean **0.603** across 25 retrieved chunks.
MiniLM-L6's empirical cosine floor on unrelated text is 0.30–0.40, so a
mean of 0.603 puts retrieved passages well above the noise floor.

### 1.7 Failure modes we observed and disclosed (now mitigated)

1. **Citation format drift (FIXED).** Pre-fix, the LLM occasionally
   omitted the `[ ]` brackets around citations; the strict regex dropped
   2/10 papers on April-28 and 10/26 papers on a longer-paper-id S2
   diagnostic. The validator now accepts both bracketed and bare forms
   and normalizes to bracketed before storage.
   Headline May-2 run: **0 / 10 dropped**.
2. **Semantic Scholar IP-blocking (FIXED).** S2 unauthenticated tier
   IP-blocked us mid-development. Two-step fix: (a) arXiv-as-discovery
   fallback path inside `RetrievalAgent` so `ara run` self-heals when S2
   is down, (b) authenticated `ARA_S2_API_KEY` wired through `S2Client`.
   Headline May-2 run: **30 S2 results, 0 retries, 0 fallback hits**.
3. **Gemini free-tier 20 RPD per model**, not the 500 RPD originally
   assumed. Switched to billed Tier-1 + `flash-lite` (~$0.03/run).
4. **`gemini-3-flash-preview` 16 K hidden thinking tokens** for a 2 KB
   JSON output. Switched to `gemini-2.5-flash-lite` (no thinking mode).

### 1.8 Demo paths

- **Offline (4 s):** `uv run ara run "anything" --cached-only` — replays
  artefacts from `demo-cache.tar.gz`. No network, no API key. Bullet-proof.
- **Live (5–6 min):** `uv run ara run "your topic"` — full real pipeline.
  Needs `ARA_GEMINI_API_KEY`, optional but strongly recommended
  `ARA_S2_API_KEY`, and ~$0.03 of Gemini billing.
- **Prior artefacts:** `runs/20260502-190241-what-is-rag/` already on
  disk — open the LaTeX, the JSON state, the verifier verdicts directly.

### 1.9 Sample analysis output (for slide 22)

A real per-paper summary fragment from the headline run, paper
`2b6af398...` ("Bounding Hallucinations: Information-Theoretic Guarantees
for RAG Systems via Merlin-Arthur Protocols", 2025):

> **Objective.** The objective is to bound hallucinations in
> Retrieval-Augmented Generation (RAG) systems using Merlin-Arthur (M/A)
> protocols, providing information-theoretic guarantees \cite{wenger_2025_bounding}
>
> **Findings.** M/A training leads to improvements in accuracy,
> completeness, soundness, and EIF (Expected Information Flow) for
> language models. `[P2b6af398...-1]`

Every claim carries a `[P<paper-id>-N]` citation that traces to a specific
passage in a specific paper. The verifier post-checks each claim against
its cited passage; unverified claims would be tagged `[UNVERIFIED]`.

---

## §2. Design System

Match the visual style of `seminar.pptx`. Specifications below are exact;
do not deviate.

### 2.1 Slide geometry

- **Aspect:** 16:9
- **Width × Height:** 10.0 × 5.625 inches (Pptx EMU: 9 144 000 × 5 143 500)
- **Set with:**
  ```python
  prs.slide_width  = Inches(10.0)
  prs.slide_height = Inches(5.625)
  ```

### 2.2 Color palette

| Role | Hex | Usage |
|---|---|---|
| Primary navy | `#0D2A4F` | Slide titles, headline numbers, page number |
| Secondary blue | `#2C4A7C` | Subtitles, section labels, "Why this matters" headers |
| Accent red | `#8B0000` | "Problem", "Failure", "Limitation" labels (sparingly) |
| Body dark | `#1A1A1A` | All body text |
| Muted gray | `#555555` | Footer attribution, page-x-of-y, low-importance captions |
| Light tint | `#E8EEF5` | Diagram box fills (light navy tint) |
| Accent fill | `#F5E8E8` | Diagram callout fills (light red tint) |
| Rule line | `#C9D2DD` | Horizontal divider under titles |

### 2.3 Typography

- **Font family:** Calibri throughout. No exceptions.
- **Title (slide headline):** 24 pt bold, color navy `#0D2A4F`
- **Section label (top-left tab, e.g. "Agent 3"):** 13 pt bold,
  color secondary blue `#2C4A7C`
- **Section name (under section label):** 18 pt bold, color navy
- **H2 within slide (e.g. "Method", "Results"):** 13 pt bold,
  color secondary blue (or accent red for "Problem"/"Limitations")
- **Body bullets:** 12 pt regular, color body dark, 1.15 line spacing
- **Sidebar / secondary column:** 11 pt regular
- **Footer attribution (bottom-left):** 8 pt regular, color muted gray
- **Page number (bottom-right):** 9 pt regular, color muted gray,
  format **`X/30`**

### 2.4 Layout grid

All coordinates in inches (use `Inches(x)`).

```
┌──────────────────────────────────────────────────────────────┐
│ TITLE BAR        title at (0.40, 0.22), 9.20 × 0.55         │
│ ───────────────  rule line at (0.40, 0.85), 9.20 × 0.02     │
│                                                              │
│ CONTENT AREA     starts y=1.00, ends y=5.20                 │
│                  full-width: x=0.40, w=9.20                 │
│                  two-column: left x=0.40 w=4.50,            │
│                             right x=5.10 w=4.50             │
│                  gutter between columns: 0.20 in            │
│                                                              │
│ FOOTER     (0.40, 5.30) attribution      (9.30, 5.30) page# │
└──────────────────────────────────────────────────────────────┘
```

The rule line under titles is a thin shape — `MSO_SHAPE.RECTANGLE` with
height `Inches(0.015)`, fill rule-line color, no border.

### 2.5 Footer (every slide except slide 1)

- **Bottom-left attribution:** `"IE624 — Agentic Research Assistant · IIT Bombay"`
  at (0.40, 5.30), width 7.50 in, 8 pt, muted gray, left-aligned.
- **Bottom-right page number:** `"3/30"` at (9.30, 5.30), width 0.50 in,
  9 pt, muted gray, right-aligned.

### 2.6 Slide-template patterns

Define a small set of reusable layout helpers in your script:

1. `add_title(slide, text)` — title bar + rule line
2. `add_footer(slide, page_num, total=30)` — attribution + page number
3. `add_section_tab(slide, label, name)` — top-left section label
   (e.g. for agent slides: "Agent 3" / "Extraction")
4. `add_bullets(slide, x, y, w, h, items, size=12, color=BODY_DARK)` —
   bulleted text block
5. `add_h2(slide, x, y, text, color=SECONDARY_BLUE)` — sub-section heading
6. `add_box(slide, x, y, w, h, fill, border, text, font_size, bold)` —
   rounded-rectangle diagram primitive
7. `add_arrow(slide, x1, y1, x2, y2)` — connector line with arrowhead
8. `add_callout(slide, x, y, w, h, text)` — light-red callout for
   "Watch out:" / "Why this matters:" notes

### 2.7 Density rules

- **Maximum 6 bullets per content block.** If you need more, split into a
  second block or use a two-column layout.
- **Maximum 12 words per bullet.** No paragraphs in bullets.
- **One diagram per slide max.**
- **Whitespace is design.** The bottom 0.4 in below the last content
  element should be empty (above the footer).

---

## §3. The 30-slide blueprint

For each slide, the section below gives:

- **Purpose** — the one thing the slide communicates
- **Title** — exact text (24 pt navy)
- **Layout** — full-width / two-column / diagram / etc.
- **Content** — exact bullet text and any callouts
- **Diagram** — full spec if the slide has one (see §4)
- **Speaker note** — one line of context for the deck reader

The page-number format is `X/30` for all 30 slides.

---

### Slide 1/30 — Title

- **Purpose:** establish project, course, team
- **Layout:** centered title block, no footer/page-number
- **Content:**
  - Headline (centered at y=1.30, 28 pt bold navy):
    `Agentic Research Assistant`
  - Subhead (y=1.95, 18 pt secondary blue):
    `A Multi-Agent RAG Pipeline for Automated Research Surveys`
  - Course tag (y=2.85, 14 pt body dark):
    `IE624 — Sustainable Technology, Innovation, and Business`
  - Team block (y=3.60, 14 pt bold body dark):
    `Souparna Bhowmik · Damor · Akash`
  - Affiliation (y=3.95, 12 pt muted gray):
    `Indian Institute of Technology Bombay · Final project · 2026`

### Slide 2/30 — Agenda

- **Purpose:** preview the talk
- **Layout:** full-width bullet list
- **Title:** `Agenda`
- **Content (single column, 7 bullets):**
  - Why automated literature reviews (slides 3–5)
  - System architecture and the six agents (slides 6–14)
  - Tooling stack and hierarchical RAG (slides 15–17)
  - Live-run results (slide 18)
  - **Walkthrough: "What is RAG?" — outputs in detail** (slides 19–23)
  - Evaluation, failure modes, lessons (slides 24–27)
  - Limitations, demo, and Q&A (slides 28–30)

### Slide 3/30 — The Problem

- **Purpose:** motivate the project
- **Layout:** two-column. Left: pain. Right: scale.
- **Title:** `Literature Reviews Are the Hidden Bottleneck`
- **Left column h2 (red):** `What it costs`
  - Weeks per topic for a graduate student
  - 80 % manual: search, download, read, extract, write
  - Citations break silently when authors / years drift
  - Cross-paper synthesis is the hardest step — and the easiest to skip
- **Right column h2 (navy):** `Why we cannot skip it`
  - Foundation of every research proposal, thesis, paper
  - Tells you what is *not* yet known — defines the gap
  - The course question: can a multi-agent RAG do it well?
- **Callout (bottom, red tint, full width):**
  `Belem et al. (2025): single-call multi-document summarisation`
  `hallucinates 75 % of the time at N > 3 papers.`

### Slide 4/30 — What We Built

- **Purpose:** state the project deliverable in one slide
- **Layout:** centered headline + 4-tile grid below
- **Title:** `Our Project: ARA — Agentic Research Assistant`
- **Headline (16 pt bold body dark, centered, y=1.10):**
  `Topic in → verified LaTeX literature survey out, in 3 min for $0.03`
- **4-tile grid (2×2, each 4.50×1.40 in, light-tint fill, navy border):**
  - Tile 1 — `6 specialised agents` / `Query · Retrieval · Extraction · Indexing · Analysis · Report`
  - Tile 2 — `Hierarchical RAG` / `Per-paper isolation; cross-paper only after each paper is grounded`
  - Tile 3 — `Verifier-checked output` / `Every generated claim re-checked against its cited passage`
  - Tile 4 — `Reproducible by design` / `FakeLLM golden state · 266 hermetic tests · 4-second offline demo`

### Slide 5/30 — Goals & Success Criteria

- **Purpose:** make success measurable
- **Layout:** two-column
- **Title:** `Goals and Success Criteria`
- **Left h2 (navy):** `Functional goals`
  - End-to-end: topic → discovery → analysis → compiled report
  - Multi-source discovery: arXiv + Semantic Scholar
  - Citation grounding: every claim links to source passage
  - Reproducible: anyone can re-run our exact 10-paper survey
- **Right h2 (navy):** `Quality bars we set`
  - 0 unsupported claims in verifier audit
  - Mean retrieval score > 0.50 (well above MiniLM noise floor)
  - Per-paper failure isolated (one bad paper ≠ pipeline crash)
  - Total cost ≤ $0.05 per survey on a billed tier

### Slide 6/30 — System Overview

- **Purpose:** show the whole pipeline in one diagram
- **Layout:** centered horizontal flow
- **Title:** `System Overview — Six Agents in a Pipeline`
- **Diagram:** see §4.1 — 6 rounded-rectangle agent boxes left-to-right,
  arrows between, input "Topic" on left, outputs "PDF + LaTeX" on right.
- **Caption (centered below diagram, 11 pt secondary blue):**
  `Each agent is a pure function state → state.`
  `The Orchestrator enforces a single-writer-per-field invariant.`

### Slide 7/30 — Single-Writer Invariant

- **Purpose:** explain the architectural backbone
- **Layout:** half-half — left text, right diagram
- **Title:** `Architectural Backbone — Single Writer Per Field`
- **Left column (4.50 in wide):**
  - h2 (navy): `The rule`
  - bullets:
    - Every field of `PipelineState` has exactly one allowed writer
    - Declared in `FIELD_OWNERS` at module top
    - Orchestrator snapshots state before each agent call
    - Diffs after the call — non-owner writes raise immediately
  - h2 (navy): `Why this matters`
  - bullets:
    - Resume from any stage (no hidden side-effects)
    - Auditable: every state change has a known cause
    - Crash early, crash loud — silent corruption is impossible
- **Right column (4.50 in wide):** see §4.2 — small ownership matrix
  (5 fields × 6 agents, ✓ where the agent owns)

### Slide 8/30 — How a Run Flows

- **Purpose:** ground the abstract diagram in concrete events
- **Layout:** numbered timeline (vertical or horizontal)
- **Title:** `How a Single Run Flows`
- **Content (numbered steps, 7 boxes left-to-right, see §4.3):**
  1. CLI: `ara run "your topic"`
  2. Orchestrator builds `PipelineState(topic=...)`
  3. Agents 1-6 execute in order, each takes a snapshot
  4. After each agent: state diffed, persisted to JSON
  5. Failure in one stage? Resume with `ara resume <run-id> --from <stage>`
  6. Final stage compiles LaTeX with `latexmk` (skipped if absent)
  7. Outputs: `runs/<id>/state.json`, `report.tex`, `report.pdf`
- **Callout (bottom):** `Determinism by default. FakeLLM + fixed seeds → byte-identical artefacts on re-run.`

### Slide 9/30 — Agent 1: Query Expansion

- **Purpose:** describe the first agent
- **Layout:** standard agent slide (see template below)
- **Section tab:** `Agent 1` / `Query`
- **Title:** `Agent 1 — Query Expansion`
- **h2 (navy):** `What it does`
  - Topic → 6 search-friendly terms via LLM expansion
  - LLM-judge accepts / rejects expansion (sanity gate)
  - Bad expansion → fail-fast, do not waste retrieval budget
- **h2 (navy):** `Inputs / Outputs`
  - In: `topic: str`
  - Out: `expanded_terms: list[str]`, `query_decision: QueryDecision`
- **h2 (red):** `Pitfall avoided`
  - Hallucinated terms with no scholarly footprint → empty retrievals
  - The judge catches this before we hit the network

### Slide 10/30 — Agent 2: Paper Retrieval

- **Section tab:** `Agent 2` / `Retrieval`
- **Title:** `Agent 2 — Paper Retrieval`
- **Two-column.**
- **Left h2 (navy):** `Two sources, one merged list`
  - arXiv Atom feed (no auth, no rate limit beyond politeness)
  - Semantic Scholar JSON API (1 RPS unauthenticated)
  - Dedup by DOI > arXiv-id > title hash
- **Right h2 (navy):** `Robustness`
  - 2 s inter-term sleep respects S2's 1 RPS quota
  - Tenacity retries on HTTP 429 with exponential backoff
  - S2 IP-block fallback path uses arXiv-direct only
- **Callout (red tint):**
  `S2 actually IP-blocked us during the live run. The fallback path saved the demo.`

### Slide 11/30 — Agent 3: PDF Extraction & Injection Defence

- **Section tab:** `Agent 3` / `Extraction`
- **Title:** `Agent 3 — PDF Extraction with Prompt-Injection Defence`
- **Two-column.**
- **Left h2 (navy):** `Why span-level`
  - PyMuPDF span-level (not page-level) lets us inspect every glyph
  - Adversarial PDFs can hide instructions the human eye won't see
- **Right h2 (red):** `Three injection vectors we strip`
  1. **White-on-white text** — same RGB as page background
  2. **Sub-0.5 pt fonts** — too small to render but readable to LLM
  3. **Off-mediabox spans** — text positioned outside the visible page
- **Callout (bottom):**
  `Test fixture: tests/fixtures/papers/adversarial_prompt_injection.pdf —`
  `four planted vectors, all stripped before the LLM sees the text.`

### Slide 12/30 — Agent 4: Indexing

- **Section tab:** `Agent 4` / `Indexing`
- **Title:** `Agent 4 — Chunk, Embed, Index`
- **Two-column.**
- **Left h2 (navy):** `Chunking`
  - Section-aware: respects `Abstract / Introduction / Methods / Results`
  - 500-token chunks with 50-token overlap
  - Overlap preserves boundary context — claim and evidence stay together
- **Right h2 (navy):** `Embeddings + index`
  - `sentence-transformers/all-MiniLM-L6-v2` (80 MB, CPU-fast)
  - L2-normalised embeddings → FAISS `IndexFlatIP` ≡ cosine similarity
  - Exact search: 657 chunks fits trivially; HNSW is over-engineering here

### Slide 13/30 — Agent 5: Hierarchical Analysis

- **Section tab:** `Agent 5` / `Analysis`
- **Title:** `Agent 5 — Hierarchical RAG Analysis`
- **Layout:** half text, half diagram
- **Left:**
  - h2 (navy): `Why hierarchical`
    - Belem 2025: 75 % hallucination at N > 3 in single-call
    - Solution: never let the LLM see > 1 paper at a time per call
  - h2 (navy): `Call budget = 4N + 2`
    - Per paper (×N): summary, methodology, limitations, verifier
    - Cross-paper (×2): methodology matrix, gap synthesis
    - 10 papers → 42 calls. Live run: 39 (2 papers failed early).
- **Right:** see §4.4 — hierarchical RAG diagram

### Slide 14/30 — Agent 6: Report Generation

- **Section tab:** `Agent 6` / `Report`
- **Title:** `Agent 6 — Verified LaTeX Output`
- **Two-column.**
- **Left h2 (navy):** `Templating`
  - Jinja2 → `report.tex` from a single skeleton
  - LaTeX-escape every interpolated string (no `$` accidents)
  - BibTeX keys collision-resolved with `-author-year` suffix
- **Right h2 (navy):** `Compile gate`
  - `latexmk` invoked if available; skipped cleanly if not
  - Final artefacts: `report.tex`, `report.pdf`, `references.bib`
  - Overleaf-compatible (we tested it)

### Slide 15/30 — Tooling Stack

- **Purpose:** show what's under the hood
- **Layout:** 3×3 grid of tool tiles, each with role
- **Title:** `Tooling Stack`
- **Tiles (each 3.00×1.20 in, light-tint fill, navy border):**
  - `LiteLLM` / Provider gateway (Gemini · OpenAI · Anthropic)
  - `Pydantic v2` / Schema-strict (`extra="forbid"`)
  - `FAISS` / Exact cosine (`IndexFlatIP`)
  - `sentence-transformers` / `all-MiniLM-L6-v2`
  - `PyMuPDF` / Span-level PDF parser
  - `Typer + structlog` / CLI + dual-sink logs
  - `Jinja2` / LaTeX templating
  - `tenacity` / 4 / 8 / 16 s backoff
  - `pytest + pytest-socket` / 266 hermetic tests

### Slide 16/30 — Why Hierarchical RAG

- **Purpose:** drive the central insight home
- **Layout:** before/after split + Belem callout
- **Title:** `Why Hierarchical RAG, Not One-Shot Synthesis`
- **Left h2 (red):** `Naïve approach`
  - Stuff all 10 papers into one prompt → ask for a survey
  - Belem et al. 2025: hallucinates 75 % at N > 3
  - LLM blends papers, attributes claims to wrong authors, invents results
- **Right h2 (navy):** `Our approach`
  - Per-paper grounding *first* (4 calls each, schema-pinned)
  - Cross-paper synthesis only after every paper is digested
  - Verifier post-hoc checks every claim against cited passage
- **Callout (full-width, bottom):**
  `Live run: 0 / 8 unsupported verdicts. Hallucination contained.`

### Slide 17/30 — Reproducibility

- **Purpose:** show this is more than a one-off demo
- **Layout:** three pillars (3 columns)
- **Title:** `Reproducibility by Design`
- **Pillar 1 — `FakeLLM`:**
  - SHA-256[:16] of prompt → canned response fixture
  - Unknown prompt? Crash with `UnknownPromptError`
  - Catches prompt drift loudly at test time
- **Pillar 2 — Golden state:**
  - `tests/fixtures/golden_state.json` pinned to a deterministic run
  - E2E smoke test asserts byte-identical state on replay
- **Pillar 3 — Retry & isolation:**
  - 4 / 8 / 16 s backoff on 503 / 429 / InternalServer
  - Per-paper failure isolated; pipeline survives 2 / 10 fails

### Slide 18/30 — Live Run Results

- **Purpose:** ground claims in measured numbers
- **Layout:** title + KPI strip + stage timing chart
- **Title:** `Live Run — 2026-05-02 — Real Numbers`
- **KPI strip (5 boxes across, each 1.85×0.90 in, navy border):**
  - `29 papers` retrieved
  - `551 chunks` indexed
  - `10 / 10` analysed
  - `42 LLM calls`
  - `≈ $0.03` total cost
- **Stage timing chart:** see §4.5 — horizontal bar chart of stage
  durations (Retrieval 162.7 s, Extraction 47.3 s, Indexing 10.5 s,
  Analysis 110.7 s, Report 0.02 s). Total 331 s wall-clock.
- **Caption (10 pt muted):** `Run-id 20260502-190241-what-is-rag · authenticated S2 · gemini-2.5-flash-lite`

### Slide 19/30 — Example Research Questions ARA Can Answer

- **Purpose:** show the breadth — this is not a one-trick pony
- **Layout:** centered headline + 6-tile grid (3×2)
- **Title:** `Example Research Questions`
- **Headline (16 pt body dark, centered, y=1.10):**
  `Any natural-language research question fits.`
- **6 tiles (each 3.00 × 1.30 in, light-tint fill, navy border, navy bold title 12 pt + body 11 pt body-dark):**
  - **CS / ML** — *"What is RAG?"* / Retrieval-augmented generation
  - **CS / Security** — *"How are LLMs jailbroken?"* / Adversarial prompt research
  - **Civil Eng.** — *"What are recent advances in self-healing concrete?"*
  - **Medicine** — *"Are GLP-1 agonists effective for cardiovascular outcomes?"*
  - **Climate** — *"Direct air capture cost trajectories 2020–2025"*
  - **Education** — *"Does retrieval practice improve long-term retention in K-12?"*
- **Caption (centered, bottom, 11 pt secondary blue):**
  `The headline run we'll walk through next: "What is RAG?"`

### Slide 20/30 — Walkthrough: "What is RAG?" — Stages 1-3

- **Purpose:** make the pipeline concrete by following one query
- **Layout:** three stacked panels, each labelled with stage number + name + measured output
- **Title:** `Walkthrough — "What is RAG?" — Discovery & Extraction`
- **Panel 1 (y=1.05, h=1.20 in, secondary-blue header):**
  - h2: `Stage 1 · Query Expansion (3.2 s)`
  - bullets:
    - LLM produced 3 search terms: *"retrieval-augmented generation"*, *"RAG explained"*, *"large language models RAG"*
    - LLM-judge accepted all three — none flagged as off-topic
- **Panel 2 (y=2.30, h=1.20 in, secondary-blue header):**
  - h2: `Stage 2 · Retrieval (162.7 s)`
  - bullets:
    - Authenticated Semantic Scholar: 30 hits across 3 terms, 0 retries, 0 IP-block events
    - 29 papers after dedup; 10 PDFs downloaded (8 via arXiv-fallback, 2 direct from S2); 19 abstract-only
- **Panel 3 (y=3.55, h=1.20 in, secondary-blue header):**
  - h2: `Stage 3 · PDF Extraction (47.3 s)`
  - bullets:
    - PyMuPDF span-level extraction across 10 PDFs (13.55 MB aggregate)
    - **31 hidden injection-vector spans stripped** — defense fired on real-world papers, not just on the synthetic test fixture

### Slide 21/30 — Walkthrough: "What is RAG?" — Stages 4-6

- **Purpose:** complete the walkthrough through analysis and report
- **Layout:** three stacked panels (same template as slide 20)
- **Title:** `Walkthrough — "What is RAG?" — Indexing, Analysis & Report`
- **Panel 1 (y=1.05, h=1.20 in):**
  - h2: `Stage 4 · Indexing (10.5 s)`
  - bullets:
    - 551 chunks across 10 papers (~55 chunks/paper avg, MiniLM-L6-v2 embeddings)
    - FAISS `IndexFlatIP` exact search; normalize_embeddings=True so inner-product == cosine
- **Panel 2 (y=2.30, h=1.20 in):**
  - h2: `Stage 5 · Hierarchical Analysis (110.7 s)`
  - bullets:
    - 4N + 2 = 42 LLM calls: 30 per-paper (10 × 3 succeeded fully) + 2 cross-paper + 10 verifier
    - **0 analysis failures, 0 unsupported verifier verdicts** — citation-validator hardening eliminated the 38 % drop-rate observed before the fix
- **Panel 3 (y=3.55, h=1.20 in):**
  - h2: `Stage 6 · Report Generation (0.02 s)`
  - bullets:
    - Jinja2 LaTeX template, 29.69 KB `report.tex`, 10 BibTeX keys, **0 unresolved citations**
    - Compile gate: `latexmk` invoked when present; clean skip otherwise → Overleaf-compatible

### Slide 22/30 — Anatomy of the Output

- **Purpose:** show *what the user actually gets* — the final report's structure
- **Layout:** left half text (what files), right half a stylised LaTeX excerpt
- **Title:** `What ARA Hands Back to the Researcher`
- **Left column (4.50 in wide):**
  - h2 (navy): `Files written to runs/<run-id>/`
  - bullets:
    - `report.tex` — citation-grounded literature review (29.69 KB)
    - `references.bib` — 10 BibTeX entries, collision-resolved
    - `state.json` — typed `PipelineState` with every measurement
    - `summaries/<paper-id>.json` — per-paper structured summaries
    - `comparison.json` — methodology matrix across all papers
    - `gaps.json` — research-gap synthesis with citations
    - `pdfs/<paper-id>.pdf` — every downloaded PDF preserved
    - `run.log` — JSON-lines log, every event tagged with `run_id` + `git_sha`
- **Right column (4.50 in wide), simulated LaTeX excerpt in monospace 10 pt body-dark, with light-tint background:**
  ```
  \section{Bounding Hallucinations: ...
   Merlin-Arthur Protocols (2025)}
  
  \textbf{Authors.} ...
  
  \textbf{Objective.} The objective
   is to bound hallucinations in
   RAG systems using Merlin-Arthur
   protocols, providing information-
   theoretic guarantees ...
   \cite{wenger_2025_bounding}
  
  \textbf{Key Findings.}
  \begin{itemize}
    \item M/A training improves
     accuracy, completeness ...
     [P2b6af398...-1]
  \end{itemize}
  ```
- **Caption (centered, bottom, 11 pt secondary blue):**
  `Every claim links to a specific passage in a specific paper.`

### Slide 23/30 — How This Helps a Researcher

- **Purpose:** translate the artefacts into research-workflow value
- **Layout:** four-tile 2×2 grid, each tile 4.50 × 1.55 in, light-tint fill, navy border
- **Title:** `What a Researcher Gets in 5 Minutes`
- **Tile 1 — `Day-1 literature scan`:**
  Survey 30 papers in five minutes; would take a graduate student a full week.
- **Tile 2 — `Methodology matrix on demand`:**
  `comparison.json` lays out *who used which dataset, technique, and metric*. The first table you'd otherwise build by hand on day three of a project.
- **Tile 3 — `Research-gap synthesis`:**
  `gaps.json` distils every paper's stated limitations into a synthesis of *what is not yet known* — the natural starting point for a thesis chapter.
- **Tile 4 — `Citation hygiene from the start`:**
  Every claim in the LaTeX traces to a specific passage. The verifier flags anything it can't ground. No silent hallucinations to discover at peer review.
- **Callout (full-width, bottom, navy tint):**
  `ARA does not replace the researcher's judgment — it removes the mechanical 80 % of the work so the human can spend the saved time on synthesis and originality.`

### Slide 24/30 — Evaluation: Retrieval Quality

- **Purpose:** show retrieval is grounded
- **Layout:** stat block + interpretation
- **Title:** `Evaluation — Retrieval Quality`
- **Left h2 (navy):** `Inner-product scores (cosine on normalised vectors)`
  - Range: **0.432 – 0.682** across 25 returned chunks
  - Mean: **0.603**
  - MiniLM-L6 empirical noise floor: 0.30 – 0.40
  - Conclusion: retrieved passages are **well above** the noise floor
- **Right h2 (navy):** `What we did not measure`
  - No human relevance judgements (out of project scope)
  - No comparison against BM25 baseline (future work)
  - Mean score is a *necessary* not *sufficient* condition

### Slide 25/30 — Evaluation: Verifier & Citation Grounding

- **Purpose:** show generation is grounded
- **Layout:** stat block + interpretation
- **Title:** `Evaluation — Verifier & Citation Grounding`
- **Bullets (full-width):**
  - **8 verifier calls** (one per analysed paper)
  - **0 unsupported verdicts** — every claim survived re-checking
  - Each claim is re-verified against the *cited passage*, not just any passage
  - `VerifierVerdict` schema: `{claim, passage_id, verdict ∈ {supported, unsupported}, reason}`
- **Callout (navy tint):**
  `0 / 8 is a strong signal but a small sample.`
  `The methodology generalises; the absolute number does not.`

### Slide 26/30 — Failure Modes Observed

- **Purpose:** be honest about what broke
- **Layout:** four red-bordered tiles in 2×2
- **Title:** `Failure Modes We Hit (and Disclosed)`
- **Tiles (each 4.50×1.55 in, accent-fill `#F5E8E8`, red border):**
  - **2 / 10 papers failed analysis** — `ValidationError` on citation
    format; per-paper isolation contained it
  - **Semantic Scholar IP-blocked us** — built arXiv-direct fallback path
  - **Gemini free-tier is 20 RPD per model** (not 500) — burned through
    it twice debugging; switched to Tier-1
  - **Gemini-3-flash thinking-token tax** — 16 K hidden tokens for 2 KB
    JSON; switched to flash-lite

### Slide 27/30 — Lessons Learned

- **Purpose:** synthesise the experience
- **Layout:** numbered list (full-width, 6 items)
- **Title:** `Lessons Learned`
- **Bullets:**
  1. **Single-writer invariant earns its keep** the first time you have to
     resume from a mid-run failure
  2. **`FakeLLM` with hash-keyed fixtures** catches prompt drift you'd
     otherwise discover in production
  3. **Hierarchical > one-shot** for any multi-document task — even at
     N = 10, single-call would have hallucinated badly
  4. **Schema-strict Pydantic + LiteLLM `response_format`** pins JSON
     contracts; prompt regressions can no longer silently degrade output
  5. **External APIs are unreliable** by default — your retry & fallback
     story is the product, not an afterthought
  6. **Document failure modes in the report.** Honesty about 2 / 10
     failures is more credible than a tidy 10 / 10 fiction

### Slide 28/30 — Limitations & Future Work

- **Purpose:** show we know what we did not solve
- **Layout:** two-column
- **Title:** `Limitations & Future Work`
- **Left h2 (red):** `Limitations`
  - PDF availability: only ~10 % of S2 indexed papers have open PDFs
  - Section-detection is regex-based; brittle on non-standard layouts
  - Verifier sample is small (8 papers); not yet a statistical claim
  - No human-judgement evaluation of summary quality
- **Right h2 (navy):** `Future work`
  - GROBID for structured PDF parsing → cleaner section detection
  - Human-rated relevance and faithfulness on a 100-paper benchmark
  - BM25 + dense hybrid retrieval (compare against pure-dense)
  - Adaptive call-budget — skip the verifier on high-confidence claims

### Slide 29/30 — Demo

- **Purpose:** prep the audience for the live demo
- **Layout:** three-pillar (3 columns)
- **Title:** `Demo — Three Paths to See ARA Run`
- **Pillar 1 — Offline (4 s):**
  - `uv run ara run "anything" --cached-only`
  - Replays artefacts from `demo-cache.tar.gz`
  - No network, no API key — bulletproof for the projector
- **Pillar 2 — Live (3 min):**
  - `uv run ara run "your topic"`
  - Real Gemini call, real arXiv & S2 fetch
  - Needs `ARA_GEMINI_API_KEY` and ~$0.03
- **Pillar 3 — Inspect prior run:**
  - `runs/20260427-173618-...` already on disk
  - Open `state.json`, `report.tex`, the verifier verdicts directly
- **Callout (bottom):**
  `Plan: pillar 1 first (guaranteed). Pillar 2 if Wi-Fi cooperates.`

### Slide 30/30 — Q & A

- **Purpose:** close
- **Layout:** centered headline + repo URL + thanks
- **Title:** `Questions?`
- **Body (centered, y=2.40, 16 pt body dark):**
  `github.com/souparna21/agentic-research-assistant (private)`
- **Body (centered, y=2.95, 14 pt secondary blue):**
  `Thank you.`
- **Body (centered, y=3.50, 12 pt muted gray):**
  `Souparna · Damor · Akash · IIT Bombay · IE624 · 2026`

---

## §4. Diagram specifications

Draw all diagrams natively with `python-pptx` shapes — do **not** insert
external images. Use only the colours from §2.2.

### §4.1 — System Overview (slide 6)

Six rounded-rectangle boxes in a horizontal flow at y=2.20:

```
┌───────┐    ┌───────┐    ┌────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐
│ Topic │ →  │Query  │ →  │Retrieve│ →  │ Extract  │ →  │ Index    │ →  │Analyse │ → Report → PDF
│ in    │    │  1    │    │   2    │    │    3     │    │   4      │    │   5    │     6
└───────┘    └───────┘    └────────┘    └──────────┘    └──────────┘    └────────┘
```

Specifications:

- Each agent box: `MSO_SHAPE.ROUNDED_RECTANGLE`, 1.30 × 0.80 in, fill
  `#E8EEF5`, border navy, navy bold text
- Agent number above box (small navy circle, 0.30 in dia)
- 1-line description below each box (10 pt secondary blue, centered)
- Arrows: thin navy connectors with arrow heads, between boxes
- Endpoints "Topic" (left) and "PDF + LaTeX" (right) styled as plain
  text with navy bold, no border
- Layout: horizontal centerline at y=2.55, x from 0.40 to 9.60,
  spacing computed evenly across 8 elements (input + 6 agents + output)

### §4.2 — Field-Ownership Matrix (slide 7)

Small table, 4.30 in wide, on the right half of slide 7. Header row =
agent names abbreviated (Q, R, E, I, A, Rep). Rows = key fields. ✓ in the
single owner cell, blank elsewhere.

| Field | Q | R | E | I | A | Rep |
|---|---|---|---|---|---|---|
| `topic` | (input) |  |  |  |  |  |
| `expanded_terms` | ✓ |  |  |  |  |  |
| `papers` |  | ✓ |  |  |  |  |
| `extracted_texts` |  |  | ✓ |  |  |  |
| `chunks` |  |  |  | ✓ |  |  |
| `per_paper_summaries` |  |  |  |  | ✓ |  |
| `report_pdf_path` |  |  |  |  |  | ✓ |

Render as a `MSO_SHAPE.RECTANGLE` grid with thin gray borders. Header row
fill `#E8EEF5`, header text navy bold, body 11 pt body dark. ✓ in
secondary blue bold.

### §4.3 — Run Flow Timeline (slide 8)

7 steps in a horizontal flow at y=1.50, with circled step numbers above
each step description:

```
 ①        ②         ③         ④         ⑤         ⑥        ⑦
 CLI →   State  →  Agents → Snapshot → Resume → latexmk → Outputs
        built     1-6        + JSON     ready    compile   to disk
        in mem    in order   persist
```

Each step: a small navy circle (0.40 in dia) with white step number,
1-line label below (12 pt body dark), centered. Connect with thin gray
arrows.

### §4.4 — Hierarchical RAG Diagram (slide 13)

Right half of slide 13, ~4.30×3.60 in canvas. Tree structure:

```
                  ┌─────────────────────┐
                  │  Cross-paper:       │
                  │  Methodology matrix │
                  │  Gap synthesis      │
                  └─────────▲───────────┘
                            │ (only after all per-paper
                            │  reasoning is grounded)
              ┌─────────────┼─────────────┐
              │             │             │
        ┌─────┴─────┐ ┌─────┴─────┐ ┌─────┴─────┐
        │ Paper 1   │ │ Paper 2   │ │   ...     │
        │ summary   │ │ summary   │ │           │
        │ + method  │ │ + method  │ │           │
        │ + limit   │ │ + limit   │ │           │
        │ + verify  │ │ + verify  │ │           │
        └─────▲─────┘ └─────▲─────┘ └─────▲─────┘
              │             │             │
        per-paper       per-paper     per-paper
        chunks          chunks        chunks
```

Specs:

- Top synthesis box: 3.60 × 0.85 in, light-tint fill, navy border, navy
  bold title.
- Three per-paper boxes: 1.20 × 1.10 in each, light-tint fill, navy
  border, body-dark 10 pt text.
- Source-chunks layer: small gray-fill rectangles (0.90 × 0.30 in) below
  each paper box, captioned "chunks" in 9 pt muted gray.
- Connectors: thin navy arrows pointing **upward** (chunks → per-paper →
  cross-paper).
- Caption below diagram (11 pt secondary blue, centered):
  `Per-paper grounding before any cross-paper synthesis.`
  `4N + 2 LLM calls. Verifier post-checks every claim.`

### §4.5 — Stage Timing Bar Chart (slide 18)

Native python-pptx chart (`XL_CHART_TYPE.BAR_CLUSTERED`), 6.20 × 2.00 in,
positioned at (0.40, 3.00):

| Stage | Duration (s) |
|---|---|
| Retrieval | 18.4 |
| Extraction | 40.4 |
| Indexing | 8.8 |
| Analysis | 99.7 |
| Report | 0.02 |

- Bars filled `#2C4A7C` secondary blue
- Category axis labels in 10 pt body dark
- Data labels at end of each bar (10 pt, navy)
- No legend (single series)
- Title above chart: `Wall-clock per stage (sec) — total 168 s`
  (12 pt navy bold)

---

## §5. Implementation hints (`python-pptx` patterns)

### 5.1 Color helpers

```python
from pptx.dml.color import RGBColor

NAVY            = RGBColor(0x0D, 0x2A, 0x4F)
SECONDARY_BLUE  = RGBColor(0x2C, 0x4A, 0x7C)
ACCENT_RED      = RGBColor(0x8B, 0x00, 0x00)
BODY_DARK       = RGBColor(0x1A, 0x1A, 0x1A)
MUTED_GRAY      = RGBColor(0x55, 0x55, 0x55)
LIGHT_TINT      = RGBColor(0xE8, 0xEE, 0xF5)
ACCENT_FILL     = RGBColor(0xF5, 0xE8, 0xE8)
RULE_LINE       = RGBColor(0xC9, 0xD2, 0xDD)
WHITE           = RGBColor(0xFF, 0xFF, 0xFF)
```

### 5.2 Reusable text-box helper

```python
def add_text(slide, x, y, w, h, text, *, size=12, bold=False,
             color=BODY_DARK, font="Calibri", align=PP_ALIGN.LEFT,
             v_align=MSO_ANCHOR.TOP):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y),
                                     Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = v_align
    tf.margin_left = tf.margin_right = Inches(0.05)
    tf.margin_top = tf.margin_bottom = Inches(0.02)
    p = tf.paragraphs[0]
    p.alignment = align
    run = p.add_run()
    run.text = text
    run.font.name = font
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shape
```

### 5.3 Bullet list helper

```python
def add_bullets(slide, x, y, w, h, items, *, size=12,
                color=BODY_DARK, font="Calibri"):
    shape = slide.shapes.add_textbox(Inches(x), Inches(y),
                                     Inches(w), Inches(h))
    tf = shape.text_frame
    tf.word_wrap = True
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.LEFT
        p.line_spacing = 1.15
        p.space_after = Pt(3)
        run = p.add_run()
        run.text = "•  " + item
        run.font.name = font
        run.font.size = Pt(size)
        run.font.color.rgb = color
    return shape
```

### 5.4 Title bar + rule line + footer

```python
def add_title(slide, text):
    add_text(slide, 0.40, 0.22, 9.20, 0.55, text,
             size=24, bold=True, color=NAVY)
    rule = slide.shapes.add_shape(MSO_SHAPE.RECTANGLE,
        Inches(0.40), Inches(0.85), Inches(9.20), Inches(0.015))
    rule.fill.solid(); rule.fill.fore_color.rgb = RULE_LINE
    rule.line.fill.background()

def add_footer(slide, page_num, total=30):
    add_text(slide, 0.40, 5.30, 7.50, 0.25,
        "IE624 — Agentic Research Assistant · IIT Bombay",
        size=8, color=MUTED_GRAY)
    add_text(slide, 9.10, 5.30, 0.70, 0.25,
        f"{page_num}/{total}",
        size=9, color=MUTED_GRAY, align=PP_ALIGN.RIGHT)
```

### 5.5 Rounded-box diagram primitive

```python
def add_box(slide, x, y, w, h, *, fill=LIGHT_TINT, border=NAVY,
            text="", size=12, bold=True, color=NAVY,
            shape_type=MSO_SHAPE.ROUNDED_RECTANGLE):
    shp = slide.shapes.add_shape(shape_type,
        Inches(x), Inches(y), Inches(w), Inches(h))
    shp.fill.solid(); shp.fill.fore_color.rgb = fill
    shp.line.color.rgb = border
    shp.line.width = Pt(1.0)
    tf = shp.text_frame
    tf.word_wrap = True
    tf.vertical_anchor = MSO_ANCHOR.MIDDLE
    p = tf.paragraphs[0]
    p.alignment = PP_ALIGN.CENTER
    run = p.add_run()
    run.text = text
    run.font.name = "Calibri"
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    return shp
```

### 5.6 Connector arrow primitive

```python
from pptx.enum.shapes import MSO_CONNECTOR

def add_arrow(slide, x1, y1, x2, y2, color=NAVY, width=1.25):
    conn = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT,
        Inches(x1), Inches(y1), Inches(x2), Inches(y2))
    conn.line.color.rgb = color
    conn.line.width = Pt(width)
    line_elem = conn.line._get_or_add_ln()
    from pptx.oxml.ns import qn
    from lxml import etree
    tail = etree.SubElement(line_elem, qn("a:tailEnd"),
        attrib={"type": "triangle", "w": "med", "len": "med"})
    return conn
```

### 5.7 Per-slide skeleton

```python
def slide_with_chrome(prs, page_num, title):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank
    add_title(slide, title)
    add_footer(slide, page_num, total=30)
    return slide
```

Slide 1 is the only exception (no footer, no rule line — centered title
block only).

### 5.8 Required imports

```python
from pptx import Presentation
from pptx.util import Inches, Pt
from pptx.dml.color import RGBColor
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.enum.shapes import MSO_SHAPE, MSO_CONNECTOR
from pptx.chart.data import CategoryChartData
from pptx.enum.chart import XL_CHART_TYPE, XL_LABEL_POSITION
```

---

## §6. Output requirements

### 6.1 Where to write

- Script: `scripts/build_presentation_v2.py` (new file; do not delete the
  old `scripts/build_presentation.py`)
- Deck: `ie624_presentation.pptx` (overwrite the existing one)

### 6.2 How to invoke

```bash
uv run --with python-pptx python scripts/build_presentation_v2.py
```

The script must be idempotent — running it twice produces the same deck.

### 6.3 Quality checklist (verify before declaring done)

- [ ] Exactly **30 slides**.
- [ ] Slide 1 has no footer or page number; slides 2-30 have both.
- [ ] Page-number format is **`X/30`** on slides 2 through 30.
- [ ] All titles are 24 pt Calibri bold navy `#0D2A4F` (slide 1 title is
      centered & 28 pt; this is the one exception).
- [ ] Title bars on slides 2-30 have a thin rule line below at y=0.85.
- [ ] Footer attribution exact text:
      `IE624 — Agentic Research Assistant · IIT Bombay`
- [ ] Every diagram is built from native shapes — no images embedded.
- [ ] Slide 6 system-overview diagram is rendered (six agent boxes with
      arrows + input/output endpoints).
- [ ] Slide 7 has the field-ownership matrix.
- [ ] Slide 13 has the hierarchical-RAG tree diagram.
- [ ] Slide 18 has the bar chart with the five stage durations.
- [ ] No text overflows its bounding box on any slide (test by opening
      the file in LibreOffice / Keynote / PowerPoint).
- [ ] No bullet has more than 12 words.
- [ ] No content slide has more than 6 bullets per block.
- [ ] All numbers cited match §1.6 exactly (29 papers / 10 PDFs /
      551 chunks / 42 LLM calls / 331 s wall-clock / $0.03 cost /
      0.432–0.682 mean 0.603 retrieval-probe scores / 0 unsupported /
      31 injection spans stripped).
- [ ] Colours match §2.2 hex values exactly.
- [ ] Font is Calibri on every run on every slide.

### 6.4 What to do if something is ambiguous

- For numbers and facts: **always use §1.6 verbatim**. Do not look them
  up elsewhere.
- For team-member surnames: read `report.tex` for the author block. If
  not present, leave first names only.
- For visual judgement calls (exact pixel of a diagram element): **err
  toward whitespace**. Prefer a slightly empty slide over an overflowing
  one.

---

## §7. Final reminder

The reference style file `seminar.pptx` already lives in the project root.
You can open it with `python-pptx` to confirm any geometry detail, but the
design system in §2 already encodes everything you need. The audience is
the IE624 evaluation panel; the deck must look professional, not flashy.
Simple, informative, and dense-but-readable beats animated, gradient-heavy
slideware every time.

When you finish, print a one-line summary:
`Generated ie624_presentation.pptx — 30 slides, <KB> KB`.
