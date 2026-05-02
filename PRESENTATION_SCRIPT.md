# IE624 Presentation Script — 20 minutes total

**Team:** Souparna Bhowmik (25D1386), Damor Jaydipkumar (22B4221), Akash Sansugu Palaniswami (21D171001)

## Time budget

| Speaker | Slides | Allocated | Buffer |
| --- | --- | --- | --- |
| **Souparna** — Project Lead, Architecture | 1–7 (7 slides) | **6:30** | 30 s |
| **Damor** — Implementation Deep-Dive | 8–13 (6 slides) | **6:30** | 30 s |
| **Akash** — Evaluation, Lessons, Demo Plan | 14–20 (7 slides) | **6:30** | 30 s |
| Q&A and live demo buffer | — | **0:30** | — |
| **Total** | 20 slides | **20:00** | 1:30 |

Each speaker should rehearse to fit slightly under their allocation so the 1:30 buffer can absorb questions or transitions.

---

## SOUPARNA — Slides 1 to 7 (6:30)

### Slide 1 — Title (0:30)

> Good morning. We're presenting our IE624 project, the **Agentic Research Assistant** — an automated multi-paper literature review system built around a six-agent RAG pipeline. I'm Souparna; with me are Damor and Akash. Over the next twenty minutes, we'll show you the problem we tackled, our architecture, the live results we measured, and what we learned. I'll cover the project context and architecture, Damor will go through the implementation, and Akash will walk through evaluation and the live demo.

### Slide 2 — The Problem (1:00)

> Manual literature reviews don't scale. arXiv added over two hundred thousand papers in 2024 alone, and computer science papers double every two years. A single survey takes weeks of expert reading and cross-comparison. Existing AI tools like ChatGPT and Gemini hallucinate citations — Belem et al. 2025 measured up to a seventy-five percent hallucination rate when these models try to summarise multiple documents. Generic RAG systems lack per-claim grounding and cross-paper synthesis, and commercial tools like Elicit and ResearchRabbit are paid and closed source. Our research question was: **can we build a free, open, end-to-end pipeline that grounds every single generated claim back to a specific passage in a specific paper?**

### Slide 3 — Goals & Constraints (1:00)

> Our deliverable is a system that takes a natural-language question and produces a compilable LaTeX literature review with BibTeX citations, where every claim is traceable to its source. We had three hard constraints. **Zero budget** — we used only free-tier APIs: Gemini Flash, Semantic Scholar, and arXiv. **Three students, one semester** — so the architecture has to favour simplicity over novelty. And **local-first** — no cloud servers, no managed databases, no deployment infrastructure. We deliberately ruled out four directions: GraphRAG, RLHF fine-tuning, a web UI, and autonomous multi-agent coordination — each adds complexity that a course project can't justify.

### Slide 4 — System Architecture (1:30)

> This is the system at a glance — six agents arranged as a strict sequential pipeline. The **Query agent** uses Gemini to expand the user's question into three to five search terms. **Retrieval** queries Semantic Scholar, falls back to arXiv for PDFs, and downloads them. **Extraction** uses PyMuPDF with a span-level prompt-injection defense. **Indexing** chunks the text, embeds with MiniLM-L6-v2, and builds a FAISS inner-product index. **Analysis** runs hierarchical RAG with a per-claim verifier. **Report** assembles everything into LaTeX with proper BibTeX citations. The grey band underneath is the critical piece — a typed Pydantic `PipelineState` that is the only inter-agent communication channel. Every stage reads it, writes only to its registered fields, and the orchestrator snapshot-diffs to enforce the contract.

### Slide 5 — Why Sequential, Not Autonomous (1:00)

> This sequential design is a deliberate response to research. Cemri et al. 2025 — the MAST paper — studied over two hundred multi-agent LLM failures and found that **forty-two percent are specification errors** — agents disagreeing on shared state shape — and another twenty-three percent are inter-agent communication errors. Autonomous coordination compounds these. Our defence is the typed `PipelineState` plus a `FIELD_OWNERS` map that registers exactly which fields each stage may write. The orchestrator snapshot-diffs non-owned fields before and after every stage and raises `FieldOwnershipViolation` on any cross-write. Across 34 plans of development and 263 tests, **zero invariant leaks reached the merge queue**.

### Slide 6 — Technology Stack (1:00)

> Quick tour of the stack. The LLM is Gemini Flash routed through LiteLLM, so swapping to GPT-4o-mini is a one-line config change. Embeddings are MiniLM-L6-v2 — five times faster than mpnet on CPU with adequate quality at our corpus scale. The vector store is FAISS `IndexFlatIP`, sub-millisecond at this scale, no tuning required. Discovery is Semantic Scholar with arXiv fallback. PDF parsing is PyMuPDF plus pymupdf4llm for section-aware Markdown. The orchestrator is custom Python — not LangGraph — because the sequential pipeline is exactly the failure-mitigation we designed for. Templates use Jinja2 for both prompts and LaTeX, and tests use pytest with pytest-socket and respx.

### Slide 7 — Stages 1 and 2 (1:00)

> The first two stages set up the corpus. **Query expansion** is one Gemini call that turns "How does RAG reduce hallucination?" into three to five focused search terms. We also use an LLM-as-judge filter to reject off-topic expansions. **Retrieval** then queries Semantic Scholar with tenacity backoff at two, four, eight, and sixteen seconds on rate-limit errors. For each paper, we follow a fallback chain: try the Semantic Scholar `openAccessPdf`, then arXiv via the `export.arxiv.org` subdomain, then graceful degradation to abstract-only. We also detect withdrawn papers via comment substring and a tiny-PDF heuristic. Two-second sleeps between term searches keep us inside Semantic Scholar's one-RPS unauthenticated quota. Damor will now take you through what happens with those PDFs once we have them.

---

## DAMOR — Slides 8 to 13 (6:30)

### Slide 8 — Stage 3: Extraction with Injection Defense (1:30)

> Thanks Souparna. I'll dive into the implementation. Stage three is **PDF extraction** using PyMuPDF and pymupdf4llm to get section-aware Markdown — preserving headings, paragraph structure, and page boundaries. The interesting part is the **prompt-injection defense**. Adversarial PDFs can hide text from human readers but feed it to LLMs. We strip four kinds of hidden text at the span level: white-on-white text where the colour delta is below 0.05, sub-half-point fonts that are invisible at any zoom, off-mediabox spans that fall outside the printable page, and zero-bbox spans that are collapsed to a point. On our live ten-paper run, we **stripped twenty-three hidden spans and one hundred thirty-three header/footer lines** — the defence is firing on real corpus papers.

### Slide 9 — Stage 4: RAG Indexing (1:00)

> Stage four turns the extracted text into a searchable index. We chunk at roughly five hundred tokens with fifty-token overlap, splitting at paragraph and section boundaries where detectable. We chose section-aware over naive chunking because research shows naive faithfulness drops to about 0.47 versus 0.79 with structure-aware splitting. Each chunk is embedded with sentence-transformers `all-MiniLM-L6-v2` — three hundred eighty-four-dimensional vectors, normalised on encode. Chunks are deduplicated by hash before indexing. We use FAISS `IndexFlatIP` on those normalised vectors — inner product on L2-normalised vectors is mathematically equivalent to cosine similarity, so we get cosine retrieval at sub-millisecond latency. On our live run, this produced six hundred fifty-seven chunks across ten papers, with a range of nineteen to one hundred fifty-nine chunks per paper.

### Slide 10 — Hierarchical RAG Analysis (1:30)

> This is the heart of the system. We mitigate Belem 2025's seventy-five-percent hallucination problem with a hierarchical structure called **4N+2**. For each of N papers, we make four LLM calls: a structured **summary** with objective, findings, and limitations; a **methodology extraction** with datasets, techniques, and metrics; a **limitations and future work** extraction; and a **verifier** that re-checks every generated claim against its cited passage at temperature zero. Then once across all papers, two **cross-paper synthesis** calls produce the methodology comparison matrix and the gap analysis. For ten papers that's forty-two calls total — well-bounded and predictable. For our live run we measured thirty-eight calls because two papers failed validation at the per-paper layer, which I'll cover next.

### Slide 11 — Per-Claim Citations and the Verifier (1:00)

> Every claim in our output carries a citation in the format **`[P<paper_id>-N]`** — we pin the paper ID inside the citation marker itself, which makes attribution drift impossible during cross-paper synthesis. This is enforced by a Pydantic `field_validator` on the `Claim.citations` field. We also use schema-strict JSON output via `extra="forbid"` — any extra field or missing field raises `ValidationError`, which our per-paper try-except catches without crashing the stage. The verifier itself is a separate LLM call per paper at temperature zero that reads each generated claim and its cited passage and votes supported, partial, or unsupported. Unsupported claims get an `[UNVERIFIED]` tag prefix in the final report. **On our live run, zero claims came back unsupported across eight verifier calls** — one hundred percent grounded.

### Slide 12 — Stage 6: LaTeX Report Generation (1:00)

> Stage six assembles everything into LaTeX. We use Jinja2 with custom delimiters — `<% %>` for blocks and `<< >>` for variables — because LaTeX itself uses curly braces. Our `latex_escape` filter handles backslash-first ordering to prevent double-escaping. **BibTeX key collisions** are a real correctness problem: two 2024 papers by authors with the same surname and the same first title word would produce the same key. Our resolver appends `_2`, `_3` suffixes in stable deterministic order. We compile with `latexmk -pdf -halt-on-error` and run a `[?]` scan to catch any unresolved citations after BibTeX runs. On the live run we produced a 24.9 KB report.tex with eight BibTeX keys, zero collisions, and zero unresolved citations.

### Slide 13 — Live Run Results (1:30)

> Here are the **measured numbers** from our live end-to-end run on April 28th. Ten papers retrieved from arXiv, ten PDFs downloaded totalling twenty megabytes, no abstract-only fallbacks. Six hundred fifty-seven chunks indexed. Thirty-nine total LLM calls, distributed as one for query expansion, twenty-eight for per-paper analysis, two for cross-paper synthesis, and eight for verification. End-to-end wall clock was one hundred sixty-eight seconds — under three minutes — on CPU. Eight of ten papers passed analysis cleanly; two failed with citation-format issues, which I'll let Akash explain because they tie directly into our evaluation. **Total cost on Tier 1 billing: about three cents.** Akash, over to you.

---

## AKASH — Slides 14 to 20 (6:30)

### Slide 14 — Evaluation: Retrieval Quality (1:00)

> Thanks Damor. We evaluated the system on the same ten-paper live run. For retrieval quality, we used a **five-query test set** spanning the main topic, methodology, datasets, limitations, future work, and a specific multi-hop query. Top-five retrieval each. The measured FAISS inner-product scores on our L2-normalised vectors ranged from **0.432 to 0.682, with a mean of 0.603 across all twenty-five returned chunks**. For context, MiniLM-L6-v2's empirical cosine-similarity floor on unrelated passages sits around 0.30 to 0.40 — so our mean of 0.603 sits comfortably above the noise floor. This validates the choice of `IndexFlatIP` with normalised embeddings for our corpus scale.

### Slide 15 — Evaluation: Grounding and Failure Modes (1:30)

> Three observed failure modes from the live run. **First**, the verifier returned zero unsupported verdicts across eight calls — one hundred percent grounded. **Second**, two of ten papers — twenty percent — failed per-paper analysis: one paper emitted a citation with missing brackets, the other emitted a `Claim` with an empty citations list. Our schema-strict validation caught both, and per-paper try-except kept the stage healthy. The remaining eight papers were summarised, methodology-extracted, verified, and rendered cleanly — exactly as designed. **Third failure mode** — Semantic Scholar IP-rate-limited us with sticky 429s for hours. The system fell back to arXiv-direct discovery via a workaround we built, and produced a usable corpus. We documented the workaround as a known operational reality of free-tier APIs.

### Slide 16 — Engineering Practices (1:00)

> A few engineering practices that paid off. We followed strict **test-driven development** — every plan landed RED, then GREEN, then a metadata commit; every PR ships green. We have **two hundred sixty-three tests**, all offline and deterministic, with zero flakes. We use `pytest-socket` to disable network globally during tests, and our `FakeLLM` keys responses by SHA-256 of the prompt — so any whitespace change in a Jinja template fails loudly instead of silently producing degraded output. **Idempotent stages** mean each stage skips if its output artifact already exists; you delete the artifact to invalidate. And we ship a **8.4 KB committed `demo-cache.tar.gz`** so `ara run --cached-only` produces a complete pipeline output in four seconds — zero network, zero API key, zero risk.

### Slide 17 — Lessons Learned (1:00)

> Five surprises from the project. **One** — the `latexmk` gate disrupted CI on machines without TeX; we added a `@pytest.mark.latex` marker for graceful skip. **Two** — `FakeLLM` prompt-hash fragility was painful; we built a recording version that auto-regenerates fixtures. **Three** — Gemini's free tier turns out to be only twenty requests per day per model, not five hundred, so we needed Tier 1 billing for the live run, costing three cents. **Four** — `gemini-3-flash-preview` burns about fourteen thousand hidden "thinking" tokens per call, so we switched to `gemini-2.5-flash-lite` which has no thinking mode. **Five** — the twenty-percent per-paper analysis failure rate was a real-world tradeoff between schema strictness and robustness; the per-paper isolation made this a non-event for the stage.

### Slide 18 — Future Work (1:00)

> A clear v2 roadmap. We've already deferred and tracked five items: HyDE query transformation to address query-document asymmetry, citation-graph snowballing to widen the corpus from a seed paper, Unpaywall as a third PDF source beyond Semantic Scholar and arXiv, an embedding swap to `multi-qa-MiniLM-L6-cos-v1`, and multilingual paper support. Pragmatic next moves: register for a Semantic Scholar API key, which is one RPS sustained with no IP-block sensitivity; loosen the citation regex to recover the twenty-percent of failures; and pre-bake the report PDF into `demo-cache.tar.gz` on a machine with `latexmk`.

### Slide 19 — Demo Plan (1:00)

> We'll show three things. **One — the offline demo:** `ara run --cached-only` produces a complete report in four seconds with no network. **Two — a live run** where you can give us a research question and we'll process ten arXiv papers in roughly two to three minutes, with the stage logs showing query expansion, retrieval, extraction firing the injection defense, indexing, hierarchical RAG, and report generation. **Three — the prior live-run artifacts** — the auto-generated literature review, the cross-paper methodology comparison, and the research-gap analysis with citations. If anything fails — network outage, API rate-limit — we have the cached path as our safety net.

### Slide 20 — Thank You and Q&A (0:30)

> Thank you. The repository is private on GitHub at `souparna21/<repo>`. The full IE624 report is in the repo root with all measured numbers. The live-run artifacts are preserved under `runs/`. **We have two hundred sixty-three passing tests and a four-second offline demo. We're happy to take your questions.**

---

## Rehearsal checklist

- Time each speaker individually with a stopwatch — aim for 30 s under allocation.
- Practice the **transition handoffs** at slides 7→8 and 13→14 explicitly.
- Have the live demo open in a second window before slide 19.
- Pre-load the prior live-run output (`runs/20260427-173618-...`) as a backup.
- Ensure laptop has internet AND the offline `--cached-only` path works.
- Bring a backup laptop with the project pre-cloned and tested.
