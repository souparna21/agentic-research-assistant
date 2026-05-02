"""Infrastructure services — stateless, reusable helpers agents depend on.

Each agent composes services (LLM client, prompt loader, fetchers, …); agents
never depend on each other. This split is what enables parallel development
and the single-writer-per-field invariant in the orchestrator.
"""
