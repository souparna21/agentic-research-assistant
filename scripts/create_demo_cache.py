"""One-shot regenerator for demo-cache.tar.gz (PKG-02).

Runs the full 6-stage smoke pipeline using ``build_stub_stages()`` (FakeLLM +
fixture corpus + real agents), then tars the produced ``runs/<run_id>/`` dir
to ``<repo_root>/demo-cache.tar.gz``.

Run once from the project root when fixtures or schema change:

    uv run python scripts/create_demo_cache.py

Idempotent — safe to re-run. Commit demo-cache.tar.gz after.

Design notes:
    - Uses the same offline setup as ``tests/e2e/test_smoke.py``: HF_HUB_OFFLINE=1
      + TRANSFORMERS_OFFLINE=1, and ``build_stub_stages()`` reads the committed
      ``tests/fixtures/fake_llm/smoke.json`` (16 entries closed in Phase 3).
    - ``DEMO_RUN_ID = "demo-v1"`` is stable across regenerations (04-RESEARCH
      Pitfall 2). ``extract_demo_cache()`` discovers the run_id from the
      tarball's top-level directory, so the coupling is one-way — this script
      is the only place the constant is pinned.
    - Uses ``tar.add(run_dir, arcname=DEMO_RUN_ID)`` so the tarball has exactly
      one top-level directory named ``demo-v1`` (contract enforced by
      ``extract_demo_cache``).
    - Size budget: ≤ 10 MB (04-RESEARCH Pitfall 5). The script warns but does
      not fail if the tarball overshoots; plan-level fix is to truncate
      ``run.log`` or exclude ``index/`` if needed.
"""

from __future__ import annotations

import os
import sys
import tarfile
import tempfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Force HF offline so MiniLMEmbedder uses the cached snapshot (no HTTP).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

DEMO_RUN_ID = "demo-v1"
"""Stable run_id across regenerations — contract with ``extract_demo_cache``."""

DEMO_QUESTION = "What is retrieval-augmented generation?"
"""Matches the seed question used by the Phase-3 smoke FakeLLM fixture."""

OUTPUT_TARBALL = REPO_ROOT / "demo-cache.tar.gz"
"""Repo-root destination for the committed demo tarball."""

SIZE_BUDGET_MB = 10.0
"""Soft size limit — warns but does not fail (manual intervention if exceeded)."""


def main() -> None:
    from ara.agents.stubs import build_stub_stages
    from ara.config import Settings
    from ara.orchestrator import Orchestrator

    with tempfile.TemporaryDirectory(prefix="demo-cache-") as tmp:
        tmp_path = Path(tmp)
        settings = Settings(runs_dir=str(tmp_path / "runs"))

        stages = build_stub_stages()
        orch = Orchestrator(
            settings,
            DEMO_RUN_ID,
            git_sha="DEMO",
            question=DEMO_QUESTION,
            stages=stages,
        )
        orch.run()

        run_dir = tmp_path / "runs" / DEMO_RUN_ID
        if not run_dir.exists():
            raise RuntimeError(f"Demo run did not produce {run_dir}")

        # Sanity — require report.tex at minimum. report.pdf is only present
        # when latexmk is on PATH (ReportAgent gracefully skips compile otherwise).
        if not (run_dir / "report.tex").exists():
            raise RuntimeError(
                f"Demo run did not produce report.tex at {run_dir / 'report.tex'}"
            )

        # Count files for summary output
        file_count = sum(1 for _ in run_dir.rglob("*") if _.is_file())

        # Tar with arcname=DEMO_RUN_ID so the tarball has exactly one top-level
        # directory named ``demo-v1`` (enforced by ``extract_demo_cache``).
        if OUTPUT_TARBALL.exists():
            OUTPUT_TARBALL.unlink()
        with tarfile.open(OUTPUT_TARBALL, mode="w:gz") as tar:
            tar.add(run_dir, arcname=DEMO_RUN_ID)

        size_bytes = OUTPUT_TARBALL.stat().st_size
        size_mb = size_bytes / (1024 * 1024)
        print(f"Wrote {OUTPUT_TARBALL} ({size_mb:.2f} MB, {file_count} files, run_id={DEMO_RUN_ID})")

        if size_mb > SIZE_BUDGET_MB:
            print(
                f"WARN: tarball {size_mb:.2f} MB exceeds {SIZE_BUDGET_MB} MB budget — "
                f"consider truncating run.log or excluding FAISS index."
            )


if __name__ == "__main__":
    main()
