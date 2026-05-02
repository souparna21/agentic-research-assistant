"""Safe, idempotent extraction of the committed demo tarball (PKG-02).

The `--cached-only` CLI mode relies on a committed `demo-cache.tar.gz` at
the repo root containing a pre-built `runs/<demo_run_id>/` directory with
every stage's artifacts (PDFs, extracted.md, FAISS index, state.json,
report.pdf). This module extracts that tarball into the live runs_dir so
the Orchestrator's delete-to-invalidate idempotency (FND-08) causes every
stage to emit `stage_skipped`, achieving zero outbound API calls on demo.

Design notes:
    - Safe extraction via `tarfile.extractall(filter="data")` — mandatory
      on Python 3.12+ to avoid DeprecationWarning and the dangerous
      `fully_trusted` fallback. Rejects absolute paths, outside-of-target
      symlinks, device files, and clears dangerous permission bits.
      Will be the default in Python 3.14 (pitfall P?/04-RESEARCH Pitfall 1).
    - Idempotent: if the discovered run_id directory already exists under
      the target runs_dir, skip extraction and return the existing run_id.
      This avoids clobbering in-flight work and makes repeated invocations
      cheap.
    - Run_id is DISCOVERED from the tarball's top-level directory name
      (not hardcoded). `scripts/create_demo_cache.py` pins this to
      `demo-v1` across regenerations (see 04-RESEARCH Pitfall 2).
    - Stdlib only: `tarfile`, `pathlib`. No heavy deps. Kept lazy-imported
      by `cli.py` to preserve the `ara --help` latency budget.
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import structlog

from ara.errors import CachedDemoMissingError

log = structlog.get_logger(__name__)

DEMO_CACHE_TARBALL: Path = Path("demo-cache.tar.gz")
"""Repo-root path of the committed demo tarball (PKG-02 artifact)."""


def extract_demo_cache(
    target_runs_dir: Path,
    tarball_path: Path = DEMO_CACHE_TARBALL,
) -> str:
    """Extract demo-cache.tar.gz into target_runs_dir; return the demo run_id.

    Idempotent: if the demo run_id directory already exists under
    target_runs_dir, skip extraction and return the existing run_id.

    Args:
        target_runs_dir: The runs directory to extract INTO (e.g. Path("runs")).
            The tarball's root entry is the ``<demo_run_id>/`` dir, so after
            extraction the layout is ``target_runs_dir/<demo_run_id>/state.json``.
            Created if missing.
        tarball_path: Path to the committed tarball. Defaults to the repo-root
            ``demo-cache.tar.gz``.

    Returns:
        The extracted demo run_id (the top-level directory name in the tarball).

    Raises:
        CachedDemoMissingError: if ``tarball_path`` does not exist, or if the
            tarball does not have exactly one top-level directory (malformed).
    """
    if not tarball_path.exists():
        raise CachedDemoMissingError(
            f"Demo cache tarball not found: {tarball_path}. "
            f"Run `uv run python scripts/create_demo_cache.py` to regenerate."
        )

    target_runs_dir.mkdir(parents=True, exist_ok=True)

    tarball_bytes = tarball_path.stat().st_size

    with tarfile.open(tarball_path, mode="r:gz") as tar:
        top_names = {m.name.split("/", 1)[0] for m in tar.getmembers() if m.name}
        if len(top_names) != 1:
            raise CachedDemoMissingError(
                f"Demo tarball has {len(top_names)} top-level dirs; expected 1. "
                f"Regenerate with `uv run python scripts/create_demo_cache.py`."
            )
        run_id = top_names.pop()

        target_run_dir = target_runs_dir / run_id
        if target_run_dir.exists():
            log.info(
                "demo_cache_reused",
                run_id=run_id,
                target_path=str(target_run_dir),
                tarball_bytes=tarball_bytes,
            )
            return run_id

        tar.extractall(path=target_runs_dir, filter="data")

    log.info(
        "demo_cache_extracted",
        run_id=run_id,
        target_path=str(target_run_dir),
        tarball_bytes=tarball_bytes,
    )
    return run_id
