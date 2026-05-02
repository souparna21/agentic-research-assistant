"""run_id generation and git_sha capture.

run_id format: YYYYMMDD-HHMMSS-<slug>  — lexicographically sortable in `ls`.
git_sha: `git rev-parse --short HEAD` with graceful fallback to "unknown".
"""

from __future__ import annotations

import subprocess
from datetime import datetime, timezone

from slugify import slugify


def make_run_id(question: str) -> str:
    """Build run_id = YYYYMMDD-HHMMSS-<slug> from the user's question.

    Empty or all-non-ASCII questions fall back to slug `query`.
    """
    ts = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    slug = slugify(question, max_length=40, word_boundary=True, separator="-") or "query"
    return f"{ts}-{slug}"


def read_git_sha() -> str:
    """Return short git SHA of HEAD, or 'unknown' if not in a git repo / git missing."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
        return out.stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError, subprocess.TimeoutExpired):
        return "unknown"
