"""Static-artifact tests for scripts/dress_rehearsal.sh — PKG-04 structural gate.

These tests validate that the dress rehearsal script exists, is executable,
and contains the mandatory shell-discipline contracts (shebang, ``set -euo
pipefail``, ``trap cleanup EXIT``, rsync excludes, ``--cached-only``
invocation). They do NOT run the script — runtime validation is the job of
``bash scripts/dress_rehearsal.sh`` itself (see plan 04-05 acceptance
criteria), which asserts end-to-end PASS in < 60s on a scratch copy of the
repo.

The unit layer here is purely a regression gate: if someone edits the script
and drops ``set -euo pipefail`` or forgets to re-chmod, CI catches it fast
(milliseconds) instead of waiting for a demo-day rehearsal run.
"""

from __future__ import annotations

import os
import stat
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "dress_rehearsal.sh"


def test_dress_rehearsal_script_exists() -> None:
    """The script file must live at scripts/dress_rehearsal.sh."""
    assert SCRIPT_PATH.exists(), f"missing: {SCRIPT_PATH}"
    assert SCRIPT_PATH.is_file(), f"not a regular file: {SCRIPT_PATH}"


def test_dress_rehearsal_script_is_executable() -> None:
    """chmod +x must have been applied (owner-executable bit)."""
    mode = SCRIPT_PATH.stat().st_mode
    assert mode & stat.S_IXUSR, (
        f"{SCRIPT_PATH} is not executable — run `chmod +x {SCRIPT_PATH}`"
    )
    assert os.access(SCRIPT_PATH, os.X_OK), (
        f"{SCRIPT_PATH} fails os.access(X_OK) despite st_mode IXUSR bit"
    )


def test_dress_rehearsal_script_has_bash_shebang() -> None:
    """First line must be the portable #!/usr/bin/env bash shebang."""
    first_line = SCRIPT_PATH.read_text(encoding="utf-8").splitlines()[0]
    assert first_line == "#!/usr/bin/env bash", (
        f"bad shebang: {first_line!r} — expected '#!/usr/bin/env bash'"
    )


def test_dress_rehearsal_script_uses_strict_mode() -> None:
    """set -euo pipefail is required for fail-fast semantics."""
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "set -euo pipefail" in text, (
        "missing `set -euo pipefail` — required for fail-fast bash hygiene"
    )


def test_dress_rehearsal_script_traps_cleanup() -> None:
    """trap cleanup EXIT must be installed to purge /tmp/ara-rehearsal-*."""
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "trap cleanup EXIT" in text, (
        "missing `trap cleanup EXIT` — scratch dir would orphan on non-zero exit"
    )


def test_dress_rehearsal_script_invokes_cached_only() -> None:
    """Script must invoke `ara --cached-only run ...` (offline demo)."""
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--cached-only" in text, (
        "script must pass --cached-only to ara (otherwise rehearsal "
        "would attempt live API calls)"
    )


def test_dress_rehearsal_script_excludes_venv_and_runs() -> None:
    """rsync must exclude .venv/ and runs/ to avoid copying heavy dirs."""
    text = SCRIPT_PATH.read_text(encoding="utf-8")
    assert "--exclude='.venv/'" in text, (
        "rsync must --exclude='.venv/' or the copy blows past 1 GB"
    )
    assert "--exclude='runs/'" in text, (
        "rsync must --exclude='runs/' or stale runs pollute the scratch copy"
    )
