"""Atomic JSON persistence for PipelineState.

Writes go through a same-directory tempfile + fsync + os.replace so Ctrl-C
never corrupts state.json. Pitfall P7: cross-filesystem temp dirs make
os.replace degrade to copy+delete (non-atomic) — hence dir=path.parent.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from ara.state import PipelineState


def atomic_write_json(state: PipelineState, path: Path) -> None:
    """Write state to path atomically. Target either has old content or new, never partial."""
    path.parent.mkdir(parents=True, exist_ok=True)
    data = state.model_dump_json(indent=2)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tf:
        tf.write(data)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    try:
        os.replace(tmp_name, str(path))
    except OSError:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise


def load_state(path: Path) -> PipelineState:
    """Load PipelineState from JSON on disk."""
    return PipelineState.model_validate_json(path.read_text(encoding="utf-8"))


def atomic_write_text(text: str, path: Path) -> None:
    """Atomic text write — same pattern as :func:`atomic_write_json`.

    Mid-write Ctrl-C leaves target byte-identical to pre-write content
    (pitfall P7 defense). Used by ExtractionAgent (plan 02-06) for
    ``runs/<run_id>/papers/<paper_id>/extracted.md`` writes.

    Args:
        text: UTF-8 text payload (markdown, JSON-as-text, etc.).
        path: Target path. Parent directories are created if missing.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=str(path.parent),
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as tf:
        tf.write(text)
        tf.flush()
        os.fsync(tf.fileno())
        tmp_name = tf.name
    try:
        os.replace(tmp_name, str(path))
    except OSError:
        try:
            os.unlink(tmp_name)
        except FileNotFoundError:
            pass
        raise
