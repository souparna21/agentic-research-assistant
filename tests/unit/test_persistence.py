"""Atomic write tests — defense against Ctrl-C corruption (pitfall P7)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest


def test_atomic_write_basic(tmp_path: Path):
    from ara.persistence import atomic_write_json, load_state
    from ara.state import PipelineState

    state = PipelineState(
        run_id="20260417-010101-basic", question="q", config_snapshot={"x": 1}
    )
    target = tmp_path / "runs" / state.run_id / "state.json"
    atomic_write_json(state, target)
    assert target.exists()
    assert load_state(target) == state


def test_atomic_write_creates_parent_dirs(tmp_path: Path):
    from ara.persistence import atomic_write_json
    from ara.state import PipelineState

    state = PipelineState(run_id="x", question="q", config_snapshot={})
    deep = tmp_path / "a" / "b" / "c" / "state.json"
    atomic_write_json(state, deep)
    assert deep.exists()


def test_atomic_write_uses_same_directory_tempfile(tmp_path: Path):
    """Pitfall P7: tempfile must land in same dir so os.replace is atomic."""
    from ara import persistence
    from ara.state import PipelineState

    state = PipelineState(run_id="x", question="q", config_snapshot={})
    target = tmp_path / "state.json"

    real_ntf = persistence.tempfile.NamedTemporaryFile
    captured_kwargs = {}

    def wrapper(*args, **kwargs):
        captured_kwargs.update(kwargs)
        return real_ntf(*args, **kwargs)

    with patch.object(persistence.tempfile, "NamedTemporaryFile", wrapper):
        persistence.atomic_write_json(state, target)

    assert Path(captured_kwargs["dir"]) == target.parent


def test_existing_target_survives_interrupted_write(tmp_path: Path):
    """If os.replace is never called (crash mid-write), the old file is intact."""
    from ara import persistence
    from ara.state import PipelineState

    s1 = PipelineState(run_id="v1", question="first", config_snapshot={})
    s2 = PipelineState(run_id="v2", question="second", config_snapshot={})
    target = tmp_path / "state.json"
    persistence.atomic_write_json(s1, target)
    original = target.read_text()

    with patch.object(persistence.os, "replace", side_effect=OSError("simulated crash")):
        with pytest.raises(OSError, match="simulated crash"):
            persistence.atomic_write_json(s2, target)

    assert target.read_text() == original




def test_atomic_write_text_creates_file(tmp_path: Path):
    """02-00: atomic_write_text writes content and creates parent dirs."""
    from ara.persistence import atomic_write_text

    p = tmp_path / "sub" / "out.md"
    atomic_write_text("# Hello\nWorld", p)
    assert p.read_text() == "# Hello\nWorld"


def test_atomic_write_text_overwrites_atomically(tmp_path: Path):
    """02-00: atomic_write_text overwrites with no leftover tempfiles."""
    from ara.persistence import atomic_write_text

    p = tmp_path / "out.md"
    atomic_write_text("old", p)
    atomic_write_text("new", p)
    assert p.read_text() == "new"
    leftovers = [f for f in tmp_path.iterdir() if f.name.startswith(".")]
    assert leftovers == []


def test_atomic_write_text_crash_safe_on_replace_failure(tmp_path: Path, monkeypatch):
    """02-00: when os.replace fails, tempfile is cleaned up and exception propagates."""
    import os as _os

    from ara.persistence import atomic_write_text

    p = tmp_path / "out.md"

    def boom(src, dst):
        raise OSError("simulated rename failure")

    monkeypatch.setattr(_os, "replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_write_text("content", p)
    assert not p.exists()
    leftovers = [f for f in tmp_path.iterdir() if f.name.startswith(".")]
    assert leftovers == [], f"Tempfile not cleaned up: {leftovers}"
