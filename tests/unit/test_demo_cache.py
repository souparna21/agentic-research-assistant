"""Unit tests for ara.demo_cache (PKG-02 / plan 04-01).

Covers the three contract points of extract_demo_cache:

1. Happy path: extracts a well-formed tarball into target_runs_dir and
   returns the discovered run_id (tarball top-level dir name).
2. Missing tarball: raises CachedDemoMissingError with an actionable
   message pointing at scripts/create_demo_cache.py.
3. Idempotent reuse: a second call with an already-extracted run_id
   does NOT re-extract (any sentinel file written between calls
   survives).

Tests construct their own tarball inline (tmp_path + tarfile stdlib) so
they do NOT depend on the committed demo-cache.tar.gz (which lands in
plan 04-02).
"""

from __future__ import annotations

import tarfile
from pathlib import Path

import pytest

from ara.demo_cache import extract_demo_cache
from ara.errors import CachedDemoMissingError


def _build_test_tarball(tarball_path: Path, run_id: str = "demo-v1") -> None:
    """Write a minimal tarball whose only top-level entry is <run_id>/."""
    src_dir = tarball_path.parent / "_src" / run_id
    src_dir.mkdir(parents=True, exist_ok=True)
    (src_dir / "state.json").write_text('{"run_id": "demo-v1"}', encoding="utf-8")
    with tarfile.open(tarball_path, mode="w:gz") as tar:
        tar.add(src_dir, arcname=run_id)


def test_extract_demo_cache_happy_path(tmp_path: Path) -> None:
    """extract_demo_cache returns run_id + materializes tarball contents under target_runs_dir."""
    tarball = tmp_path / "demo-cache.tar.gz"
    _build_test_tarball(tarball)

    runs_dir = tmp_path / "runs"

    rid = extract_demo_cache(runs_dir, tarball_path=tarball)

    assert rid == "demo-v1"
    assert (runs_dir / "demo-v1" / "state.json").exists()
    assert (runs_dir / "demo-v1" / "state.json").read_text(encoding="utf-8") == (
        '{"run_id": "demo-v1"}'
    )


def test_missing_tarball_raises(tmp_path: Path) -> None:
    """Missing tarball raises CachedDemoMissingError with an actionable message."""
    missing = tmp_path / "nope.tar.gz"
    with pytest.raises(CachedDemoMissingError) as exc_info:
        extract_demo_cache(tmp_path / "runs", tarball_path=missing)
    assert "create_demo_cache.py" in str(exc_info.value)


def test_idempotent_reuses_existing(tmp_path: Path) -> None:
    """Second call with same run_id already extracted must skip extraction.

    We write a SENTINEL file between calls — if the second call re-extracts,
    the sentinel would be wiped out or mingled with tarball contents. The
    contract is "skip extraction entirely", so the sentinel must survive
    byte-identical.
    """
    tarball = tmp_path / "demo-cache.tar.gz"
    _build_test_tarball(tarball)
    runs_dir = tmp_path / "runs"

    rid1 = extract_demo_cache(runs_dir, tarball_path=tarball)

    sentinel = runs_dir / rid1 / "SENTINEL.txt"
    sentinel.write_text("survives-reuse", encoding="utf-8")

    rid2 = extract_demo_cache(runs_dir, tarball_path=tarball)

    assert rid2 == rid1
    assert sentinel.exists(), "sentinel must survive idempotent reuse"
    assert sentinel.read_text(encoding="utf-8") == "survives-reuse"
