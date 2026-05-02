"""run_id + git_sha generation tests."""

from __future__ import annotations

import re
import subprocess


def test_run_id_format():
    from ara.runids import make_run_id

    rid = make_run_id("What is RAG?")
    assert re.match(r"^\d{8}-\d{6}-[a-z0-9\-]+$", rid), f"bad format: {rid}"
    assert len(rid) <= 60


def test_run_id_unicode_is_slugified():
    from ara.runids import make_run_id

    rid = make_run_id("什麼是RAG?")
    slug_portion = rid.split("-", 2)[2]
    assert re.match(r"^[a-z0-9\-]+$", slug_portion), f"non-ASCII leaked: {slug_portion}"


def test_run_id_empty_question_uses_fallback():
    from ara.runids import make_run_id

    rid = make_run_id("")
    assert rid.split("-", 2)[2] == "query"


def test_git_sha_in_real_repo():
    from ara.runids import read_git_sha

    sha = read_git_sha()
    assert sha == "unknown" or re.match(r"^[0-9a-f]{7,40}$", sha), f"bad sha: {sha}"


def test_git_sha_falls_back_on_missing_git(monkeypatch):
    from ara import runids

    def fake_run(*args, **kwargs):
        raise FileNotFoundError("git not found")

    monkeypatch.setattr(runids.subprocess, "run", fake_run)
    assert runids.read_git_sha() == "unknown"


def test_git_sha_falls_back_on_called_process_error(monkeypatch):
    from ara import runids

    def fake_run(*args, **kwargs):
        raise subprocess.CalledProcessError(returncode=128, cmd=["git"])

    monkeypatch.setattr(runids.subprocess, "run", fake_run)
    assert runids.read_git_sha() == "unknown"
