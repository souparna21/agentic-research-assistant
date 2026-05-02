"""Repository hygiene tests (FND-10).

The gitleaks integration test is slow on first run (pre-commit downloads the
gitleaks binary) but subsequent runs are fast. It's marked ``@pytest.mark.integration``
to allow CI to exclude it when iterating.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_env_gitignored() -> None:
    gi = (REPO_ROOT / ".gitignore").read_text()
    lines = {line.strip() for line in gi.splitlines()}
    assert ".env" in lines, f".env must be in .gitignore (got lines: {sorted(lines)})"


def test_env_example_exists() -> None:
    assert (REPO_ROOT / ".env.example").exists(), ".env.example must be committed"


def test_env_example_has_required_vars() -> None:
    """Every ARA_* Settings field must appear in .env.example (commented or not)."""
    env = (REPO_ROOT / ".env.example").read_text()
    required = [
        "ARA_LLM_MODEL",
        "ARA_GEMINI_API_KEY",
        "ARA_RUNS_DIR",
        "ARA_PROMPTS_DIR",
        "ARA_LOG_LEVEL",
    ]
    for var in required:
        assert re.search(rf"(^|\n)#?\s*{re.escape(var)}=", env), (
            f"{var} missing from .env.example"
        )


def test_precommit_config_parses() -> None:
    cfg = yaml.safe_load((REPO_ROOT / ".pre-commit-config.yaml").read_text())
    assert "repos" in cfg and isinstance(cfg["repos"], list)
    repo_urls = {r.get("repo") for r in cfg["repos"]}
    assert any("gitleaks" in (u or "") for u in repo_urls), "gitleaks repo missing"
    assert any("detect-secrets" in (u or "") for u in repo_urls), (
        "detect-secrets repo missing"
    )
    assert any("ruff" in (u or "") for u in repo_urls), "ruff repo missing"


def test_precommit_validate_config() -> None:
    """pre-commit validate-config catches malformed YAML / missing rev."""
    result = subprocess.run(
        ["uv", "run", "pre-commit", "validate-config"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        timeout=60,
    )
    assert result.returncode == 0, (
        f"pre-commit validate-config failed:\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )


def test_secrets_baseline_is_valid_json() -> None:
    data = json.loads((REPO_ROOT / ".secrets.baseline").read_text())
    assert "version" in data
    assert "plugins_used" in data


SYNTHETIC_AIZA_LEAK = "AIzaSyB1x7PqZkL3mNtR8vWxY5jDhF4eK2gC9zQ"


@pytest.mark.integration
def test_gitleaks_blocks_fake_aiza(tmp_path: Path) -> None:
    """A file containing a plausible ``AIza...`` GCP key MUST be blocked by gitleaks.

    This is the FND-10 acceptance criterion. The gitleaks pre-commit hook
    uses ``gitleaks git --pre-commit --staged`` (``pass_filenames: false``),
    so the leak must be actually staged in a throw-away git repo before
    ``pre-commit run gitleaks`` can see it. We therefore initialize a fresh
    repo in ``tmp_path``, copy the pre-commit config into it, stage a file
    containing a synthetic ``AIzaSy...`` string with entropy >= 3 and the
    exact 35-character tail the ``gcp-api-key`` rule requires, and assert
    that ``pre-commit run gitleaks`` exits non-zero.

    Skipped when pre-commit / git / uv are not installed (e.g. in a stripped
    CI container). First-run is slow because pre-commit fetches the gitleaks
    binary; subsequent runs reuse the cached binary.
    """
    if shutil.which("git") is None:
        pytest.skip("git CLI not available on PATH")
    pre_commit_bin = shutil.which("pre-commit") or str(
        REPO_ROOT / ".venv" / "bin" / "pre-commit"
    )
    if not Path(pre_commit_bin).exists():
        pytest.skip("pre-commit CLI not available on PATH or in .venv")

    sandbox = tmp_path / "sandbox"
    sandbox.mkdir()
    subprocess.run(
        ["git", "init", "--quiet", "--initial-branch=main"],
        cwd=sandbox,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@example.invalid"],
        cwd=sandbox,
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=sandbox,
        check=True,
        capture_output=True,
    )

    (sandbox / ".pre-commit-config.yaml").write_text(
        (REPO_ROOT / ".pre-commit-config.yaml").read_text()
    )

    leak = sandbox / "leak.env"
    leak.write_text(f"GOOGLE_API_KEY={SYNTHETIC_AIZA_LEAK}\n")
    subprocess.run(
        ["git", "add", "leak.env"],
        cwd=sandbox,
        check=True,
        capture_output=True,
    )

    result = subprocess.run(
        [pre_commit_bin, "run", "gitleaks", "--all-files"],
        capture_output=True,
        text=True,
        cwd=sandbox,
        timeout=180,
    )
    assert result.returncode != 0, (
        f"gitleaks did NOT block the fake AIza string!\n"
        f"stdout: {result.stdout}\nstderr: {result.stderr}"
    )
