"""structlog + contextvars + redaction tests (FND-09 + pitfall P14).

Covers the logging subsystem contract:
    * contextvars binding of run_id + git_sha applies to EVERY log line
    * stdlib logs (LiteLLM / httpx / urllib3) flow through the same processor
      chain — so their records also carry run_id + git_sha
    * secret-key redaction (api_key|secret|token|password|bearer, case-insensitive)
    * dual sink: JSON to run.log + human-readable to stderr
    * parent-directory creation on configure_logging
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import pytest


@pytest.fixture(autouse=True)
def _reset_logging():
    """structlog + stdlib logging are process-global; reset between tests."""
    import structlog
    from structlog.contextvars import clear_contextvars

    clear_contextvars()
    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(logging.WARNING)
    yield
    clear_contextvars()
    structlog.reset_defaults()
    root.handlers.clear()


def _read_log_lines(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def test_contextvars_bound(tmp_path: Path):
    import structlog

    from ara.logging import configure_logging

    log_path = tmp_path / "run.log"
    configure_logging(log_path, log_level="INFO", run_id="RID-123", git_sha="cafebabe")
    structlog.get_logger().info("hello_event", custom="v")

    records = _read_log_lines(log_path)
    assert len(records) == 1
    rec = records[0]
    assert rec["run_id"] == "RID-123"
    assert rec["git_sha"] == "cafebabe"
    assert rec["event"] == "hello_event"
    assert rec["custom"] == "v"


def test_stdlib_logs_include_run_id(tmp_path: Path):
    from ara.logging import configure_logging

    log_path = tmp_path / "run.log"
    configure_logging(log_path, log_level="INFO", run_id="RID-STDLIB", git_sha="deadbeef")
    logging.getLogger("third_party_lib").info("stdlib-event")

    records = _read_log_lines(log_path)
    assert any(
        r.get("run_id") == "RID-STDLIB" and r.get("git_sha") == "deadbeef"
        for r in records
    ), f"No stdlib record carried run_id. Got: {records}"


def test_api_key_redacted(tmp_path: Path):
    import structlog

    from ara.logging import configure_logging

    log_path = tmp_path / "run.log"
    configure_logging(log_path, log_level="INFO", run_id="R", git_sha="S")
    structlog.get_logger().info(
        "leak_test",
        api_key="AIzaLEAKED1234567890",
        gemini_api_key="AIzaALSOLEAKED",
        password="hunter2",
        access_token="tok-secret",
        bearer="BearerXYZ",
        safe_value="keep-me",
    )

    content = log_path.read_text()
    assert "AIzaLEAKED" not in content
    assert "AIzaALSOLEAKED" not in content
    assert "hunter2" not in content
    assert "tok-secret" not in content
    assert "BearerXYZ" not in content
    assert "keep-me" in content


def test_redacts_case_insensitive_keys(tmp_path: Path):
    import structlog

    from ara.logging import configure_logging

    log_path = tmp_path / "run.log"
    configure_logging(log_path, log_level="INFO", run_id="R", git_sha="S")
    structlog.get_logger().info(
        "case_test",
        API_KEY="upper",
        My_Secret="mixed",
        bearerToken="camel",
        PASSWORD="ALLCAPS",
    )
    content = log_path.read_text()
    for leak in ("upper", "mixed", "camel", "ALLCAPS"):
        assert leak not in content, f"{leak!r} leaked through redaction"


def test_dual_sink_writes_both(tmp_path, capsys):
    import structlog

    from ara.logging import configure_logging

    log_path = tmp_path / "run.log"
    configure_logging(log_path, log_level="INFO", run_id="R", git_sha="S")
    structlog.get_logger().info("dual_event", x=1)

    records = _read_log_lines(log_path)
    assert any(r["event"] == "dual_event" for r in records)

    captured = capsys.readouterr()
    assert "dual_event" in captured.err


def test_configure_logging_creates_parent_dir(tmp_path: Path):
    from ara.logging import configure_logging

    deep = tmp_path / "does" / "not" / "exist" / "run.log"
    configure_logging(deep, log_level="INFO", run_id="R", git_sha="S")
    assert deep.parent.is_dir()
