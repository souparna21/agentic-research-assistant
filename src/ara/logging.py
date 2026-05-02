"""structlog configuration with contextvars + stdlib bridge + secret redaction.

`configure_logging()` is called exactly once per process by the orchestrator.
After that:
    * `structlog.get_logger().info(...)` emits JSON to `runs/<run_id>/run.log`
      and a human-readable line to stderr.
    * `logging.getLogger(...)` (stdlib) records flow through the SAME processor
      chain — so LiteLLM / httpx / any stdlib-using library inherits `run_id` +
      `git_sha` automatically (pitfall P13 reproducibility evidence).
    * Any log field whose key contains `api_key|secret|token|password|bearer`
      (case-insensitive) is replaced with `<REDACTED>` before rendering
      (pitfall P14 defense — last line of defense alongside pre-commit hooks
      from plan 07).

Canonical source: 01-RESEARCH.md §"Pattern 3: structlog + stdlib Dual-Sink
with contextvars". Processor chain order matters — `merge_contextvars` MUST
be first so downstream processors see the bound `run_id` + `git_sha`.

Idempotent: calling twice replaces root-logger handlers and re-binds the
contextvars — useful for orchestrator rebuilds in tests.
"""

from __future__ import annotations

import logging
import re
import sys
from collections.abc import MutableMapping
from pathlib import Path
from typing import Any

import structlog
from structlog.contextvars import bind_contextvars, merge_contextvars
from structlog.typing import Processor

_SECRET_KEY_RE = re.compile(r"(api_key|secret|token|password|bearer)", re.IGNORECASE)


def _redact_secrets(
    logger: Any, method_name: str, event_dict: MutableMapping[str, Any]
) -> MutableMapping[str, Any]:
    """structlog processor: replace any field whose KEY matches `_SECRET_KEY_RE`."""
    for key in list(event_dict):
        if isinstance(key, str) and _SECRET_KEY_RE.search(key):
            event_dict[key] = "<REDACTED>"
    return event_dict


def configure_logging(
    log_path: Path,
    log_level: str = "INFO",
    *,
    run_id: str,
    git_sha: str,
) -> None:
    """Wire structlog + stdlib to share a processor chain.

    Args:
        log_path: Destination file for JSON-line records. Parent directory is
            created if missing.
        log_level: Standard logging level name (DEBUG / INFO / WARNING / ERROR).
        run_id: Bound into every log record via contextvars.
        git_sha: Bound into every log record via contextvars.

    Idempotent — replaces root-logger handlers and re-binds context on every
    call. Call once at orchestrator startup per process.
    """
    log_path = Path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    shared_processors: list[Processor] = [
        merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        _redact_secrets,
    ]

    structlog.configure(
        processors=shared_processors
        + [structlog.stdlib.ProcessorFormatter.wrap_for_formatter],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    json_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )
    console_formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=shared_processors,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(json_formatter)
    stderr_handler = logging.StreamHandler(sys.stderr)
    stderr_handler.setFormatter(console_formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stderr_handler)
    root.setLevel(getattr(logging, log_level.upper(), logging.INFO))

    bind_contextvars(run_id=run_id, git_sha=git_sha)
