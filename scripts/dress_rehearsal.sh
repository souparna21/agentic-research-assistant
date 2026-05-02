#!/usr/bin/env bash
# scripts/dress_rehearsal.sh — demo-day validation of ara --cached-only.
#
# Copies the current repo to a scratch /tmp/ara-rehearsal-<timestamp>/,
# runs `uv sync`, `cp .env.example .env`, `ara --cached-only run "..."`,
# verifies runs/<demo_id>/report.pdf (or report.tex if no latexmk) exists
# and is non-empty, cleans up, prints PASS/FAIL + elapsed time.
#
# Exits 0 on PASS, non-zero on FAIL. Target: < 60 seconds on a clean machine
# with uv pre-installed.
#
# Usage:
#     bash scripts/dress_rehearsal.sh

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TIMESTAMP="$(date +%Y%m%d-%H%M%S)"
SCRATCH="/tmp/ara-rehearsal-${TIMESTAMP}"
DEMO_QUESTION="What is retrieval-augmented generation?"

cleanup() {
  if [[ -n "${SCRATCH:-}" && -d "${SCRATCH}" ]]; then
    rm -rf "${SCRATCH}"
  fi
}
trap cleanup EXIT

t0=$(date +%s)

echo "[dress-rehearsal] Copying repo to ${SCRATCH}"
mkdir -p "${SCRATCH}"
# Use rsync to exclude .venv/, runs/, caches (keeps copy fast + clean).
# Fall back to cp if rsync is absent (unusual on modern Linux/macOS).
if command -v rsync >/dev/null 2>&1; then
  rsync -a \
    --exclude='.venv/' \
    --exclude='runs/' \
    --exclude='__pycache__/' \
    --exclude='.pytest_cache/' \
    --exclude='.mypy_cache/' \
    --exclude='.ruff_cache/' \
    --exclude='.tmp/' \
    "${REPO_ROOT}/" "${SCRATCH}/"
else
  cp -r "${REPO_ROOT}/." "${SCRATCH}/"
fi

cd "${SCRATCH}"

echo "[dress-rehearsal] uv sync"
uv sync --quiet

echo "[dress-rehearsal] bootstrap .env (cached-only — no real key needed)"
cp .env.example .env

echo "[dress-rehearsal] ara --cached-only run \"${DEMO_QUESTION}\""
uv run ara --cached-only run "${DEMO_QUESTION}"

# Verify output. The demo tarball extracts as runs/<DEMO_RUN_ID>/.
# DEMO_RUN_ID is a stable name ("demo-v1") committed inside the tarball;
# we enumerate runs/ rather than hard-coding it for defensive robustness.
shopt -s nullglob
RUN_DIRS=("${SCRATCH}/runs"/*/)
shopt -u nullglob
if [[ ${#RUN_DIRS[@]} -ne 1 ]]; then
  echo "[dress-rehearsal] FAIL: expected 1 run dir under runs/, got ${#RUN_DIRS[@]}"
  exit 1
fi
RUN_DIR="${RUN_DIRS[0]}"

# Primary artifact: report.pdf (present if tarball was built on a latexmk-equipped machine).
# Fallback artifact: report.tex (structural validity — always shipped).
PDF="${RUN_DIR}report.pdf"
TEX="${RUN_DIR}report.tex"

if [[ -s "${PDF}" ]]; then
  MAGIC="$(head -c 5 "${PDF}")"
  if [[ "${MAGIC}" != "%PDF-" ]]; then
    echo "[dress-rehearsal] FAIL: ${PDF} has bad magic bytes: ${MAGIC}"
    exit 1
  fi
  ARTIFACT_MSG="report.pdf ($(wc -c < "${PDF}") bytes)"
elif [[ -s "${TEX}" ]]; then
  if ! grep -q 'begin{document}' "${TEX}"; then
    echo "[dress-rehearsal] FAIL: ${TEX} missing \\begin{document}"
    exit 1
  fi
  ARTIFACT_MSG="report.tex ($(wc -c < "${TEX}") bytes) — no latexmk in tarball build"
else
  echo "[dress-rehearsal] FAIL: neither ${PDF} nor ${TEX} present/non-empty"
  exit 1
fi

# Also assert state.json is present — stable contract from extract_demo_cache.
STATE="${RUN_DIR}state.json"
if [[ ! -s "${STATE}" ]]; then
  echo "[dress-rehearsal] FAIL: ${STATE} not present/non-empty"
  exit 1
fi

t1=$(date +%s)
elapsed=$((t1 - t0))
echo "[dress-rehearsal] PASS (${elapsed}s) — ${ARTIFACT_MSG}"
