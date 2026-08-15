#!/usr/bin/env bash
# PostToolUse (Edit|Write|MultiEdit): format the edited file with ruff.
# Silent on success; leaves every edit ruff-clean per CLAUDE.md.
#
# Markdown is in scope because ruff >= 0.16 formats Python fences inside .md,
# and CI's `ruff format --check .` covers CLAUDE.md and README.md alongside the
# 5 .py files (7 total). `ruff check` is Python-only -- it reports "No Python
# files found" on .md -- so only the .py arm pays for that second subprocess.
#
# This hook is a formatter, not a gate: it reports nothing back and swallows
# every failure. Whole-project verification (ruff check ., ty, pytest) lives in
# verify-on-stop.sh, which derives its own trigger from the repo contents
# rather than from anything this hook records.
set -euo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[ -n "$file" ] || exit 0
[ -f "$file" ] || exit 0

[ -n "${CLAUDE_PROJECT_DIR:-}" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

# In-project files only. An edit to a scratch script elsewhere on the
# filesystem is not this repo's to reformat under this repo's ruff config.
case "$file" in
  "$CLAUDE_PROJECT_DIR"/*) ;;
  *) exit 0 ;;
esac

case "$file" in
  *.py)
    uv run ruff format "$file" >/dev/null 2>&1 || true
    uv run ruff check --fix "$file" >/dev/null 2>&1 || true
    ;;
  *.md)
    uv run ruff format "$file" >/dev/null 2>&1 || true
    ;;
esac
exit 0
