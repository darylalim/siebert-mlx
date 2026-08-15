#!/usr/bin/env bash
# PostToolUse (Edit|Write|MultiEdit): format + lint-fix edited files with ruff.
# Silent on success; leaves every edit ruff-clean per CLAUDE.md.
#
# Markdown is in scope because ruff >= 0.16 formats Python fences inside .md, so
# CI's `ruff format --check .` covers CLAUDE.md and README.md alongside the 5 .py
# files (7 total). Without *.md here, a docs-only edit that adds a mis-formatted
# ```python fence is clean locally and fails CI.
set -euo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

[ -n "${CLAUDE_PROJECT_DIR:-}" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0

case "$file" in
  # Python changed: mark the turn dirty so verify-on-stop.sh knows to run the
  # (whole-project, multi-second) ty + pytest gate. Markdown reaches neither
  # checker, so it gets formatted without setting the marker.
  *.py) touch .claude/.dirty ;;
  *.md) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

uv run ruff format "$file" >/dev/null 2>&1 || true
uv run ruff check --fix "$file" >/dev/null 2>&1 || true
exit 0
