#!/usr/bin/env bash
# PostToolUse (Edit|Write|MultiEdit): format + lint-fix edited Python files with ruff.
# Silent on success; leaves every edit ruff-clean per CLAUDE.md.
set -euo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac
[ -f "$file" ] || exit 0

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0
uv run ruff format "$file" >/dev/null 2>&1 || true
uv run ruff check --fix "$file" >/dev/null 2>&1 || true
exit 0
