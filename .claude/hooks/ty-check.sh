#!/usr/bin/env bash
# PostToolUse (Edit|Write|MultiEdit): type-check the project with ty after Python edits.
# On errors, exit 2 so ty's output is fed back to Claude as actionable feedback.
set -euo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')

case "$file" in
  *.py) ;;
  *) exit 0 ;;
esac

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

if ! out=$(uv run ty check . 2>&1); then
  printf 'ty reported type errors after editing %s:\n\n%s\n' "$file" "$out" >&2
  exit 2
fi
exit 0
