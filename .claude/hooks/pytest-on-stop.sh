#!/usr/bin/env bash
# Stop hook: run the pytest suite when Claude finishes a turn; surface failures.
# exit 2 keeps Claude working to fix failing tests; stop_hook_active guards against loops.
set -euo pipefail

input=$(cat)
active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false')
[ "$active" = "true" ] && exit 0

cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

if ! out=$(uv run pytest -q 2>&1); then
  printf 'pytest is failing — please fix before finishing:\n\n%s\n' "$out" >&2
  exit 2
fi
exit 0
