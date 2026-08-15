#!/usr/bin/env bash
# Stop hook: gate the turn on `ty check .` + `pytest`, but only when Python
# actually changed this turn.
#
# Replaces two earlier hooks:
#   - ty-check.sh ran on PostToolUse, so a whole-project check fired after every
#     single edit (N-1 redundant runs per multi-file turn) and its exit 2 tripped
#     on the transient cross-file breakage that is normal mid-refactor.
#   - pytest-on-stop.sh ran unconditionally, so every conversational turn paid
#     ~8s to re-confirm a suite nothing had touched.
# Both checks are whole-project and belong at the turn boundary, run once.
#
# exit 2 keeps Claude working to fix what failed; stop_hook_active guards the loop.
set -euo pipefail

input=$(cat)
active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false')
[ "$active" = "true" ] && exit 0

[ -n "${CLAUDE_PROJECT_DIR:-}" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" || exit 0

# Set by ruff-fix.sh on any *.py edit. No marker means no Python changed, so
# both gates below would only restate the previous turn's result.
marker=".claude/.dirty"
[ -f "$marker" ] || exit 0

if ! out=$(uv run ty check . 2>&1); then
  printf 'ty reported type errors — please fix before finishing:\n\n%s\n' "$out" >&2
  exit 2
fi

if ! out=$(uv run pytest -q 2>&1); then
  printf 'pytest is failing — please fix before finishing:\n\n%s\n' "$out" >&2
  exit 2
fi

# Cleared only after both gates pass, so a failing turn is re-checked on the
# next Stop even if the follow-up edits never touch Python again.
rm -f "$marker"
exit 0
