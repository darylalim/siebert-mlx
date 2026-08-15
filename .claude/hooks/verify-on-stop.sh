#!/usr/bin/env bash
# Stop hook: gate the turn on ruff + ty + pytest, but only when something the
# suite actually depends on has changed since the last time all four passed.
#
# The trigger is a content fingerprint of the verifiable tree, computed here.
# The first version had ruff-fix.sh touch a marker file on *.py edits, which
# made the signal blind to everything a PostToolUse formatter never sees -- a
# .py rewritten by Bash/sed, `git checkout`/`revert`/`stash pop`, and
# dependency edits all skipped the gate -- while a throwaway .py in a scratch
# directory *outside* the repo triggered it. Deriving the signal from the
# repo's own contents fixes both directions and drops the cross-hook coupling.
#
# A fingerprint, not `git status`: status compares against HEAD, so an
# uncommitted work-in-progress tree reads dirty all session and every
# conversational turn pays the full gate. This compares against the last state
# that passed, so a turn that changes nothing is free regardless of the diff.
#
# exit 2 keeps Claude working to fix what failed; stop_hook_active guards the loop.
set -euo pipefail

input=$(cat)
active=$(printf '%s' "$input" | jq -r '.stop_hook_active // false')
[ "$active" = "true" ] && exit 0

[ -n "${CLAUDE_PROJECT_DIR:-}" ] || exit 0
cd "$CLAUDE_PROJECT_DIR" 2>/dev/null || exit 0

# Tracked *and* untracked-but-not-ignored, so a brand-new test file counts.
# samples/ is in scope because the suite reads mixed_sample.csv and
# blank_cells.csv; pyproject.toml and uv.lock because a dependency bump is
# exactly how this repo last shipped a silent behavior change (dba72ff moved
# the encoder from fp16 to fp32 activations with CI green).
fingerprint() {
  local files
  files=$(git ls-files --cached --others --exclude-standard \
    -- '*.py' pyproject.toml uv.lock samples 2>/dev/null) || return 0
  [ -n "$files" ] || return 0
  printf '%s\n' "$files" | tr '\n' '\0' | xargs -0 shasum 2>/dev/null |
    shasum | cut -d' ' -f1
}

stamp=".claude/.verified"
current=$(fingerprint)
# Empty means no git tree or nothing verifiable -- nothing to gate on.
[ -n "$current" ] || exit 0
[ "$current" = "$(cat "$stamp" 2>/dev/null)" ] && exit 0

# Every gate CI's `check` job runs, in CI's order: the cheap lint/format checks
# fail in milliseconds and cost nothing to put first. ruff-fix.sh cannot stand
# in for `ruff check .` -- it applies only *safe* fixes and swallows the exit
# code, so unfixable B/SIM/RUF violations survive it silently.
gate() { # gate <label> <cmd>...
  local out
  if ! out=$("${@:2}" 2>&1); then
    printf '%s — please fix before finishing (paths are in the output):\n\n%s\n' \
      "$1" "$out" >&2
    exit 2
  fi
}

gate 'ruff found lint errors' uv run ruff check .
gate 'ruff found formatting drift' uv run ruff format --check .
gate 'ty reported type errors' uv run ty check .
gate 'pytest is failing' uv run pytest -q

# Recorded only after every gate passes, so a red turn is re-checked on the
# next Stop even if the follow-up edits never touch a verifiable file again.
# Failure to write is non-fatal: the cost is one redundant gate next turn,
# which is strictly better than a hook that errors after all checks passed.
printf '%s\n' "$current" >"$stamp" 2>/dev/null || true
exit 0
