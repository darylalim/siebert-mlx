#!/usr/bin/env bash
# PreToolUse (Edit|Write|MultiEdit): block edits to secret-bearing .env files.
# .env holds HF_TOKEN and is gitignored; templates (.env.example etc.) stay editable.
set -euo pipefail

input=$(cat)
file=$(printf '%s' "$input" | jq -r '.tool_input.file_path // empty')
[ -n "$file" ] || exit 0

base=$(basename "$file")
case "$base" in
  .env.example | .env.sample | .env.template)
    exit 0
    ;;
  .env | .env.*)
    reason="Refusing to edit $base via a tool — it holds secrets (e.g. HF_TOKEN) and is gitignored. Edit it by hand if you truly need to."
    jq -n --arg r "$reason" '{
      hookSpecificOutput: {
        hookEventName: "PreToolUse",
        permissionDecision: "deny",
        permissionDecisionReason: $r
      }
    }'
    exit 0
    ;;
esac
exit 0
