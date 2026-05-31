#!/usr/bin/env bash
# Smoke tests for scripts/phases/setup-project-claudemd.sh
#
# Covers the /scholar-init Step 1.2.5 auto-managed project memory file:
#   - fresh create (host=claude-code → CLAUDE.md only)
#   - idempotent no-op on re-run
#   - host=codex → AGENTS.md; host=unknown → both
#   - append-preserve when a memory file already exists without markers
#   - existing project never backfills the other memory file
#   - invalid --mode rejected (this fork is lean-only)
#   - CFPS LOCAL_MODE conditional block injection
#   - marker namespace is open-scholar-skill: (not scholar-full-paper:)

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
SCRIPT="${REPO_ROOT}/scripts/phases/setup-project-claudemd.sh"
TEMPLATE="${REPO_ROOT}/scripts/templates/claudemd-auto-rules-lean.md"

if [ ! -f "$SCRIPT" ]; then
  echo "FATAL: setup-project-claudemd.sh not found at $SCRIPT"
  exit 1
fi
if [ ! -f "$TEMPLATE" ]; then
  echo "FATAL: lean template not found at $TEMPLATE"
  exit 1
fi

TMPDIR_BASE="$(mktemp -d -t claudemd-smoke.XXXXXX)"
cleanup() { rm -rf "$TMPDIR_BASE" 2>/dev/null || true; }
trap cleanup EXIT

PASS=0
FAIL=0
pass() { PASS=$((PASS + 1)); echo "  PASS: $1"; }
fail() { FAIL=$((FAIL + 1)); echo "  FAIL: $1"; }

# Helper: make a fresh project dir with an (optionally) populated sidecar.
make_proj() {
  local dir="$1"; local sidecar="${2:-{\}}"
  mkdir -p "$dir/.claude"
  printf '%s\n' "$sidecar" > "$dir/.claude/safety-status.json"
}

# ─── Test 1: fresh create, host=claude-code ────────────────────────────
echo "Test 1: fresh create writes CLAUDE.md only (host=claude-code)"
P1="$TMPDIR_BASE/p1"
make_proj "$P1"
OUT=$(SCHOLAR_HOST_AGENT_OVERRIDE=claude-code bash "$SCRIPT" "$P1" --mode lean 2>&1)
if [ -f "$P1/CLAUDE.md" ] && [ ! -f "$P1/AGENTS.md" ]; then
  pass "created CLAUDE.md, no AGENTS.md"
else
  fail "expected CLAUDE.md only"; echo "$OUT" | sed 's/^/      /'
fi
if grep -q "open-scholar-skill:BEGIN auto-rules v2-lean" "$P1/CLAUDE.md" \
   && grep -q "open-scholar-skill:END auto-rules" "$P1/CLAUDE.md"; then
  pass "markers use the open-scholar-skill namespace"
else
  fail "markers missing or wrong namespace"
fi
if ! grep -q "scholar-full-paper" "$P1/CLAUDE.md"; then
  pass "no scholar-full-paper reference leaked into the block"
else
  fail "scholar-full-paper reference present (should be excluded in this fork)"
fi

# ─── Test 2: idempotent no-op ──────────────────────────────────────────
echo ""
echo "Test 2: re-run is an idempotent no-op"
BEFORE=$(cksum "$P1/CLAUDE.md")
OUT=$(SCHOLAR_HOST_AGENT_OVERRIDE=claude-code bash "$SCRIPT" "$P1" --mode lean 2>&1)
AFTER=$(cksum "$P1/CLAUDE.md")
if [ "$BEFORE" = "$AFTER" ] && echo "$OUT" | grep -q "no-op"; then
  pass "second run left the file byte-identical and reported no-op"
else
  fail "second run changed the file or did not report no-op"; echo "$OUT" | sed 's/^/      /'
fi

# ─── Test 3: host=codex → AGENTS.md ────────────────────────────────────
echo ""
echo "Test 3: host=codex writes AGENTS.md only"
P3="$TMPDIR_BASE/p3"
make_proj "$P3"
SCHOLAR_HOST_AGENT_OVERRIDE=codex bash "$SCRIPT" "$P3" --mode lean >/dev/null 2>&1
if [ -f "$P3/AGENTS.md" ] && [ ! -f "$P3/CLAUDE.md" ]; then
  pass "created AGENTS.md, no CLAUDE.md"
else
  fail "expected AGENTS.md only"
fi

# ─── Test 4: host=unknown → both ───────────────────────────────────────
echo ""
echo "Test 4: host=unknown writes both files"
P4="$TMPDIR_BASE/p4"
make_proj "$P4"
SCHOLAR_HOST_AGENT_OVERRIDE=unknown bash "$SCRIPT" "$P4" --mode lean >/dev/null 2>&1
if [ -f "$P4/CLAUDE.md" ] && [ -f "$P4/AGENTS.md" ]; then
  pass "created both CLAUDE.md and AGENTS.md"
else
  fail "expected both files"
fi

# ─── Test 5: append-preserve existing file without markers ─────────────
echo ""
echo "Test 5: append preserves existing user content"
P5="$TMPDIR_BASE/p5"
make_proj "$P5"
printf '# My Project\n\nImportant user notes.\n' > "$P5/CLAUDE.md"
SCHOLAR_HOST_AGENT_OVERRIDE=claude-code bash "$SCRIPT" "$P5" --mode lean >/dev/null 2>&1
if grep -q "Important user notes." "$P5/CLAUDE.md" \
   && grep -q "open-scholar-skill:BEGIN auto-rules" "$P5/CLAUDE.md"; then
  pass "user content preserved and block appended"
else
  fail "append did not preserve content or add block"
fi

# ─── Test 6: existing project never backfills the other file ───────────
echo ""
echo "Test 6: existing CLAUDE.md → AGENTS.md is not backfilled"
P6="$TMPDIR_BASE/p6"
make_proj "$P6"
printf '# Existing\n' > "$P6/CLAUDE.md"
# Detected host is codex, but since CLAUDE.md exists, only it should refresh.
SCHOLAR_HOST_AGENT_OVERRIDE=codex bash "$SCRIPT" "$P6" --mode lean >/dev/null 2>&1
if [ -f "$P6/CLAUDE.md" ] && [ ! -f "$P6/AGENTS.md" ]; then
  pass "refreshed existing CLAUDE.md without backfilling AGENTS.md"
else
  fail "unexpectedly backfilled AGENTS.md"
fi

# ─── Test 7: invalid --mode rejected ───────────────────────────────────
echo ""
echo "Test 7: --mode full is rejected (lean-only fork)"
P7="$TMPDIR_BASE/p7"
make_proj "$P7"
if SCHOLAR_HOST_AGENT_OVERRIDE=claude-code bash "$SCRIPT" "$P7" --mode full >/dev/null 2>&1; then
  fail "accepted --mode full"
else
  pass "rejected --mode full"
fi

# ─── Test 8: CFPS LOCAL_MODE conditional block ─────────────────────────
echo ""
echo "Test 8: CFPS .dta LOCAL_MODE injects the conditional block"
P8="$TMPDIR_BASE/p8"
make_proj "$P8" '{ "data/raw/cfps2018.dta": "LOCAL_MODE" }'
SCHOLAR_HOST_AGENT_OVERRIDE=claude-code bash "$SCRIPT" "$P8" --mode lean >/dev/null 2>&1
if grep -q "CFPS data handling" "$P8/CLAUDE.md"; then
  pass "CFPS LOCAL_MODE block injected"
else
  fail "CFPS LOCAL_MODE block missing"
fi

# A project without a CFPS .dta LOCAL_MODE entry must NOT get the block.
if ! grep -q "CFPS data handling" "$P1/CLAUDE.md"; then
  pass "no CFPS block when no CFPS .dta LOCAL_MODE entry"
else
  fail "CFPS block injected without a matching sidecar entry"
fi

# ─── Summary ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════"
echo "Results: $PASS passed, $FAIL failed"
if [ "$FAIL" -gt 0 ]; then
  echo ">>> FAILED"
  exit 1
else
  echo ">>> PASSED"
  exit 0
fi
