#!/usr/bin/env bash
# check-agents.sh — Lint agent, skill, and governance files
# Referenced by: ai/governance/workflow-governance.md, ai/skills/skill-lint-agents.md
# Run: ./scripts/check-agents.sh
# Exit 0: all checks pass. Exit 1: one or more checks failed.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PASS=0
FAIL=0

ok()   { echo "  [OK]   $1"; }
fail() { echo "  [FAIL] $1"; FAIL=$((FAIL + 1)); }

lint_agents_md() {
  local file="$REPO_ROOT/AGENTS.md"
  echo ""
  echo "Checking $file ..."

  if [[ ! -f "$file" ]]; then
    fail "AGENTS.md not found"
    return
  fi

  for field in "file:" "version:" "purpose:" "priority:" "security_model:"; do
    if grep -q "^${field}" "$file"; then
      ok "frontmatter: ${field}"
    else
      fail "frontmatter missing: ${field}"
    fi
  done

  for section in "SYSTEM ROLE" "AUTHORITY HIERARCHY" "OPERATIONAL BOUNDARIES" "SECURITY MODEL"; do
    if grep -q "# .*${section}" "$file"; then
      ok "section present: ${section}"
    else
      fail "section missing: ${section}"
    fi
  done
}

lint_skill_files() {
  local skills_dir="$REPO_ROOT/ai/skills"
  echo ""
  echo "Checking skill files in $skills_dir ..."

  if [[ ! -d "$skills_dir" ]]; then
    fail "ai/skills/ directory not found"
    return
  fi

  local count=0
  for skill in "$skills_dir"/skill-*.md; do
    [[ -f "$skill" ]] || continue
    local name
    name="$(basename "$skill")"
    count=$((count + 1))

    if grep -q "^# Skill:" "$skill"; then
      ok "$name: has '# Skill:' header"
    else
      fail "$name: missing '# Skill:' header"
    fi

    if grep -q "Executes:" "$skill"; then
      ok "$name: has 'Executes:' section"
    else
      fail "$name: missing 'Executes:' section"
    fi
  done

  if [[ $count -eq 0 ]]; then
    fail "No skill-*.md files found in ai/skills/"
  else
    ok "Found $count skill file(s)"
  fi
}

lint_governance_files() {
  local gov_dir="$REPO_ROOT/ai/governance"
  echo ""
  echo "Checking governance files in $gov_dir ..."

  if [[ ! -d "$gov_dir" ]]; then
    fail "ai/governance/ directory not found"
    return
  fi

  local count=0
  for g in "$gov_dir"/*.md; do
    [[ -f "$g" ]] || continue
    local name
    name="$(basename "$g")"
    count=$((count + 1))

    if grep -q "^#" "$g"; then
      ok "$name: has at least one heading"
    else
      fail "$name: no headings found"
    fi
  done

  if [[ $count -eq 0 ]]; then
    fail "No .md files found in ai/governance/"
  else
    ok "Found $count governance file(s)"
  fi
}

# ── Run all checks ──────────────────────────────────────────────────────────
echo "=============================="
echo " check-agents.sh"
echo "=============================="

lint_agents_md
lint_skill_files
lint_governance_files

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo "=============================="
if [[ $FAIL -eq 0 ]]; then
  echo " RESULT: PASS (all checks OK)"
  echo "=============================="
  exit 0
else
  echo " RESULT: FAIL ($FAIL check(s) failed)"
  echo "=============================="
  exit 1
fi
