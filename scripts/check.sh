#!/usr/bin/env bash
# Everything CI runs, in one command, before you push.
#
# On 2026-09-20 a corrected figure changed a string that
# tests/test_parsers.py asserts on. The guard suite was re-run, the
# parser suite was not, and main went red. Running one command instead
# of remembering two is the whole point.
#
# As a pre-commit hook:
#   ln -sf ../../scripts/check.sh .git/hooks/pre-commit
# Skip it for a single commit with: git commit --no-verify
set -uo pipefail
cd "$(dirname "$0")/.."

# PREFLIGHT. A missing dependency is a reason not to push -- you have
# not actually verified anything -- but it is not the same as broken
# code, and saying so plainly keeps this from crying wolf. The desktop
# VM, for instance, has no scipy.
missing=$(python3 - <<'PY'
import importlib
print(" ".join(m for m in ("pandas", "numpy", "scipy", "pytest")
                if importlib.util.find_spec(m) is None))
PY
)
if [ -n "$missing" ]; then
  echo "ENVIRONMENT INCOMPLETE -- missing:$missing"
  echo "  the suites cannot run, so nothing here has been verified."
  echo "  fix:  pip install -r requirements.txt"
  exit 2
fi

fail=0
run() {
  printf '\n=== %s ===\n' "$1"; shift
  if "$@"; then :; else echo "  ^ FAILED"; fail=1; fi
}

run "core tests"    python3 -m pytest tests/core -q
run "test manifest" python3 tools/test_manifest.py --check
run "guard tests"   python3 tests/test_model_guards.py
run "parser tests"  python3 tests/test_parsers.py

if [ -d frontend/node_modules ]; then
  run "frontend build" bash -c 'cd frontend && npm run build'
else
  printf '\n=== frontend build ===\n  skipped (no frontend/node_modules; run: cd frontend && npm install)\n'
fi

printf '\n'
if [ "$fail" -ne 0 ]; then
  echo "CHECKS FAILED -- do not push."
  exit 1
fi
echo "all checks passed"
