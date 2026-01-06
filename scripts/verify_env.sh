#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="${0:A:h:h}"
cd "$REPO_ROOT"

# Match the setup script defaults unless VENV_DIR is explicitly provided.
DEFAULT_ENV_NAME="reasoning-embedder"
ENV_NAME="${ENV_NAME:-$DEFAULT_ENV_NAME}"
SAFE_NAME="${ENV_NAME// /-}"
SAFE_NAME="${SAFE_NAME//[^A-Za-z0-9._-]/-}"

# macOS-only: run under Rosetta if available.
ARCH_PREFIX=()
if command -v arch >/dev/null 2>&1; then
  if arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
    ARCH_PREFIX=(arch -x86_64)
  fi
fi

if [[ -z "${VENV_DIR:-}" ]]; then
  if [[ ${#ARCH_PREFIX[@]} -gt 0 ]]; then
    VENV_DIR=".venv-${SAFE_NAME}-x86"
  else
    VENV_DIR=".venv-${SAFE_NAME}"
  fi
fi

if [[ ! -x "$REPO_ROOT/$VENV_DIR/bin/python" ]]; then
  echo "ERROR: venv not found at $VENV_DIR. Run scripts/setup_rosetta_venv.sh first (or set VENV_DIR)." >&2
  exit 1
fi

PY="$REPO_ROOT/$VENV_DIR/bin/python"

# Sanity: confirm runtime arch (x86_64 under Rosetta when available; otherwise native)
if [[ ${#ARCH_PREFIX[@]} -gt 0 ]]; then
  "${ARCH_PREFIX[@]}" "$PY" -c "import platform; assert platform.machine()=='x86_64'; print('✓ x86_64 runtime confirmed')"
else
  "$PY" -c "import platform; print('✓ native runtime confirmed:', platform.machine())"
fi

# Basic imports and core smoke test
"${ARCH_PREFIX[@]}" "$PY" scripts/smoke_test.py

# Run unit tests (includes HF-optional prepare test; should skip unless env var set)
"${ARCH_PREFIX[@]}" "$PY" -m pytest -q

echo "✓ verify_env OK"
