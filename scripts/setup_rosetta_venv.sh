#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="${0:A:h:h}"
cd "$REPO_ROOT"

PY_X86="${PY_X86:-/usr/local/bin/python3.11}"

# macOS-specific: `arch -x86_64 <cmd>` runs under Rosetta when available.
# On other machines, we fall back to native execution.
ARCH_PREFIX=()
if command -v arch >/dev/null 2>&1; then
  # This is a safe probe on macOS; on non-macOS `arch` typically doesn't accept -x86_64.
  if arch -x86_64 /usr/bin/true >/dev/null 2>&1; then
    ARCH_PREFIX=(arch -x86_64)
  fi
fi

if [[ ${#ARCH_PREFIX[@]} -gt 0 ]]; then
  VENV_DIR="${VENV_DIR:-.venv-x86}"
else
  VENV_DIR="${VENV_DIR:-.venv}"
fi

if [[ ${#ARCH_PREFIX[@]} -gt 0 ]]; then
  if [[ ! -x "$PY_X86" ]]; then
    echo "ERROR: x86_64 Python not found at $PY_X86" >&2
    echo "Install Rosetta + x86_64 Homebrew python@3.11 so /usr/local/bin/python3.11 exists, or set PY_X86." >&2
    exit 1
  fi

  # Ensure we can run it under Rosetta
  "${ARCH_PREFIX[@]}" "$PY_X86" -c "import platform; assert platform.machine()=='x86_64'; print('✓ Using x86_64 Python (Rosetta)')"
  PYTHON="$PY_X86"
else
  PYTHON="${PYTHON:-python3}"
  "$PYTHON" -c "import platform; print('✓ Using native Python on', platform.machine())"
fi

if [[ -d "$VENV_DIR" ]]; then
  echo "Removing existing $VENV_DIR"
  rm -rf "$VENV_DIR"
fi

echo "Creating venv: $VENV_DIR"
"${ARCH_PREFIX[@]}" "$PYTHON" -m venv "$VENV_DIR"

PIP="$REPO_ROOT/$VENV_DIR/bin/pip"
PY="$REPO_ROOT/$VENV_DIR/bin/python"

# Upgrade core tooling
"${ARCH_PREFIX[@]}" "$PIP" install --upgrade pip setuptools wheel

# Install dependencies (keep editable install last)
if [[ -f requirements.txt ]]; then
  "${ARCH_PREFIX[@]}" "$PIP" install -r requirements.txt
fi

# Install project
"${ARCH_PREFIX[@]}" "$PIP" install -e . --no-deps

# Print environment info
"${ARCH_PREFIX[@]}" "$PY" -c "import platform,sys; print('python',sys.version.split()[0]); print('machine',platform.machine()); print('executable',sys.executable)"

if [[ ${#ARCH_PREFIX[@]} -gt 0 ]]; then
  echo "✓ Rosetta venv ready: $VENV_DIR"
else
  echo "✓ Venv ready: $VENV_DIR"
fi
