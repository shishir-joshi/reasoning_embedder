#!/usr/bin/env zsh
set -euo pipefail

REPO_ROOT="${0:A:h:h}"
cd "$REPO_ROOT"

PY_X86="${PY_X86:-/usr/local/bin/python3.11}"

# Logical env/kernel name (used for venv default name + Jupyter kernelspec).
DEFAULT_ENV_NAME="reasoning-embedder"
ENV_NAME="${ENV_NAME:-}"
if [[ -z "$ENV_NAME" && -t 0 ]]; then
  echo -n "Environment/kernel name [${DEFAULT_ENV_NAME}]: "
  read -r ENV_NAME
fi
ENV_NAME="${ENV_NAME:-$DEFAULT_ENV_NAME}"

# Sanitize for filesystem + kernelspec name.
SAFE_NAME="${ENV_NAME// /-}"
SAFE_NAME="${SAFE_NAME//[^A-Za-z0-9._-]/-}"
if [[ -z "$SAFE_NAME" ]]; then
  echo "ERROR: Could not derive a safe ENV_NAME from input: '$ENV_NAME'" >&2
  exit 1
fi

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
  VENV_DIR="${VENV_DIR:-.venv-${SAFE_NAME}-x86}"
else
  VENV_DIR="${VENV_DIR:-.venv-${SAFE_NAME}}"
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

# Configure pip cache usage (default: use cache for speed; set NO_CACHE=1 to disable)
PIP_CACHE_FLAG=""
if [[ "${NO_CACHE:-}" == "1" ]]; then
  PIP_CACHE_FLAG="--no-cache-dir"
  echo "ℹ️  pip cache disabled (NO_CACHE=1)"
fi

# Upgrade core tooling
"${ARCH_PREFIX[@]}" "$PIP" install $PIP_CACHE_FLAG --upgrade pip setuptools wheel

# Install dependencies (keep editable install last)
if [[ -f requirements.txt ]]; then
  "${ARCH_PREFIX[@]}" "$PIP" install $PIP_CACHE_FLAG -r requirements.txt
fi

# Install project
"${ARCH_PREFIX[@]}" "$PIP" install $PIP_CACHE_FLAG -e . --no-deps

# Print environment info
"${ARCH_PREFIX[@]}" "$PY" -c "import platform,sys; print('python',sys.version.split()[0]); print('machine',platform.machine()); print('executable',sys.executable)"

# Register Jupyter kernel
ARCH="$(${ARCH_PREFIX[@]} "$PY" -c "import platform; print(platform.machine())")"
KERNEL_NAME="${KERNEL_NAME:-$SAFE_NAME}"
KERNEL_NAME="${KERNEL_NAME// /-}"
KERNEL_NAME="${KERNEL_NAME//[^A-Za-z0-9._-]/-}"
if [[ -z "$KERNEL_NAME" ]]; then
  echo "ERROR: Could not derive a safe KERNEL_NAME from input." >&2
  exit 1
fi
DISPLAY_NAME="${DISPLAY_NAME:-${ENV_NAME} (${ARCH})}"

# Idempotency: ipykernel's CLI flags vary by version; we explicitly remove any
# existing kernelspec directory before installing.
KERNELS_DIR="$(${ARCH_PREFIX[@]} "$PY" -c "import os; import jupyter_core.paths as p; print(os.path.join(p.jupyter_data_dir(), 'kernels'))")"
TARGET_KERNEL_DIR="$KERNELS_DIR/$KERNEL_NAME"
if [[ -d "$TARGET_KERNEL_DIR" ]]; then
  rm -rf "$TARGET_KERNEL_DIR"
fi

"${ARCH_PREFIX[@]}" "$PY" -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$DISPLAY_NAME"

echo "✓ Registered Jupyter kernel: $DISPLAY_NAME (name: $KERNEL_NAME)"

if [[ ${#ARCH_PREFIX[@]} -gt 0 ]]; then
  echo "✓ Rosetta venv ready: $VENV_DIR"
else
  echo "✓ Venv ready: $VENV_DIR"
fi
