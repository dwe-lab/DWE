#!/usr/bin/env bash
# Install locked artifacts, optionally from a reachable PEP 503 mirror.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PWD/.local/cache/uv}"
uv_bin="${UV_BIN:-uv}"
mkdir -p .local/setup
if [[ -n "${PACKAGE_INDEX:-}" ]]; then
    if [[ ! -x .venv/bin/python ]]; then
        "$uv_bin" venv --python "${PYTHON_BIN:-python3.12}" .venv
    fi
    "$uv_bin" export --locked --all-extras --no-emit-project \
        --format requirements-txt --output-file .local/setup/requirements.txt > /dev/null
    "$uv_bin" pip sync --python .venv/bin/python --require-hashes \
        --default-index "$PACKAGE_INDEX" .local/setup/requirements.txt
    "$uv_bin" pip install --python .venv/bin/python --no-deps \
        --default-index "$PACKAGE_INDEX" -e .
else
    "$uv_bin" sync --locked --all-extras
fi
.venv/bin/python - <<'PY'
import pathlib
import sysconfig

header = pathlib.Path(sysconfig.get_path("include"), "Python.h")
if not header.is_file():
    raise SystemExit(
        f"Missing {header}. Use a Python runtime with development headers for Triton compilation."
    )
print(f"Python development header: {header}")
PY

.venv/bin/python scripts/patch_trl.py
