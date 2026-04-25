#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MICROMAMBA_BIN="/home/hansol/tvm_xgb_diag/artifacts/micromamba/bin/micromamba"
ENV_ROOT="/home/hansol/tvm_xgb_diag/envs/wheel_xgb176_iso"

export PYTHONNOUSERSITE=1

exec "$MICROMAMBA_BIN" run -p "$ENV_ROOT" \
  python "$PROJECT_ROOT/tools/run_cpu_transfer_memory_benchmark.py" "$@"
