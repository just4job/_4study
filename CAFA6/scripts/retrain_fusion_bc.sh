#!/usr/bin/env bash
# Retrain + test B (PPI+concat) và C (no-PPI+attention) — profile match_a
set -euo pipefail

export DATA_DIR="${DATA_DIR:-/kaggle/working/CAFA6}"
export DGL_CUDA="${DGL_CUDA:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

# Mặc định: 3 nhánh × 2 config, epoch giống phương pháp A
exec python scripts/retrain_fusion_bc.py --profile match_a "$@"
