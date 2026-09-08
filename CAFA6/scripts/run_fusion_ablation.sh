#!/usr/bin/env bash
# Train + eval ablation fusion — 3 nhánh × 2 hướng (Kaggle / Linux)
set -euo pipefail

export DATA_DIR="${DATA_DIR:-/kaggle/working/CAFA6}"
export DGL_CUDA="${DGL_CUDA:-1}"

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python scripts/run_fusion_ablation.py "$@"
