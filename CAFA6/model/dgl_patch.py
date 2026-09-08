"""Ensure DGL imports on Kaggle (CUDA + torchdata patches)."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _fix_script() -> Path:
    return Path(__file__).resolve().parents[1] / "scripts" / "kaggle_fix_dgl.py"


def _run_fix_script(*extra: str) -> None:
    subprocess.run([sys.executable, str(_fix_script()), *extra], check=True)


def _dgl_cuda_ok() -> bool:
    """Apply patches then test CUDA in-process (subprocess import skips patches)."""
    try:
        _run_fix_script("--no-install")
        import dgl
        import torch

        g = dgl.graph(([0, 1], [1, 2]))
        if torch.cuda.is_available():
            g = g.to("cuda")
        return str(g.device).startswith("cuda")
    except Exception:
        return False


def ensure_dgl_importable(verbose: bool = True) -> None:
    """Install/patch DGL once per session; skip pip if CUDA already works."""
    if _dgl_cuda_ok():
        if verbose:
            print("[dgl_patch] DGL CUDA OK")
        return

    if verbose:
        print("[dgl_patch] Installing DGL CUDA + patches...")
    _run_fix_script()


def apply_dgl_patches_only(verbose: bool = True) -> None:
    """Apply on-disk DGL patches without attempting any pip install."""
    if verbose:
        print("[dgl_patch] Applying DGL patches only...")
    _run_fix_script("--no-install")


def main() -> None:
    ensure_dgl_importable()


if __name__ == "__main__":
    main()
