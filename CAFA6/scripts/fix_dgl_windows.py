#!/usr/bin/env python3
"""Patch DGL on Windows/local when graphbolt DLL missing (final_pro env).

Run:
  python scripts/fix_dgl_windows.py
  python -c "import dgl; print(dgl.__version__)"

Optional reinstall CPU wheel matching torch:
  python scripts/fix_dgl_windows.py --reinstall-cpu
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.kaggle_fix_dgl import apply_patches_only, pip  # noqa: E402


def reinstall_dgl_cpu() -> None:
    import torch

    tv = ".".join(torch.__version__.split(".")[:2])
    pip("uninstall", "-y", "dgl")
    urls = [
        f"https://data.dgl.ai/wheels/torch-{tv}/cpu/repo.html",
        "https://data.dgl.ai/wheels/repo.html",
    ]
    for url in urls:
        print(f"[install] dgl -f {url}")
        if pip("install", "dgl==2.4.0", "-f", url) == 0:
            break
    else:
        pip("install", "dgl==2.4.0")


def main() -> None:
    if "--reinstall-cpu" in sys.argv:
        reinstall_dgl_cpu()
    apply_patches_only()
    import dgl

    g = dgl.graph(([0, 1], [1, 2]))
    print("OK", "dgl", dgl.__version__, "nodes", g.num_nodes())


if __name__ == "__main__":
    main()
