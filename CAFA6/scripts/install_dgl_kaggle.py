"""
Install DGL with CUDA on Kaggle T4.

Run in notebook (after Restart session if pip/torch were broken):
  !python /kaggle/working/CAFA6/scripts/kaggle_fix_dgl.py

This file is a thin alias for backward compatibility.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    script = _REPO / "scripts" / "kaggle_fix_dgl.py"
    print(f"Running {script} ...")
    subprocess.run([sys.executable, str(script)], check=True)


if __name__ == "__main__":
    main()
