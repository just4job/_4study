#!/usr/bin/env python3
"""Vá DGL ở máy LOCAL khi thiếu thư viện C++ graphbolt (.dll/.so).

Tên file nói "windows" vì lần đầu gặp trên Windows, nhưng apply_patches_only()
không phụ thuộc hệ điều hành — Linux/macOS dùng y hệt.

Triệu chứng nó sửa:
    FileNotFoundError: Cannot find DGL C++ graphbolt library at
        .../dgl/graphbolt/libgraphbolt_pytorch_<torch version>.so

DGL ship sẵn 1 file .so BUILD RIÊNG CHO TỪNG BẢN TORCH. Dùng torch mới hơn
mọi bản dgl đang có (vd. torch 2.14 với dgl 2.1.0) thì file đó không tồn tại
và sẽ không bao giờ có. Script bọc load_graphbolt() trong try/except và biến
`from . import distributed` thành tuỳ chọn, nên `import dgl` chạy tiếp.

An toàn với repo này: pipeline chỉ dùng core graph API (dgl.graph, to_simple,
remove_self_loop, edge_subgraph, add_self_loop), không đụng graphbolt hay
dgl.distributed. Chỉ train_Struct2GO.py (trainer CŨ) cần dgl.dataloading;
train_Struct2GO2.py dùng torch.utils.data.DataLoader.

Chạy:
  python scripts/fix_dgl_windows.py
  python -c "import dgl; print(dgl.__version__)"

Cài lại wheel CPU khớp torch trước khi vá:
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
