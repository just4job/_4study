"""Kaggle: dgl==2.1.0 + patch all torchdata imports in site-packages/dgl."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_REPO))


def pip(*args: str) -> int:
    return subprocess.run([sys.executable, "-m", "pip", *args]).returncode


def install_dgl() -> None:
    import torch

    pip("uninstall", "-y", "dgl", "torchdata")
    cuda = torch.version.cuda
    if cuda:
        cu = "cu" + cuda.replace(".", "")
        tv = ".".join(torch.__version__.split(".")[:2])
        for url in (
            f"https://data.dgl.ai/wheels/torch-{tv}/{cu}/repo.html",
            f"https://data.dgl.ai/wheels/{cu}/repo.html",
            "https://data.dgl.ai/wheels/cu124/repo.html",
        ):
            if pip("install", "-q", "dgl==2.1.0", "-f", url) == 0:
                return
    pip("install", "-q", "dgl==2.1.0")


def main() -> None:
    install_dgl()
    from model.dgl_patch import ensure_dgl_importable

    ensure_dgl_importable()
    import torch

    import dgl

    g = dgl.graph(([0, 1], [1, 2]))
    if torch.cuda.is_available():
        g = g.to("cuda")
    print(
        f"OK  DGL {dgl.__version__}  PyTorch {torch.__version__}  "
        f"CUDA {torch.version.cuda}  device={g.device}"
    )


if __name__ == "__main__":
    main()
