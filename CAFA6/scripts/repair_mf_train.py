#!/usr/bin/env python3
"""Copy mf_train_dataset hợp lệ từ /kaggle/input → DATA_DIR/divided_data.

final-data: mf_train thường EOF; mf_valid OK (422 labels).
mf-train1: đọc được nhưng 5136 labels — KHÔNG dùng với valid final-data.
Upload kaggle_mf_train.zip từ pack_for_kaggle.py (--branch mf --splits train).
"""
from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]


def _register_pickle() -> None:
    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import __main__

    from data_processing.divide_data import MyDataSet

    __main__.MyDataSet = MyDataSet


def _try_load(path: Path) -> int | None:
    try:
        with open(path, "rb") as handle:
            ds = pickle.load(handle)
        return len(ds) if hasattr(ds, "__len__") else 0
    except Exception:
        return None


def _label_dim(path: Path) -> int | None:
    try:
        import numpy as np

        with open(path, "rb") as handle:
            ds = pickle.load(handle)
        return int(np.asarray(ds[0][2]).reshape(-1).shape[0])
    except Exception:
        return None


def find_best_mf_train(
    search_roots: list[Path],
    required_label_dim: int | None = None,
) -> tuple[Path, int]:
    """Chọn mf_train đọc được; ưu tiên label_dim khớp valid (422 cho CAFA6)."""
    candidates: list[tuple[Path, int, int | None]] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("mf_train_dataset"):
            if not path.is_file() or path.stat().st_size < 500_000_000:
                continue
            n = _try_load(path)
            dim = _label_dim(path) if n is not None else None
            status = "OK" if n is not None else "FAIL"
            dim_s = f"labels={dim}" if dim is not None else "labels=?"
            print(
                f"  [{status}] {path.stat().st_size / 1e6:8.1f} MB  n={n}  {dim_s}  {path}"
            )
            if n is not None:
                candidates.append((path, n, dim))

    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy mf_train_dataset đọc được trong /kaggle/input. "
            "Upload kaggle_mf.zip (422 labels) từ pack_for_kaggle.py trên máy local."
        )

    if required_label_dim is not None:
        matched = [c for c in candidates if c[2] == required_label_dim]
        if matched:
            best = max(matched, key=lambda c: c[0].stat().st_size)
            print(f"\n  → chọn theo label_dim={required_label_dim}: {best[0]}")
            return best[0], best[1]
        dims = sorted({c[2] for c in candidates if c[2] is not None})
        raise FileNotFoundError(
            f"Không có mf_train labels={required_label_dim} trong /kaggle/input "
            f"(tìm thấy labels={dims}). "
            "KHÔNG dùng mf-train1 (5136) với valid final-data (422). "
            "Upload pack_for_kaggle.py --branch mf --splits train."
        )

    best = max(candidates, key=lambda c: c[0].stat().st_size)
    if best[2] not in (None, 422):
        print(
            f"\n  [WARN] Chọn file lớn nhất labels={best[2]} — "
            "có thể không khớp mf_valid (422). Kiểm tra trước train."
        )
    return best[0], best[1]


def _copy_split(src: Path, data_dir: Path, split: str) -> int | None:
    import shutil

    dst = data_dir / "divided_data" / f"mf_{split}_dataset"
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    shutil.copy2(src, dst)
    n = _try_load(dst)
    print(f"  mf_{split}: {src.stat().st_size / 1e6:.1f} MB -> {dst.name}  n={n}")
    return n


def _infer_required_label_dim(data_dir: Path) -> int | None:
    valid_dst = data_dir / "divided_data" / "mf_valid_dataset"
    if valid_dst.is_file():
        return _label_dim(valid_dst)
    return None


def mf_train_ok(data_dir: Path, required_label_dim: int | None = None) -> bool:
    """True nếu mf_train đọc được và khớp label_dim với valid (nếu có)."""
    dst = data_dir / "divided_data" / "mf_train_dataset"
    if not dst.is_file() or dst.stat().st_size < 500_000_000:
        return False
    if _try_load(dst) is None:
        return False
    dim_train = _label_dim(dst)
    dim_required = required_label_dim or _infer_required_label_dim(data_dir)
    if dim_required is not None and dim_train is not None and dim_train != dim_required:
        return False
    return True


def repair_mf_train(
    data_dir: Path,
    input_root: Path | None = None,
    *,
    required_label_dim: int | None = None,
    train_only: bool = False,
) -> Path:
    """Copy mf_train từ input; mặc định giữ mf_valid/test đã link từ final-data."""
    _register_pickle()
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "divided_data").mkdir(parents=True, exist_ok=True)

    dim_required = required_label_dim or _infer_required_label_dim(data_dir)
    if dim_required is not None:
        print(f"=== Yêu cầu mf_train labels={dim_required} (khớp mf_valid) ===")

    roots = []
    if input_root is not None:
        roots.append(input_root)
    roots.append(Path("/kaggle/input"))

    print("=== Tìm mf_train hợp lệ ===")
    src_train, n_train = find_best_mf_train(roots, required_label_dim=dim_required)
    package_dir = src_train.parent

    print(f"\n=== Copy mf_train từ: {src_train} ===")
    _copy_split(src_train, data_dir, "train")

    if not train_only:
        for split in ("valid", "test"):
            sibling = package_dir / f"mf_{split}_dataset"
            valid_dst = data_dir / "divided_data" / f"mf_{split}_dataset"
            if valid_dst.is_file() and split == "valid" and dim_required is not None:
                print(f"  [KEEP] giữ mf_{split} hiện tại (final-data, labels={dim_required})")
                continue
            if sibling.is_file():
                n = _copy_split(sibling, data_dir, split)
                if n is None:
                    print(f"  [WARN] mf_{split} trong package không đọc được: {sibling}")
            else:
                print(f"  [SKIP] không có mf_{split} trong package")

    dst = data_dir / "divided_data" / "mf_train_dataset"
    n2 = _try_load(dst)
    if n2 is None:
        raise RuntimeError(f"VERIFY FAIL sau copy: {dst}")

    valid_dst = data_dir / "divided_data" / "mf_valid_dataset"
    dim_train = _label_dim(dst)
    if valid_dst.is_file():
        n_v = _try_load(valid_dst)
        dim_valid = _label_dim(valid_dst)
        print(f"\nVERIFY train n={n2} labels={dim_train}, valid n={n_v} labels={dim_valid}")
        if dim_train is not None and dim_valid is not None and dim_train != dim_valid:
            raise RuntimeError(
                f"mf_train ({dim_train} labels) và mf_valid ({dim_valid} labels) KHÔNG KHỚP. "
                "mf-train1 (5136) không dùng được với final-data valid (422). "
                "Upload mf_train từ pack_for_kaggle.py (cùng 422 labels) hoặc bỏ MF ablation."
            )
    else:
        print(
            "\n[WARN] Không có mf_valid trong package — kiểm tra label_dim thủ công trước train."
        )

    print(f"VERIFY OK — mf_train n={n2}, labels={dim_train}")
    return dst


def main() -> int:
    parser = argparse.ArgumentParser(description="Sửa mf_train_dataset trên Kaggle")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(os.environ.get("DATA_DIR", REPO)),
    )
    parser.add_argument(
        "--require-label-dim",
        type=int,
        default=None,
        help="Chỉ chấp nhận mf_train có số GO terms này (vd. 422)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Chỉ copy mf_train; giữ mf_valid/test từ final-data",
    )
    args = parser.parse_args()
    try:
        repair_mf_train(
            args.data_dir,
            required_label_dim=args.require_label_dim,
            train_only=args.train_only,
        )
    except Exception as exc:
        print(f"FAIL: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
