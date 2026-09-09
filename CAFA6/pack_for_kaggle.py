"""
pack_for_kaggle.py
==================
Đóng gói các file cần thiết để upload lên Kaggle thành 1 zip.

Output: kaggle_data.zip với cấu trúc:
  divided_data/
      mf_train_dataset
      mf_valid_dataset
      bp_train_dataset (nếu có)
      bp_valid_dataset (nếu có)
      cc_train_dataset (nếu có)
      cc_valid_dataset (nếu có)
  proceed_data/
      label_mf_network
      label_bp_network (nếu có)
      label_cc_network (nếu có)
      ppi_graph_global
      ppi_protein_index
      split_mf.json / split_bp.json / split_cc.json     (nếu đã chạy split_protein_ids.py)
      ppi_graph_train_mf / _bp / _cc                    (nếu đã chạy 4_build_ppi_graph.py sau đó)

Sau khi tạo xong:
  1. Vào https://www.kaggle.com/datasets → New Dataset
  2. Upload kaggle_data.zip
  3. Đặt tên dataset, ví dụ "cafa6-data"
"""
import json
import os
import sys
import zipfile
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DIVIDED_DIR = BASE_DIR / "divided_data"
PROC_DIR    = BASE_DIR / "proceed_data"


def collect_files(branches: tuple[str, ...], splits: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for branch in branches:
        label_path = PROC_DIR / f"label_{branch}_network"
        vocab_path = PROC_DIR / f"label_vocab_{branch}.json"
        for split in splits:
            path = DIVIDED_DIR / f"{branch}_{split}_dataset"
            if path.exists():
                files.append(path)
            else:
                print(f"[WARN] Thiếu {path}")
        if label_path.exists():
            files.append(label_path)
        if vocab_path.exists():
            files.append(vocab_path)
        acs_path = PROC_DIR / f"human_{branch.upper()}_ACS.json"
        if acs_path.exists():
            files.append(acs_path)
        # PPI leakage guard build-time (xem README mục 4.5): nếu đã chạy
        # split_protein_ids.py + 4_build_ppi_graph.py, pack luôn split_{ns}.json +
        # ppi_graph_train_{ns} — train_Struct2GO2.py trên Kaggle sẽ dùng thẳng, không
        # cần tự mask lúc runtime (đỡ phải load {branch}_test_dataset chỉ để mask).
        split_path = PROC_DIR / f"split_{branch}.json"
        ppi_train_path = PROC_DIR / f"ppi_graph_train_{branch}"
        if split_path.exists():
            files.append(split_path)
        if ppi_train_path.exists():
            files.append(ppi_train_path)
        elif split_path.exists():
            print(
                f"[WARN] Có split_{branch}.json nhưng thiếu ppi_graph_train_{branch} — "
                "chạy lại data_processing/4_build_ppi_graph.py để build."
            )

    if branches == ("mf", "cc", "bp"):
        for f in (PROC_DIR / "ppi_graph_global", PROC_DIR / "ppi_protein_index"):
            if f.exists():
                files.append(f)
            else:
                print(f"[WARN] Thiếu {f}")
    return files


def total_size_mb(paths):
    return sum(p.stat().st_size for p in paths) / (1024 * 1024)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Pack CAFA6 data for Kaggle upload")
    parser.add_argument(
        "--branch",
        choices=["mf", "cc", "bp", "all"],
        default="all",
        help="Chỉ đóng gói một nhánh (vd. mf sau khi sửa mf_train)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Tên file zip (mặc định: kaggle_data.zip hoặc kaggle_mf.zip)",
    )
    parser.add_argument(
        "--splits",
        nargs="+",
        default=["train", "valid", "test"],
        choices=["train", "valid", "test"],
        help="Chỉ pack một split (vd. --splits train ~3.5GB)",
    )
    args = parser.parse_args()

    branches = ("mf", "cc", "bp") if args.branch == "all" else (args.branch,)
    required_files = collect_files(branches, tuple(args.splits))
    out_zip = args.output or (BASE_DIR / ("kaggle_data.zip" if args.branch == "all" else f"kaggle_{args.branch}.zip"))

    if not required_files:
        print("ERROR: Không tìm thấy file dataset nào. Đã chạy data_processing/divide_data.py chưa?")
        sys.exit(1)

    # Verify MF label count before upload — so với label_vocab_mf.json (Bước 2b,
    # đã lọc min-count đúng) thay vì hardcode 422 (số cũ từ 1 lần build thủ công
    # trước khi có bộ lọc đúng trong pipeline, xem README mục 3 — Bước 2b).
    if "mf" in branches and (DIVIDED_DIR / "mf_train_dataset").exists():
        try:
            import pickle
            import numpy as np
            import __main__
            from data_processing.divide_data import MyDataSet
            __main__.MyDataSet = MyDataSet
            with open(DIVIDED_DIR / "mf_train_dataset", "rb") as f:
                ds = pickle.load(f)
            dim = int(np.asarray(ds[0][2]).reshape(-1).shape[0])
            print(f"mf_train: n={len(ds)} labels={dim}")

            vocab_path = PROC_DIR / "label_vocab_mf.json"
            expected_dim = None
            expected_source = None
            if vocab_path.exists():
                with open(vocab_path, "r", encoding="utf-8") as f:
                    expected_dim = len(json.load(f))
                expected_source = "label_vocab_mf.json"
            else:
                expected_dim = 422
                expected_source = "số cũ 'final-data' (fallback — chưa chạy split_protein_ids.py)"

            if dim != expected_dim:
                print(
                    f"[WARN] mf_train labels={dim} khác {expected_source} "
                    f"(labels={expected_dim}). Kiểm tra lại trước khi upload."
                )
        except Exception as exc:
            print(f"[WARN] Không verify mf_train: {exc}")

    total_mb = total_size_mb(required_files)
    print(f"\nTotal size: {total_mb:,.1f} MB ({total_mb/1024:.2f} GB)")
    print(f"File count: {len(required_files)}\n")

    print("Files to pack:")
    for f in required_files:
        size_mb = f.stat().st_size / (1024 * 1024)
        rel = f.relative_to(BASE_DIR)
        print(f"  {size_mb:>8.1f} MB  {rel}")
    print()

    if total_mb > 20 * 1024:
        print("[WARN] Tổng > 20 GB — Kaggle giới hạn 20GB/dataset. Cân nhắc tách thành nhiều dataset.")

    # Tạo zip với compression nhẹ (pickle đã khá compact)
    print(f"Creating {out_zip}...")
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED, compresslevel=1) as zf:
        for f in required_files:
            arcname = f.relative_to(BASE_DIR)
            print(f"  + {arcname}")
            zf.write(f, arcname=str(arcname))

    out_size_mb = out_zip.stat().st_size / (1024 * 1024)
    print(f"\nDone: {out_zip} ({out_size_mb:,.1f} MB)")
    print("\nNext steps:")
    print("  1. https://www.kaggle.com/datasets -> New Dataset")
    print("  2. Upload kaggle_data.zip")
    print("  3. Name dataset (e.g. cafa6-data)")
    print("  4. New Notebook -> Add Data -> select dataset")
    print("  5. Copy cells from kaggle_notebook.md")


if __name__ == "__main__":
    main()
