#!/usr/bin/env python3
"""Nâng cấp DATA CŨ (build theo pipeline trên `main`, chưa có Bước 2b) để chạy
đúng với code mới — KHÔNG cần build lại từ đầu, KHÔNG đụng tới divided_data.

Vì sao làm được: mọi thứ cần thiết đã nằm sẵn trong bản pack cũ —
  - `divided_data/{ns}_{train,valid,test}_dataset`  -> suy ra ĐÚNG split đã dùng
  - `proceed_data/ppi_graph_global` + `ppi_protein_index` -> dựng ppi_graph_train_{ns}
  - `proceed_data/human_{NS}_ACS.json` + `label_vocab_{ns}.json` -> dựng lại
    label_{ns}_network CHỈ từ protein train

Script tạo/cập nhật 3 thứ cho mỗi nhánh:
  [1] proceed_data/split_{ns}.json        (suy ra từ divided_data — giữ nguyên
                                           phân vùng cũ, KHÔNG chia lại)
  [2] proceed_data/ppi_graph_train_{ns}   (PPI leakage guard build-time)
  [3] proceed_data/label_{ns}_network     (co-occurrence chỉ từ train; bản cũ
                                           được backup thành *.bak)

KHÔNG sửa được bằng script này: bộ lọc tần suất GO term (vocab min-count). Chiều
nhãn đã "đóng băng" trong emb_label_* / divided_data nên muốn lọc vocab thì phải
chạy lại pipeline từ Bước 2b -> 6 -> 7 -> 8 ở máy local (xem README mục 3).

Cách chạy:
    python scripts/migrate_old_data.py --dry-run     # xem trước, không ghi gì
    python scripts/migrate_old_data.py               # chạy thật
    python scripts/migrate_old_data.py --branch mf
    DATA_DIR=/kaggle/working/CAFA6 python scripts/migrate_old_data.py --materialize

Trên Kaggle `proceed_data` là symlink tới /kaggle/input (chỉ đọc) — dùng
`--materialize` để copy nó thành thư mục ghi được trong /kaggle/working trước.
Sau khi chạy, kiểm tra lại bằng: python scripts/audit_data.py
"""
from __future__ import annotations

import argparse
import gc
import json
import os
import pickle
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from data_processing.split_utils import build_train_only_ppi_graph, save_split  # noqa: E402

ACS_FILES = {"mf": "human_MF_ACS.json", "cc": "human_CC_ACS.json", "bp": "human_BP_ACS.json"}
SPLITS = ("train", "valid", "test")


def _load_pickle(path: Path):
    import __main__

    from data_processing.divide_data import MyDataSet

    __main__.MyDataSet = MyDataSet
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path) -> None:
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def materialize_proceed_data(proc: Path) -> None:
    """Kaggle: proceed_data là symlink tới /kaggle/input (read-only) — copy thành
    thư mục thật để ghi được. divided_data (nặng hàng GB) KHÔNG bị đụng tới."""
    if not proc.is_symlink():
        return
    real = proc.resolve()
    size_mb = sum(f.stat().st_size for f in real.rglob("*") if f.is_file()) / 1e6
    print(f"  proceed_data là symlink -> {real} ({size_mb:,.0f} MB), đang copy thành thư mục ghi được…")
    tmp = proc.parent / f"{proc.name}_rw_tmp"
    if tmp.exists():
        shutil.rmtree(tmp)
    shutil.copytree(real, tmp, symlinks=False)
    proc.unlink()
    tmp.rename(proc)
    print(f"  OK — {proc} giờ là thư mục thật, ghi được")


def derive_split(div: Path, branch: str) -> dict[str, list[str]] | None:
    """Đọc protein ID thật trong từng dataset đã chia — giữ NGUYÊN phân vùng cũ.

    Load lần lượt từng split rồi giải phóng ngay (dataset có thể vài GB).
    """
    out: dict[str, list[str]] = {}
    for split in SPLITS:
        path = div / f"{branch}_{split}_dataset"
        if not path.is_file():
            print(f"  [FAIL] thiếu {path.name} — không suy ra được split")
            return None
        try:
            ds = _load_pickle(path)
        except Exception as exc:
            print(f"  [FAIL] không đọc được {path.name}: {type(exc).__name__}: {exc}")
            return None
        ids = list(getattr(ds, "list", []))
        if not ids:
            print(f"  [FAIL] {path.name} không có danh sách protein (.list rỗng)")
            return None
        out[split] = ids
        print(f"  {path.name}: {len(ids):,} protein")
        del ds
        gc.collect()

    tr, va, te = (set(out[s]) for s in SPLITS)
    dup = []
    if tr & va:
        dup.append(f"train∩valid={len(tr & va)}")
    if tr & te:
        dup.append(f"train∩test={len(tr & te)}")
    if va & te:
        dup.append(f"valid∩test={len(va & te)}")
    if dup:
        print(f"  [FAIL] divided_data cũ đã bị trùng protein giữa các tập: {', '.join(dup)}")
        return None
    return out


def rebuild_ppi_train_graph(proc: Path, branch: str, split: dict, dry_run: bool) -> bool:
    """ppi_graph_train_{ns} = ppi_graph_global đã cắt cạnh chạm valid/test."""
    g_path = proc / "ppi_graph_global"
    idx_path = proc / "ppi_protein_index"
    out_path = proc / f"ppi_graph_train_{branch}"
    if not g_path.is_file() or not idx_path.is_file():
        print(f"  [FAIL] thiếu {g_path.name} hoặc {idx_path.name}")
        return False

    ppi_graph = _load_pickle(g_path)
    index = _load_pickle(idx_path)
    hidden_acs = set(split["valid"]) | set(split["test"])
    hidden_ids = {index[ac] for ac in hidden_acs if ac in index}
    if not hidden_ids:
        print("  [FAIL] không map được protein valid/test nào sang node PPI — sai ppi_protein_index?")
        return False

    train_graph = build_train_only_ppi_graph(ppi_graph, hidden_ids)
    kept, total = train_graph.num_edges(), ppi_graph.num_edges()
    print(
        f"  ppi_graph_train_{branch}: giữ {kept:,}/{total:,} cạnh ({100.0 * kept / max(total, 1):.1f}%), "
        f"ẩn {len(hidden_ids):,} node valid/test"
    )
    if kept >= total:
        print("  [FAIL] không cạnh nào bị cắt — split hoặc ppi_protein_index không khớp")
        return False
    if not dry_run:
        _save_pickle(train_graph, out_path)
        print(f"  -> đã ghi {out_path}")
    return True


def _build_label_graph(go_terms: list[str], protein_labels: dict):
    """Co-occurrence GO label. Bản sao logic của 3_build_graph_dataset.build_label_graph()
    (file đó không import được vì tên bắt đầu bằng số) — giữ 2 nơi đồng bộ khi sửa."""
    import dgl
    import numpy as np

    n = len(go_terms)
    term_idx = {t: i for i, t in enumerate(go_terms)}
    co_occur = np.zeros((n, n), dtype=np.int32)
    for _pid, terms in protein_labels.items():
        idxs = [term_idx[t] for t in terms if t in term_idx]
        for i in idxs:
            for j in idxs:
                if i != j:
                    co_occur[i, j] = 1
    np.fill_diagonal(co_occur, 0)
    src, dst = np.where(co_occur > 0)
    return dgl.graph((src, dst), num_nodes=n)


def rebuild_label_network(proc: Path, branch: str, split: dict, dry_run: bool) -> bool:
    """label_{ns}_network tính lại CHỈ từ annotation của protein train.

    Giữ nguyên danh sách node (= label_vocab_{ns}.json cũ) để không lệch chiều
    nhãn với emb_label_* / divided_data đang có.
    """
    vocab_path = proc / f"label_vocab_{branch}.json"
    acs_path = proc / ACS_FILES[branch]
    net_path = proc / f"label_{branch}_network"
    if not vocab_path.is_file() or not acs_path.is_file():
        print(f"  [SKIP] thiếu {vocab_path.name} hoặc {acs_path.name} — bỏ qua label_network")
        return True

    with open(vocab_path, "r", encoding="utf-8") as f:
        vocab = json.load(f)
    with open(acs_path, "r", encoding="utf-8") as f:
        acs = json.load(f)

    if net_path.is_file():
        old_net = _load_pickle(net_path)
        if int(old_net.num_nodes()) != len(vocab):
            print(
                f"  [FAIL] label_{branch}_network có {old_net.num_nodes():,} node nhưng "
                f"label_vocab_{branch}.json có {len(vocab):,} term — dữ liệu không đồng bộ. "
                "Chạy scripts/audit_data.py để xem chi tiết; nên rebuild pipeline thay vì migrate."
            )
            return False
        old_edges = int(old_net.num_edges())
    else:
        old_edges = None

    train_ids = set(split["train"])
    train_labels = {pid: terms for pid, terms in acs.items() if pid in train_ids}
    new_net = _build_label_graph(vocab, train_labels)
    msg = (
        f"  label_{branch}_network: {new_net.num_nodes():,} node, {new_net.num_edges():,} cạnh "
        f"(chỉ từ {len(train_labels):,} protein train"
    )
    msg += f"; bản cũ {old_edges:,} cạnh)" if old_edges is not None else ")"
    print(msg)

    if not dry_run:
        if net_path.is_file():
            backup = net_path.with_suffix(net_path.suffix + ".bak")
            if not backup.exists():
                shutil.copy2(net_path, backup)
                print(f"  -> backup bản cũ: {backup.name}")
        _save_pickle(new_net, net_path)
        print(f"  -> đã ghi {net_path}")
    return True


def migrate_branch(proc: Path, div: Path, branch: str, dry_run: bool) -> bool:
    print(f"\n{'=' * 70}\n  NHÁNH {branch.upper()}\n{'=' * 70}")

    split_path = proc / f"split_{branch}.json"
    if split_path.is_file():
        print(f"  [SKIP] đã có {split_path.name} — data này không phải bản cũ, bỏ qua nhánh")
        return True

    print("[1/3] Suy ra split từ divided_data (giữ nguyên phân vùng cũ)…")
    split = derive_split(div, branch)
    if split is None:
        return False
    total = sum(len(split[s]) for s in SPLITS)
    print(
        f"  Tỉ lệ thực tế: {len(split['train']) / total:.3f} / "
        f"{len(split['valid']) / total:.3f} / {len(split['test']) / total:.3f} (train/valid/test)"
    )
    if not dry_run:
        save_split(
            proc, branch, split["train"], split["valid"], split["test"],
            # seed=-1 đánh dấu "suy ra từ divided_data có sẵn", không phải chia
            # lại bằng seed — ratio ghi theo tỉ lệ THỰC TẾ của data cũ.
            seed=-1,
            train_ratio=round(len(split["train"]) / total, 4),
            valid_ratio=round(len(split["valid"]) / total, 4),
        )
        print(f"  -> đã ghi {split_path}")

    print("[2/3] Dựng ppi_graph_train (PPI leakage guard build-time)…")
    if not rebuild_ppi_train_graph(proc, branch, split, dry_run):
        return False

    print("[3/3] Dựng lại label_network chỉ từ train…")
    return rebuild_label_network(proc, branch, split, dry_run)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Nâng cấp data cũ (pipeline main) để chạy với code mới",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", REPO)))
    parser.add_argument("--branch", choices=["mf", "cc", "bp", "all"], default="all")
    parser.add_argument("--dry-run", action="store_true", help="Chỉ in ra sẽ làm gì, không ghi file")
    parser.add_argument(
        "--materialize", action="store_true",
        help="Copy proceed_data (symlink read-only trên Kaggle) thành thư mục ghi được",
    )
    args = parser.parse_args()

    proc = args.data_dir / "proceed_data"
    div = args.data_dir / "divided_data"
    print(f"DATA_DIR: {args.data_dir}   dry_run={args.dry_run}")
    if not proc.is_dir() or not div.is_dir():
        print("[FAIL] Không thấy proceed_data/ hoặc divided_data/ — sai --data-dir?", file=sys.stderr)
        return 1

    if proc.is_symlink() and not args.dry_run:
        if args.materialize:
            materialize_proceed_data(proc)
        else:
            print(
                "[FAIL] proceed_data là symlink (thường là /kaggle/input, chỉ đọc) nên không ghi được.\n"
                "  - Trên Kaggle: chạy lại với --materialize\n"
                "  - Hoặc chạy script này ở máy local rồi pack + upload dataset mới",
                file=sys.stderr,
            )
            return 1

    branches = ["mf", "cc", "bp"] if args.branch == "all" else [args.branch]
    failed = [b for b in branches if not migrate_branch(proc, div, b, args.dry_run)]

    print(f"\n{'=' * 70}")
    if failed:
        print(f"  THẤT BẠI ở nhánh: {', '.join(failed)}", file=sys.stderr)
        print("  Không migrate được -> build lại pipeline từ Bước 2b (README mục 3).")
        return 1
    if args.dry_run:
        print("  DRY-RUN xong — chạy lại không có --dry-run để ghi thật.")
    else:
        print("  XONG. Kiểm tra lại: python scripts/audit_data.py")
        print("  Lưu ý: bộ lọc vocab (min-count) KHÔNG sửa được bằng migrate —")
        print("  muốn có luôn phần đó thì build lại pipeline từ Bước 2b ở máy local.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
