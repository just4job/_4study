#!/usr/bin/env python3
"""Nâng cấp DATA CŨ (build theo pipeline trên `main`, chưa có Bước 2b) để chạy
đúng với code mới — KHÔNG cần build lại từ đầu, KHÔNG đụng tới divided_data.

Vì sao làm được: mọi thứ cần thiết đã nằm sẵn trong bản pack cũ —
  - `divided_data/{ns}_{train,valid,test}_dataset`  -> suy ra ĐÚNG split đã dùng
  - `proceed_data/ppi_graph_global` + `ppi_protein_index` -> dựng ppi_graph_train_{ns}
  - `proceed_data/human_{NS}_ACS.json` + `label_vocab_{ns}.json` -> dựng lại
    label_{ns}_network CHỈ từ protein train

Mặc định script tạo/cập nhật 3 thứ cho mỗi nhánh:
  [1] proceed_data/split_{ns}.json        (suy ra từ divided_data — giữ nguyên
                                           phân vùng cũ, KHÔNG chia lại)
  [2] proceed_data/ppi_graph_train_{ns}   (PPI leakage guard build-time)
  [3] proceed_data/label_{ns}_network     (co-occurrence chỉ từ train; bản cũ
                                           được backup thành *.bak)

Tuỳ chọn `--relabel` làm thêm việc thứ 4 — bộ lọc tần suất GO term (vocab
min-count, đếm CHỈ trên train):
  [4] divided_data/{ns}_{train,valid,test}_dataset  (ghi đè `ds.label`)
      + proceed_data/label_vocab_{ns}.json
Vector nhãn chỉ phụ thuộc human_{NS}_ACS.json + vocab nên tính lại được từ chính
bản pack, KHÔNG cần contact map / raw_data. Đổi lại: phải đọc + ghi lại toàn bộ
divided_data, và `num_labels` thay đổi nên checkpoint cũ không dùng lại được —
phải train lại từ đầu (giống khi build lại pipeline).

Cách chạy:
    python scripts/migrate_old_data.py --dry-run     # xem trước, không ghi gì
    python scripts/migrate_old_data.py               # chạy thật (3 việc đầu)
    python scripts/migrate_old_data.py --branch mf
    python scripts/migrate_old_data.py --relabel --dry-run   # xem vocab mới bao nhiêu nhãn
    python scripts/migrate_old_data.py --relabel             # + lọc vocab (train lại từ đầu)
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

from data_processing.split_utils import (  # noqa: E402
    build_train_only_ppi_graph,
    compute_label_vocab,
    load_split,
    load_vocab,
    save_split,
    save_vocab,
    zero_positive_terms,
)

ACS_FILES = {"mf": "human_MF_ACS.json", "cc": "human_CC_ACS.json", "bp": "human_BP_ACS.json"}
SPLITS = ("train", "valid", "test")


def _load_pickle(path: Path):
    import __main__

    from data_processing.divide_data import MyDataSet

    __main__.MyDataSet = MyDataSet
    with open(path, "rb") as f:
        return pickle.load(f)


def _save_pickle(obj, path: Path) -> None:
    """Ghi qua file tạm rồi os.replace — nếu ghi dở (hết đĩa/ngắt giữa chừng)
    thì file cũ vẫn nguyên vẹn. os.replace cũng thay được cả symlink (Kaggle
    trỏ divided_data/* sang /kaggle/input) mà không ghi đè vào input read-only."""
    tmp = path.with_name(path.name + ".tmp_migrate")
    try:
        with open(tmp, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, path)
    finally:
        if tmp.exists():
            tmp.unlink()


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


def relabel_datasets(
    proc: Path, div: Path, branch: str, split: dict, min_count: int, dry_run: bool
) -> bool:
    """Áp bộ lọc tần suất GO term (min-count, đếm CHỈ trên train) lên data cũ —
    bằng cách tính lại vector nhãn của từng protein trong divided_data.

    Vì sao làm được mà không cần contact map: vector nhãn chỉ phụ thuộc
    `human_{NS}_ACS.json` (protein -> danh sách GO term) và vocab. Graph cấu
    trúc / seq_feature / ppi_node_id trong dataset được GIỮ NGUYÊN, chỉ ghi đè
    `ds.label`. Split cũng giữ nguyên (đọc từ `split`), nên không sinh leak mới.

    Đánh đổi: phải đọc + ghi lại toàn bộ divided_data (vài GB/nhánh) và
    `num_labels` thay đổi -> checkpoint cũ KHÔNG dùng lại được, phải train lại
    từ đầu. Cần chỗ trống >= kích thước dataset LỚN NHẤT (ghi từng file một).
    """
    import torch

    acs_path = proc / ACS_FILES[branch]
    if not acs_path.is_file():
        print(f"  [FAIL] thiếu {acs_path.name} — không tính lại nhãn được")
        return False
    with open(acs_path, "r", encoding="utf-8") as f:
        acs = json.load(f)

    old_vocab = load_vocab(proc, branch)
    new_vocab = compute_label_vocab(acs, split["train"], min_count)
    if not new_vocab:
        print(f"  [FAIL] min_count={min_count} lọc sạch toàn bộ nhãn — hạ ngưỡng xuống")
        return False

    old_n = len(old_vocab) if old_vocab is not None else None
    print(
        f"  vocab mới: {len(new_vocab):,} GO term (min_count={min_count}, đếm trên "
        f"{len(split['train']):,} protein train"
        + (f"; bản cũ {old_n:,} term)" if old_n is not None else ")")
    )
    if old_vocab is not None and set(new_vocab) == set(old_vocab):
        print("  [SKIP] vocab không đổi — data cũ đã được lọc đúng ngưỡng này rồi")
        return True

    for name in ("valid", "test"):
        missing = zero_positive_terms(acs, split[name], new_vocab)
        if missing:
            print(
                f"  [WARN] {len(missing):,}/{len(new_vocab):,} nhãn không có positive nào "
                f"trong {name} — F1 của các nhãn đó ở {name} sẽ là 0/undefined"
            )

    term2idx = {t: i for i, t in enumerate(new_vocab)}
    dim = len(new_vocab)
    for name in SPLITS:
        path = div / f"{branch}_{name}_dataset"
        if not path.is_file():
            print(f"  [FAIL] thiếu {path.name}")
            return False
        ds = _load_pickle(path)
        new_label = {}
        pos_total = 0
        for pid in ds.list:
            vec = torch.zeros(dim, dtype=torch.float32)
            for term in acs.get(pid, ()):
                idx = term2idx.get(term)
                if idx is not None:
                    vec[idx] = 1.0
                    pos_total += 1
            new_label[pid] = vec
        n = max(len(ds.list), 1)
        print(
            f"  {path.name}: {len(ds.list):,} protein x {dim:,} nhãn, "
            f"trung bình {pos_total / n:.1f} nhãn dương/protein"
        )
        if not dry_run:
            ds.label = new_label
            _save_pickle(ds, path)
            print(f"  -> đã ghi đè {path.name}")
        del ds, new_label
        gc.collect()

    if not dry_run:
        save_vocab(proc, branch, new_vocab)
        print(f"  -> đã ghi label_vocab_{branch}.json ({dim:,} term)")
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


def rebuild_label_network(
    proc: Path, branch: str, split: dict, dry_run: bool, expect_vocab_change: bool = False
) -> bool:
    """label_{ns}_network tính lại CHỈ từ annotation của protein train.

    Node của đồ thị = label_vocab_{ns}.json ĐANG có trên đĩa, nên gọi hàm này
    SAU relabel_datasets() (nếu chạy --relabel) để dùng vocab mới; không có
    --relabel thì vocab không đổi và chiều nhãn vẫn khớp divided_data cũ.
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
        # expect_vocab_change=True khi vừa chạy --relabel: chiều nhãn ĐÃ đổi có
        # chủ đích nên lệch node là bình thường, không phải dấu hiệu data hỏng.
        if not expect_vocab_change and int(old_net.num_nodes()) != len(vocab):
            print(
                f"  [FAIL] label_{branch}_network có {old_net.num_nodes():,} node nhưng "
                f"label_vocab_{branch}.json có {len(vocab):,} term — dữ liệu không đồng bộ. "
                "Chạy scripts/audit_data.py để xem chi tiết; nên rebuild pipeline thay vì migrate."
            )
            return False
        old_edges = int(old_net.num_edges())
        old_nodes = int(old_net.num_nodes())
        if old_nodes != len(vocab):
            print(f"  (chiều nhãn đổi: {old_nodes:,} -> {len(vocab):,} node do --relabel)")
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


def migrate_branch(
    proc: Path, div: Path, branch: str, dry_run: bool,
    relabel: bool = False, min_count: int = 100,
) -> bool:
    print(f"\n{'=' * 70}\n  NHÁNH {branch.upper()}\n{'=' * 70}")

    steps = 4 if relabel else 3
    split_path = proc / f"split_{branch}.json"
    existing = load_split(proc, branch)
    if existing is not None and not relabel:
        print(f"  [SKIP] đã có {split_path.name} — data này không phải bản cũ, bỏ qua nhánh")
        return True

    if existing is not None:
        # Đã migrate trước đó rồi, giờ chỉ chạy thêm --relabel: DÙNG LẠI split cũ,
        # tuyệt đối không suy ra/chia lại (sẽ lệch với ppi_graph_train đã dựng).
        print(f"[1/{steps}] Đã có {split_path.name} — dùng lại split đó, không chia lại")
        split = {s: list(existing[s]) for s in SPLITS}
        for name in SPLITS:
            print(f"  {name}: {len(split[name]):,} protein")
    else:
        print(f"[1/{steps}] Suy ra split từ divided_data (giữ nguyên phân vùng cũ)…")
        split = derive_split(div, branch)
        if split is None:
            return False
    total = max(sum(len(split[s]) for s in SPLITS), 1)
    print(
        f"  Tỉ lệ thực tế: {len(split['train']) / total:.3f} / "
        f"{len(split['valid']) / total:.3f} / {len(split['test']) / total:.3f} (train/valid/test)"
    )
    if not dry_run and existing is None:
        save_split(
            proc, branch, split["train"], split["valid"], split["test"],
            # seed=-1 đánh dấu "suy ra từ divided_data có sẵn", không phải chia
            # lại bằng seed — ratio ghi theo tỉ lệ THỰC TẾ của data cũ.
            seed=-1,
            train_ratio=round(len(split["train"]) / total, 4),
            valid_ratio=round(len(split["valid"]) / total, 4),
        )
        print(f"  -> đã ghi {split_path}")

    print(f"[2/{steps}] Dựng ppi_graph_train (PPI leakage guard build-time)…")
    if (proc / f"ppi_graph_train_{branch}").is_file() and existing is not None:
        print(f"  [SKIP] đã có ppi_graph_train_{branch} (split không đổi nên vẫn đúng)")
    elif not rebuild_ppi_train_graph(proc, branch, split, dry_run):
        return False

    if relabel:
        print(f"[3/{steps}] Lọc vocab theo tần suất + tính lại nhãn trong divided_data…")
        if not relabel_datasets(proc, div, branch, split, min_count, dry_run):
            return False

    print(f"[{steps}/{steps}] Dựng lại label_network chỉ từ train…")
    return rebuild_label_network(proc, branch, split, dry_run, expect_vocab_change=relabel)


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
    parser.add_argument(
        "--relabel", action="store_true",
        help="GHI ĐÈ divided_data: lọc vocab theo min-count (đếm trên train) rồi tính "
             "lại vector nhãn. num_labels đổi -> PHẢI train lại từ đầu.",
    )
    parser.add_argument("--min-bp", type=int, default=250, help="min-count cho nhánh BP (--relabel)")
    parser.add_argument("--min-other", type=int, default=100, help="min-count cho MF/CC (--relabel)")
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

    if args.relabel and not args.dry_run:
        print(
            "[!] --relabel sẽ GHI ĐÈ divided_data/*_dataset (nhãn mới) và "
            "label_vocab_*.json.\n"
            "    Sau khi chạy, num_labels đổi -> checkpoint cũ KHÔNG dùng lại được."
        )
        # Kiểm tra sớm: fail ở đây rẻ hơn nhiều so với fail sau khi đã load
        # xong 1 dataset vài GB. Trên Kaggle divided_data/* là symlink tới
        # /kaggle/input, nhưng thư mục CHA vẫn ghi được -> os.replace thay
        # symlink bằng file thật, hợp lệ.
        if not os.access(div, os.W_OK):
            print(
                f"[FAIL] {div} không ghi được — --relabel cần ghi đè dataset.\n"
                "  - Trên Kaggle: chạy scripts/kaggle_link_data.py để divided_data nằm "
                "trong /kaggle/working (thư mục thật), đừng symlink cả thư mục.\n"
                "  - Hoặc chạy --relabel ở máy local rồi pack_for_kaggle.py upload lại.",
                file=sys.stderr,
            )
            return 1
        biggest = max(
            (p.stat().st_size for p in div.glob("*_dataset") if p.is_file()), default=0
        )
        free = shutil.disk_usage(div).free
        print(
            f"    Cần trống >= {biggest / 1e9:.1f} GB (file dataset lớn nhất); "
            f"{div} còn {free / 1e9:.1f} GB."
        )
        if biggest and free < biggest * 1.1:
            print(
                "[FAIL] Không đủ chỗ trống để ghi file tạm. Chạy từng nhánh "
                "(--branch mf) hoặc dọn bớt /kaggle/working.",
                file=sys.stderr,
            )
            return 1

    branches = ["mf", "cc", "bp"] if args.branch == "all" else [args.branch]
    failed = [
        b for b in branches
        if not migrate_branch(
            proc, div, b, args.dry_run,
            relabel=args.relabel,
            min_count=args.min_bp if b == "bp" else args.min_other,
        )
    ]

    print(f"\n{'=' * 70}")
    if failed:
        print(f"  THẤT BẠI ở nhánh: {', '.join(failed)}", file=sys.stderr)
        print("  Không migrate được -> build lại pipeline từ Bước 2b (README mục 3).")
        return 1
    if args.dry_run:
        print("  DRY-RUN xong — chạy lại không có --dry-run để ghi thật.")
    elif args.relabel:
        print("  XONG (có --relabel). Kiểm tra lại: python scripts/audit_data.py")
        print("  num_labels đã đổi -> train LẠI TỪ ĐẦU, đừng nạp checkpoint cũ.")
    else:
        print("  XONG. Kiểm tra lại: python scripts/audit_data.py")
        print("  Lưu ý: bộ lọc vocab (min-count) chưa được áp — chạy thêm --relabel")
        print("  (ghi đè divided_data, phải train lại) hoặc build lại pipeline từ Bước 2b.")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
