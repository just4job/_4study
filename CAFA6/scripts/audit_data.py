#!/usr/bin/env python3
"""Audit dữ liệu đã chuẩn bị — kiểm tra ĐÚNG những thứ có thể sai sau khi pipeline
đổi (Bước 2b split sớm, vocab lọc min-count, PPI leakage guard, chống threshold-leak).

Chạy:
    python scripts/audit_data.py                      # kiểm tra nhẹ, cả 3 nhánh
    python scripts/audit_data.py --branch mf          # 1 nhánh
    python scripts/audit_data.py --deep               # + load divided_data (nặng, vài GB)
    DATA_DIR=/kaggle/working/CAFA6 python scripts/audit_data.py

Các nhóm kiểm tra:
  [1] Sự tồn tại của artifact pipeline mới (split_/label_vocab_/ppi_graph_train_)
  [2] Tính toàn vẹn của split (3 tập rời nhau, không trùng protein)
  [3] Vocab: đã lọc tần suất chưa, min-count trên train có đúng không
  [4] PPI leakage guard: ppi_graph_train_{ns} THỰC SỰ không còn cạnh chạm valid/test
  [5] Khớp chiều nhãn giữa vocab / label_network / dataset
  [6] Mất cân bằng: số label không có positive nào ở valid/test
  [7] (--deep) divided_data khớp split_{ns}.json, ppi_node_id hợp lệ

Exit code 0 = không có FAIL (WARN vẫn có thể có), 1 = có FAIL.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

ACS_FILES = {"mf": "human_MF_ACS.json", "cc": "human_CC_ACS.json", "bp": "human_BP_ACS.json"}
MIN_COUNT_DEFAULT = {"bp": 250, "mf": 100, "cc": 100}

_STATUS_ORDER = {"FAIL": 0, "WARN": 1, "OK": 2, "SKIP": 3}


class Report:
    def __init__(self) -> None:
        self.rows: list[tuple[str, str, str]] = []

    def add(self, status: str, check: str, detail: str = "") -> None:
        self.rows.append((status, check, detail))
        icon = {"OK": "  OK  ", " WARN ": " WARN ", "WARN": " WARN ", "FAIL": " FAIL ", "SKIP": " SKIP "}[status]
        print(f"[{icon}] {check}" + (f"\n           {detail}" if detail else ""))

    @property
    def n_fail(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == "FAIL")

    @property
    def n_warn(self) -> int:
        return sum(1 for s, _, _ in self.rows if s == "WARN")


def _load_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_pickle(path: Path):
    import __main__

    from data_processing.divide_data import MyDataSet

    __main__.MyDataSet = MyDataSet
    with open(path, "rb") as f:
        return pickle.load(f)


def _try_import_torch():
    try:
        import torch  # noqa: F401

        return True
    except Exception:
        return False


# ── [1] Artifact tồn tại ─────────────────────────────────────────────────────
def check_artifacts(proc: Path, div: Path, branch: str, rep: Report) -> dict:
    """Trả về dict path của các artifact tìm thấy (để các check sau dùng lại)."""
    found = {}
    required = {
        "acs": proc / ACS_FILES[branch],
        "split": proc / f"split_{branch}.json",
        "vocab": proc / f"label_vocab_{branch}.json",
        "label_network": proc / f"label_{branch}_network",
        "ppi_global": proc / "ppi_graph_global",
        "ppi_index": proc / "ppi_protein_index",
        "ppi_train": proc / f"ppi_graph_train_{branch}",
    }
    missing_new = []
    for key, path in required.items():
        if path.is_file():
            found[key] = path
        elif key in ("split", "vocab", "ppi_train"):
            missing_new.append(path.name)
        else:
            rep.add("FAIL", f"[{branch}] thiếu {path.name}", str(path))

    for split in ("train", "valid", "test"):
        p = div / f"{branch}_{split}_dataset"
        if p.is_file():
            found[f"ds_{split}"] = p
        else:
            rep.add("WARN", f"[{branch}] thiếu divided_data/{p.name}",
                    "train/valid bắt buộc để train; test cần cho eval + chống threshold-leak")

    if missing_new:
        rep.add(
            "FAIL",
            f"[{branch}] thiếu artifact của pipeline mới: {', '.join(missing_new)}",
            "Chạy lại: split_protein_ids.py --force -> 4_build_ppi_graph.py -> "
            "3_build_graph_dataset.py -> divide_data.py --force (README mục 3, Bước 2b)",
        )
    else:
        rep.add("OK", f"[{branch}] có đủ artifact pipeline mới (split/vocab/ppi_graph_train)")
    return found


# ── [2] Split toàn vẹn ───────────────────────────────────────────────────────
def check_split(found: dict, branch: str, rep: Report) -> dict | None:
    if "split" not in found:
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra split (thiếu file)")
        return None
    split = _load_json(found["split"])
    tr, va, te = set(split["train"]), set(split["valid"]), set(split["test"])

    overlaps = []
    if tr & va:
        overlaps.append(f"train∩valid={len(tr & va)}")
    if tr & te:
        overlaps.append(f"train∩test={len(tr & te)}")
    if va & te:
        overlaps.append(f"valid∩test={len(va & te)}")
    if overlaps:
        rep.add("FAIL", f"[{branch}] split BỊ TRÙNG protein giữa các tập", ", ".join(overlaps))
    else:
        rep.add(
            "OK",
            f"[{branch}] split rời nhau: {len(tr):,} train / {len(va):,} valid / {len(te):,} test",
            f"seed={split.get('seed')}, tỉ lệ={split.get('train_ratio')}/{split.get('valid_ratio')}",
        )

    if "acs" in found:
        acs = _load_json(found["acs"])
        covered = tr | va | te
        missing = set(acs.keys()) - covered
        if missing:
            rep.add(
                "WARN",
                f"[{branch}] {len(missing):,} protein trong ACS không nằm trong split nào",
                "split cũ hơn dữ liệu ACS? chạy split_protein_ids.py --force để cập nhật",
            )
    return split


# ── [3] Vocab đã lọc đúng chưa ───────────────────────────────────────────────
def check_vocab(found: dict, split: dict | None, branch: str, min_count: int, rep: Report) -> list | None:
    if "vocab" not in found:
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra vocab (thiếu file)")
        return None
    vocab = _load_json(found["vocab"])
    if "acs" not in found or split is None:
        rep.add("WARN", f"[{branch}] vocab={len(vocab):,} label (không đối chiếu được min-count)")
        return vocab

    acs = _load_json(found["acs"])
    all_terms = {t for terms in acs.values() for t in terms}
    train_ids = set(split["train"])
    counts: dict[str, int] = {}
    for pid in train_ids:
        for t in acs.get(pid, []):
            counts[t] = counts.get(t, 0) + 1

    below = [t for t in vocab if counts.get(t, 0) < min_count]
    detail = (
        f"{len(vocab):,}/{len(all_terms):,} term giữ lại (min_count={min_count} trên "
        f"{len(train_ids):,} protein train)"
    )
    if below and len(vocab) >= len(all_terms):
        # Giữ NGUYÊN mọi term, trong đó có term dưới ngưỡng => vocab chưa qua lọc.
        rep.add(
            "FAIL",
            f"[{branch}] vocab CHƯA được lọc tần suất: {len(vocab):,}/{len(all_terms):,} term",
            f"{len(below)} label có count train < {min_count}. Đây là bug cũ "
            "(3_build_graph_dataset.py rebuild vocab không lọc rồi ghi đè). Chạy lại "
            "split_protein_ids.py --force rồi 3_build_graph_dataset.py.",
        )
    elif below:
        rep.add(
            "WARN",
            f"[{branch}] {len(below)} label trong vocab có count train < min_count",
            f"{detail}. Có thể vocab build với --min-bp/--min-other khác giá trị đang kiểm tra, "
            f"hoặc split đã đổi sau khi build vocab. Ví dụ: {below[:5]}",
        )
    else:
        rep.add("OK", f"[{branch}] vocab đã lọc đúng", detail)

    # Sanity: label chỉ có ở valid/test (không có tín hiệu train) phải bị loại
    leaked = [t for t in vocab if counts.get(t, 0) == 0]
    if leaked:
        rep.add(
            "FAIL",
            f"[{branch}] {len(leaked)} label trong vocab KHÔNG có positive nào ở train",
            f"Model không thể học các label này. Ví dụ: {leaked[:5]}",
        )
    return vocab


# ── [4] PPI leakage guard thực sự hoạt động ─────────────────────────────────
def check_ppi_guard(found: dict, split: dict | None, branch: str, rep: Report) -> None:
    if "ppi_train" not in found or "ppi_global" not in found or "ppi_index" not in found:
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra PPI guard (thiếu file)")
        return
    if split is None:
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra PPI guard (thiếu split)")
        return
    if not _try_import_torch():
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra PPI guard (môi trường không có torch/dgl)")
        return

    import torch

    try:
        g_full = _load_pickle(found["ppi_global"])
        g_train = _load_pickle(found["ppi_train"])
        index = _load_pickle(found["ppi_index"])
    except Exception as exc:
        rep.add("FAIL", f"[{branch}] không load được PPI graph", f"{type(exc).__name__}: {exc}")
        return

    if g_full.num_nodes() != g_train.num_nodes():
        rep.add(
            "FAIL",
            f"[{branch}] ppi_graph_train có số node khác ppi_graph_global",
            f"{g_train.num_nodes():,} vs {g_full.num_nodes():,} — node id sẽ lệch, "
            "ppi_node_id trong dataset trỏ sai node",
        )
        return

    if g_train.num_edges() >= g_full.num_edges():
        rep.add(
            "FAIL",
            f"[{branch}] ppi_graph_train KHÔNG bị cắt cạnh nào",
            f"{g_train.num_edges():,}/{g_full.num_edges():,} cạnh — guard không có tác dụng. "
            "Có thể 4_build_ppi_graph.py chạy khi chưa có split_{ns}.json.",
        )
        return

    hidden_acs = set(split["valid"]) | set(split["test"])
    hidden_ids = sorted({index[ac] for ac in hidden_acs if ac in index})
    if not hidden_ids:
        rep.add("WARN", f"[{branch}] không map được protein valid/test nào sang node PPI")
        return

    mask = torch.zeros(g_train.num_nodes(), dtype=torch.bool)
    mask[torch.tensor(hidden_ids, dtype=torch.long)] = True
    src, dst = g_train.edges()
    bad = int((mask[src.cpu()] | mask[dst.cpu()]).sum())

    kept_pct = 100.0 * g_train.num_edges() / max(g_full.num_edges(), 1)
    if bad:
        rep.add(
            "FAIL",
            f"[{branch}] ppi_graph_train VẪN còn {bad:,} cạnh chạm node valid/test",
            "Guard không đúng — build lại 4_build_ppi_graph.py sau khi có split_{ns}.json",
        )
    else:
        rep.add(
            "OK",
            f"[{branch}] PPI guard đúng: 0 cạnh chạm valid/test",
            f"giữ {g_train.num_edges():,}/{g_full.num_edges():,} cạnh ({kept_pct:.1f}%), "
            f"ẩn {len(hidden_ids):,} node",
        )


# ── [5] Khớp chiều nhãn ──────────────────────────────────────────────────────
def check_label_dims(found: dict, vocab: list | None, branch: str, deep: bool, rep: Report) -> None:
    dims: dict[str, int] = {}
    if vocab is not None:
        dims["label_vocab.json"] = len(vocab)

    if "label_network" in found and _try_import_torch():
        try:
            dims["label_network"] = int(_load_pickle(found["label_network"]).num_nodes())
        except Exception as exc:
            rep.add("WARN", f"[{branch}] không đọc được label_network", f"{type(exc).__name__}: {exc}")

    if deep and _try_import_torch():
        for split in ("train", "valid", "test"):
            key = f"ds_{split}"
            if key not in found:
                continue
            try:
                import numpy as np

                ds = _load_pickle(found[key])
                dims[f"{split}_dataset"] = int(np.asarray(ds[0][2]).reshape(-1).shape[0])
            except Exception as exc:
                rep.add("WARN", f"[{branch}] không đọc được {found[key].name}", f"{type(exc).__name__}: {exc}")

    if len(dims) < 2:
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra chiều nhãn (không đủ nguồn đối chiếu)")
        return

    uniq = set(dims.values())
    detail = ", ".join(f"{k}={v}" for k, v in dims.items())
    if len(uniq) == 1:
        rep.add("OK", f"[{branch}] chiều nhãn khớp nhau ({uniq.pop()})", detail)
    else:
        rep.add(
            "FAIL",
            f"[{branch}] CHIỀU NHÃN LỆCH giữa các artifact",
            detail + " — trộn dữ liệu cũ/mới. Build lại từ 3_build_graph_dataset.py trở đi.",
        )


# ── [6] Mất cân bằng: label không có positive ở valid/test ──────────────────
def check_imbalance(found: dict, split: dict | None, vocab: list | None, branch: str, rep: Report) -> None:
    if split is None or vocab is None or "acs" not in found:
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra mất cân bằng (thiếu split/vocab/ACS)")
        return
    acs = _load_json(found["acs"])
    vocab_set = set(vocab)

    def zero_positive(keys) -> list[str]:
        present: set[str] = set()
        for pid in keys:
            for t in acs.get(pid, []):
                if t in vocab_set:
                    present.add(t)
            if len(present) == len(vocab_set):
                break
        return sorted(vocab_set - present)

    zv, zt = zero_positive(split["valid"]), zero_positive(split["test"])
    if zv or zt:
        rep.add(
            "WARN",
            f"[{branch}] {len(zv)}/{len(vocab)} label không có positive ở valid, "
            f"{len(zt)}/{len(vocab)} ở test",
            "F1/AUPR cho các label này ở split đó vô nghĩa. Cân nhắc tăng --min-bp/--min-other "
            f"hoặc stratified split. Ví dụ: {(zv or zt)[:5]}",
        )
    else:
        rep.add("OK", f"[{branch}] mọi label trong vocab đều có positive ở cả valid và test")


# ── [7] (deep) divided_data khớp split ───────────────────────────────────────
def check_divided_data(found: dict, split: dict | None, branch: str, rep: Report) -> None:
    if split is None:
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra divided_data (thiếu split)")
        return
    if not _try_import_torch():
        rep.add("SKIP", f"[{branch}] bỏ qua kiểm tra divided_data (môi trường không có torch/dgl)")
        return
    expected = {s: set(split[s]) for s in ("train", "valid", "test")}
    seen: dict[str, set] = {}

    for s in ("train", "valid", "test"):
        key = f"ds_{s}"
        if key not in found:
            continue
        try:
            ds = _load_pickle(found[key])
        except Exception as exc:
            rep.add("FAIL", f"[{branch}] không load được {found[key].name}", f"{type(exc).__name__}: {exc}")
            continue
        ids = set(getattr(ds, "list", []))
        seen[s] = ids
        stray = ids - expected[s]
        if stray:
            rep.add(
                "FAIL",
                f"[{branch}] {found[key].name} chứa {len(stray):,} protein KHÔNG thuộc nhóm '{s}' trong split",
                f"Ví dụ: {sorted(stray)[:5]} — dataset build từ split khác. Chạy divide_data.py --force.",
            )
        else:
            dropped = len(expected[s]) - len(ids)
            rep.add(
                "OK",
                f"[{branch}] {found[key].name}: {len(ids):,} protein, đúng nhóm '{s}'",
                f"({dropped:,} protein trong split bị loại vì thiếu contact map — bình thường)",
            )

    if len(seen) >= 2:
        pairs = [("train", "valid"), ("train", "test"), ("valid", "test")]
        bad = [f"{a}∩{b}={len(seen[a] & seen[b])}" for a, b in pairs if a in seen and b in seen and seen[a] & seen[b]]
        if bad:
            rep.add("FAIL", f"[{branch}] divided_data BỊ TRÙNG protein giữa các tập", ", ".join(bad))
        else:
            rep.add("OK", f"[{branch}] divided_data: 3 tập rời nhau")


def audit_branch(proc: Path, div: Path, branch: str, min_count: int, deep: bool, rep: Report) -> None:
    print(f"\n{'=' * 70}\n  NHÁNH {branch.upper()}\n{'=' * 70}")
    found = check_artifacts(proc, div, branch, rep)
    split = check_split(found, branch, rep)
    vocab = check_vocab(found, split, branch, min_count, rep)
    check_ppi_guard(found, split, branch, rep)
    check_label_dims(found, vocab, branch, deep, rep)
    check_imbalance(found, split, vocab, branch, rep)
    if deep:
        check_divided_data(found, split, branch, rep)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Audit dữ liệu CAFA6 đã chuẩn bị (split/vocab/PPI guard/chiều nhãn)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", REPO)),
        help="Thư mục chứa proceed_data/ và divided_data/",
    )
    parser.add_argument("--branch", choices=["mf", "cc", "bp", "all"], default="all")
    parser.add_argument("--min-bp", type=int, default=MIN_COUNT_DEFAULT["bp"])
    parser.add_argument("--min-other", type=int, default=MIN_COUNT_DEFAULT["mf"])
    parser.add_argument(
        "--deep", action="store_true",
        help="Load cả divided_data (vài GB) để đối chiếu protein ID + chiều nhãn thực tế",
    )
    args = parser.parse_args()

    proc = args.data_dir / "proceed_data"
    div = args.data_dir / "divided_data"
    print(f"DATA_DIR: {args.data_dir}")
    print(f"proceed_data: {'OK' if proc.is_dir() else 'MISSING'}   "
          f"divided_data: {'OK' if div.is_dir() else 'MISSING'}   "
          f"deep={args.deep}")
    if not proc.is_dir():
        print("\n[FAIL] Không thấy proceed_data — sai --data-dir?", file=sys.stderr)
        return 1

    rep = Report()
    branches = ["mf", "cc", "bp"] if args.branch == "all" else [args.branch]
    for branch in branches:
        min_count = args.min_bp if branch == "bp" else args.min_other
        audit_branch(proc, div, branch, min_count, args.deep, rep)

    print(f"\n{'=' * 70}\n  TỔNG KẾT: {rep.n_fail} FAIL, {rep.n_warn} WARN, "
          f"{len(rep.rows)} kiểm tra\n{'=' * 70}")
    if rep.n_fail:
        print("Có lỗi cần sửa trước khi train — xem các dòng [ FAIL ] ở trên.")
        return 1
    if not args.deep:
        print("Đã qua kiểm tra nhẹ. Chạy thêm `--deep` để đối chiếu cả divided_data.")
    print("Không có FAIL.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
