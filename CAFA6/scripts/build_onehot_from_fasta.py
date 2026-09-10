#!/usr/bin/env python3
"""Dựng protein_node2onehot từ seq.fasta thay vì từ ~40 GB PDB AlphaFold.

Vì sao làm được: seq2onehot() trong data_processing/get_sequence.py là hàm THUẦN
trên chuỗi amino acid — không dùng toạ độ 3D. Bản gốc lấy chuỗi từ ATOM record của
file PDB, nhưng chuỗi đó chính là chuỗi UniProt (AlphaFold model đúng chuỗi
UniProt). Nên chỉ cần seq.fasta (~15 MB) là đủ, miễn là chuỗi CĂN ĐÚNG với thứ tự
node trong contact map.

Script KHÔNG tin điều đó mà tự kiểm chứng: với từng protein, so len(chuỗi FASTA)
với số node trong proceed_data/proteins_edges/{ID}.txt (= max chỉ số node + 1,
đúng công thức 3_build_graph_dataset.py dùng), rồi in phân bố chênh lệch. Bạn nhìn
số liệu rồi mới quyết định có ghi hay không.

Ba tình huống chênh lệch:
  len == n_node   Khớp hoàn hảo.
  len >  n_node   FASTA dài hơn — protein dài bị AlphaFold cắt fragment và pipeline
                  chỉ dùng F1. build_node_feature() cắt oh[:n_node], tức lấy đúng
                  phần đầu mà F1 phủ, nên vẫn căn đúng.
  len <  n_node   FASTA NGẮN hơn contact map. Không giải thích được bằng fragment;
                  các residue cuối sẽ nhận vector 0. Đây là dấu hiệu sai ID hoặc
                  sai phiên bản dữ liệu.

Chạy:
    python scripts/build_onehot_from_fasta.py             # chỉ đối chiếu, không ghi
    python scripts/build_onehot_from_fasta.py --write     # ghi protein_node2onehot
    python scripts/build_onehot_from_fasta.py --write --force   # ghi kể cả khi lệch nhiều

Exit code: 0 = căn khớp tốt, 1 = lệch nhiều / thiếu dữ liệu.
"""
from __future__ import annotations

import argparse
import gzip
import os
import pickle
import sys
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np  # noqa: E402

# Dùng lại đúng hàm gốc để bảng vocab 26 ký tự không bao giờ lệch giữa 2 đường.
from data_processing.get_sequence import seq2onehot  # noqa: E402

# Ngưỡng "căn khớp đủ tốt": bao nhiêu phần trăm protein phải có len >= n_node.
MIN_ALIGN_RATE = 0.98


def load_fasta(path: Path) -> dict[str, str]:
    """Parse y hệt 5_build_seq_feature.py: header sp|ID|NAME -> lấy trường giữa."""
    seqs: dict[str, str] = {}
    opener = gzip.open if path.suffix == ".gz" else open
    cur: str | None = None
    buf: list[str] = []
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            if line.startswith(">"):
                if cur is not None:
                    seqs[cur] = "".join(buf)
                parts = line[1:].split()[0].split("|")
                cur = parts[1] if len(parts) >= 2 else parts[0]
                buf = []
            else:
                buf.append(line.strip())
    if cur is not None:
        seqs[cur] = "".join(buf)
    return seqs


def node_count(edge_file: Path) -> int | None:
    """Số node của contact map = max chỉ số + 1 (giống 3_build_graph_dataset.py:285)."""
    try:
        data = np.loadtxt(edge_file, dtype=np.int64)
    except Exception:
        return None
    if data.size == 0:
        return None
    if data.ndim == 1:
        data = data.reshape(1, 2)
    return int(data.max()) + 1


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dựng protein_node2onehot từ FASTA (không cần PDB)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    default_raw = os.environ.get("RAW_DIR") or (
        "D:/raw_data" if os.name == "nt" else str(REPO.parent / "raw_data")
    )
    parser.add_argument("--fasta", type=Path, default=Path(default_raw) / "seq.fasta")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", REPO)))
    parser.add_argument("--write", action="store_true", help="Ghi protein_node2onehot")
    parser.add_argument("--force", action="store_true", help="Ghi kể cả khi căn lệch nhiều")
    args = parser.parse_args()

    proc = args.data_dir / "proceed_data"
    edges_dir = proc / "proteins_edges"
    out_path = proc / "protein_node2onehot"

    if not args.fasta.is_file():
        print(f"[FAIL] không thấy {args.fasta}", file=sys.stderr)
        return 1
    if not edges_dir.is_dir():
        print(f"[FAIL] không thấy {edges_dir} — chạy 2_extract_struct_map.py trước", file=sys.stderr)
        return 1

    print(f"FASTA        : {args.fasta}")
    print(f"contact map  : {edges_dir}")
    seqs = load_fasta(args.fasta)
    print(f"  đọc được {len(seqs):,} chuỗi từ FASTA")

    edge_files = sorted(edges_dir.glob("*.txt"))
    print(f"  {len(edge_files):,} contact map\n")

    exact = shorter_fasta = longer_fasta = 0
    missing: list[str] = []
    unreadable: list[str] = []
    diffs: Counter[int] = Counter()
    onehot: dict[str, np.ndarray] = {}
    bad_char: list[str] = []

    for i, ef in enumerate(edge_files, 1):
        if i % 2000 == 0:
            print(f"  ...{i:,}/{len(edge_files):,}")
        pid = ef.stem
        seq = seqs.get(pid)
        if seq is None:
            missing.append(pid)
            continue
        n = node_count(ef)
        if n is None:
            unreadable.append(pid)
            continue

        d = len(seq) - n
        diffs[d] += 1
        if d == 0:
            exact += 1
        elif d > 0:
            longer_fasta += 1
        else:
            shorter_fasta += 1

        if args.write:
            try:
                onehot[pid] = seq2onehot(seq).astype(np.int8)
            except KeyError as exc:
                # vocab 26 ký tự của seq2onehot không có ký tự này
                bad_char.append(f"{pid}:{exc}")

    total = exact + longer_fasta + shorter_fasta
    print(f"\n{'=' * 70}\n  ĐỐI CHIẾU len(FASTA) vs số node contact map\n{'=' * 70}")
    if not total:
        print("[FAIL] không so được protein nào", file=sys.stderr)
        return 1

    def pct(x: int) -> str:
        return f"{100.0 * x / total:.2f}%"

    print(f"  Khớp hoàn hảo (len == n_node)   : {exact:,}  ({pct(exact)})")
    print(f"  FASTA dài hơn (len >  n_node)   : {longer_fasta:,}  ({pct(longer_fasta)})"
          f"  -> bị cắt oh[:n_node], vẫn căn đúng")
    print(f"  FASTA NGẮN hơn (len <  n_node)  : {shorter_fasta:,}  ({pct(shorter_fasta)})"
          f"  -> residue cuối nhận vector 0")
    if missing:
        print(f"  Không có trong FASTA            : {len(missing):,}  (vd. {missing[:5]})")
    if unreadable:
        print(f"  Contact map không đọc được      : {len(unreadable):,}  (vd. {unreadable[:5]})")

    top = diffs.most_common(8)
    print("\n  Chênh lệch hay gặp nhất (len - n_node: số protein):")
    for d, c in top:
        print(f"    {d:+6d} : {c:,}")

    aligned = exact + longer_fasta
    rate = aligned / total
    print(f"\n  Căn khớp được: {aligned:,}/{total:,} ({100 * rate:.2f}%)")

    ok = rate >= MIN_ALIGN_RATE
    if ok:
        print(f"  => ĐẠT (>= {100 * MIN_ALIGN_RATE:.0f}%). Dựng onehot từ FASTA là hợp lệ, "
              f"không cần tải PDB.")
    else:
        print(f"  => KHÔNG ĐẠT (< {100 * MIN_ALIGN_RATE:.0f}%). Chuỗi FASTA không căn với "
              f"contact map —\n     nhiều khả năng FASTA và bộ PDB khác phiên bản. "
              f"Nên tải PDB rồi chạy get_sequence.py.")

    if not args.write:
        print("\n  (chạy lại với --write để ghi protein_node2onehot)")
        return 0 if ok else 1

    if bad_char:
        print(f"\n[WARN] {len(bad_char):,} protein có ký tự ngoài bảng 26 ký tự của "
              f"seq2onehot, bị bỏ qua (vd. {bad_char[:3]})")
    if not ok and not args.force:
        print("\n[FAIL] Căn lệch quá nhiều — KHÔNG ghi. Thêm --force nếu vẫn muốn.",
              file=sys.stderr)
        return 1
    if not onehot:
        print("\n[FAIL] không dựng được vector nào", file=sys.stderr)
        return 1

    # int8 thay vì int64 mặc định của seq2onehot: nhỏ hơn 8 lần, và
    # 3_build_graph_dataset.py vốn ép về float32 khi dùng nên dtype không ảnh hưởng.
    tmp = out_path.with_name(out_path.name + ".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(onehot, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, out_path)
    size_mb = out_path.stat().st_size / 1e6
    print(f"\n  -> đã ghi {out_path}  ({len(onehot):,} protein, {size_mb:,.0f} MB, dtype int8)")
    print("  Kiểm tra lại: python scripts/check_inputs.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
