"""
split_protein_ids.py  (Bước 2b — chạy NGAY SAU go_anno.py, TRƯỚC bước 6/7)
===========================================================================
Chia train/valid/test theo protein ID cho từng nhánh GO (bp/mf/cc), TRƯỚC khi
build ppi_graph_global (bước 6) và label_network (bước 7).

Vì sao đặt sớm: pipeline cũ chỉ chia split ở bước 8 (divide_data.py) — nghĩa là
bước 6 (PPI graph) và bước 7 (label co-occurrence network) chạy TRƯỚC đó phải
tính trên TOÀN BỘ protein (không phân biệt train/valid/test), gây rò rỉ gián
tiếp thống kê valid/test vào các artifact được dùng lúc train. Đưa bước chia
split lên đây để bước 6/7 có thể tự lọc ra "chỉ tính từ train" khi cần
(xem README mục 4.5 và mục 3 — Pipeline xử lý dữ liệu).

Input:
  proceed_data/human_BP_ACS.json
  proceed_data/human_MF_ACS.json
  proceed_data/human_CC_ACS.json

Output:
  proceed_data/split_bp.json
  proceed_data/split_mf.json
  proceed_data/split_cc.json
  proceed_data/label_vocab_bp.json / _mf / _cc   ← GO term lọc theo tần suất,
                                                    ĐẾM CHỈ TRÊN PROTEIN TRAIN

Lưu ý: đây là split theo "protein có GO annotation cho nhánh đó" — TẬP LỚN
HƠN so với "protein có đủ dữ liệu cuối cùng" (emb_graph_{ns}, sau khi bước 3/7
lọc bỏ protein thiếu contact map). divide_data.py (bước 8) sẽ tự giao
(intersect) split này với emb_graph_{ns}.keys() để ra dataset cuối cùng — 1
protein bị bỏ vì thiếu contact map thì bỏ luôn ở cả 3 tập, không đổi ý nghĩa
train/valid/test của các protein còn lại.

Bộ lọc tần suất GO term (min-count): trước đây 2_build_go_namespace.py có tính
bộ lọc này (--min-bp/--min-other) nhưng 3_build_graph_dataset.py lại REBUILD
vocab từ đầu (không lọc) rồi ghi đè lên — bộ lọc coi như vô hiệu, khiến cả
những GO term chỉ xuất hiện ở 1 protein cũng thành nhãn phải học (mất cân bằng
cực đoan). Script này tính lại bộ lọc, đúng min-count mặc định
(min_bp=250, min_other=100), nhưng CHỈ ĐẾM TRÊN TẬP TRAIN (không phải toàn bộ
dữ liệu như 2_build_go_namespace.py) — tránh để thống kê tần suất của valid/test
rò rỉ vào việc quyết định "GO term nào được coi là nhãn hợp lệ".
3_build_graph_dataset.py giờ đọc thẳng label_vocab_{ns}.json này làm vocab
chính thức thay vì tự rebuild.

Cảnh báo mất cân bằng do split (chỉ in log, không ghi file, không đổi split):
sau khi chia, script đếm số GO term trong label_vocab_{ns}.json KHÔNG có
positive nào ở valid hoặc test — random split thuần theo protein ID (không
stratify theo nhãn) có thể khiến 1 số label vừa đủ ngưỡng min-count gần như
vắng mặt ở 1 split, làm F1/AUPR cho label đó ở split đó không ổn định/vô nghĩa.
"""

import argparse
import json
import os
from pathlib import Path

if __name__ == "__main__" and __package__ in (None, ""):
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data_processing.split_utils import (
    compute_label_vocab,
    save_split,
    save_vocab,
    split_protein_ids,
    zero_positive_terms,
)


def _resolve_data_dir() -> Path:
    env_data_dir = os.environ.get("DATA_DIR")
    candidates = []
    if env_data_dir:
        candidates.append(Path(env_data_dir))
    candidates.extend([Path(__file__).resolve().parents[1], Path.cwd()])
    for candidate in candidates:
        if (candidate / "proceed_data").is_dir():
            return candidate
    return Path(env_data_dir) if env_data_dir else Path(__file__).resolve().parents[1]


ACS_FILES = {
    "bp": "human_BP_ACS.json",
    "mf": "human_MF_ACS.json",
    "cc": "human_CC_ACS.json",
}


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        "--namespace", type=str, default="all", choices=["all", "bp", "mf", "cc"]
    )
    parser.add_argument("--seed", type=int, default=42, help="Phải khớp seed dùng ở divide_data.py")
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--valid-ratio", type=float, default=0.2)
    parser.add_argument(
        "--min-bp", type=int, default=250,
        help="Số protein TRAIN tối thiểu để giữ 1 GO term nhánh BP (khớp mặc định 2_build_go_namespace.py)",
    )
    parser.add_argument(
        "--min-other", type=int, default=100,
        help="Số protein TRAIN tối thiểu để giữ 1 GO term nhánh MF/CC",
    )
    parser.add_argument("--force", action="store_true", help="Ghi đè split_{ns}.json đã tồn tại")
    args = parser.parse_args()

    data_dir = _resolve_data_dir()
    proc_dir = data_dir / "proceed_data"

    ns_list = ["bp", "mf", "cc"] if args.namespace == "all" else [args.namespace]
    for ns in ns_list:
        acs_path = proc_dir / ACS_FILES[ns]
        out_path = proc_dir / f"split_{ns}.json"
        if out_path.exists() and not args.force:
            print(f"skip {ns}: {out_path.name} đã tồn tại (dùng --force để ghi đè)")
            continue
        if not acs_path.exists():
            print(f"[SKIP] {ns}: không tìm thấy {acs_path} — chạy go_anno.py trước")
            continue

        with open(acs_path, "r", encoding="utf-8") as f:
            protein_labels: dict = json.load(f)

        train_keys, valid_keys, test_keys = split_protein_ids(
            protein_labels.keys(),
            train_ratio=args.train_ratio,
            valid_ratio=args.valid_ratio,
            seed=args.seed,
        )
        path = save_split(
            proc_dir, ns, train_keys, valid_keys, test_keys,
            seed=args.seed, train_ratio=args.train_ratio, valid_ratio=args.valid_ratio,
        )
        print(
            f"{ns}: {len(train_keys):,} train / {len(valid_keys):,} valid / "
            f"{len(test_keys):,} test  (seed={args.seed}) -> {path}"
        )

        min_count = args.min_bp if ns == "bp" else args.min_other
        all_terms_unfiltered = len({t for terms in protein_labels.values() for t in terms})
        vocab = compute_label_vocab(protein_labels, train_keys, min_count)
        vocab_out = save_vocab(proc_dir, ns, vocab)
        print(
            f"{ns}: label_vocab = {len(vocab):,} GO term "
            f"(min_count={min_count}, chỉ đếm trên {len(train_keys):,} protein train; "
            f"trước khi lọc: {all_terms_unfiltered:,} term) -> {vocab_out}"
        )

        # Cảnh báo (không đổi split): random split thuần theo protein ID không
        # stratify theo nhãn — 1 số label vừa đủ ngưỡng min-count có thể gần như
        # vắng mặt ở valid/test, khiến F1/AUPR cho label đó ở split đó không ổn
        # định hoặc vô nghĩa. Xem README mục 3 (Bước 2b) / mục đề xuất cải tiến.
        zero_valid = zero_positive_terms(protein_labels, valid_keys, vocab)
        zero_test = zero_positive_terms(protein_labels, test_keys, vocab)
        if zero_valid or zero_test:
            examples = (zero_valid or zero_test)[:5]
            print(
                f"  [WARN] {ns}: {len(zero_valid)}/{len(vocab)} label không có positive "
                f"nào ở valid, {len(zero_test)}/{len(vocab)} label ở test (random split "
                f"không stratify theo nhãn). Ví dụ: {examples}"
            )


if __name__ == "__main__":
    main()
