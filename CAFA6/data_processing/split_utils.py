"""
split_utils.py
===============
Logic chia train/valid/test THEO PROTEIN ID — dùng chung cho toàn bộ pipeline
(split_protein_ids.py, 4_build_ppi_graph.py, 3_build_graph_dataset.py,
divide_data.py) để đảm bảo 1 protein luôn nằm đúng 1 nhóm xuyên suốt các bước
xử lý dữ liệu, thay vì mỗi script tự random lại (dễ lệch nhau / dễ leak).

File output: proceed_data/split_{ns}.json
    {
      "seed": 42,
      "train_ratio": 0.7,
      "valid_ratio": 0.2,
      "train": ["A0A...", ...],
      "valid": [...],
      "test": [...]
    }
"""
from __future__ import annotations

import json
import random
from pathlib import Path


def split_protein_ids(
    protein_ids, train_ratio: float = 0.7, valid_ratio: float = 0.2, seed: int = 42
):
    """Chia 1 danh sách protein ID thành (train, valid, test).

    Sort trước khi shuffle để kết quả không phụ thuộc thứ tự input (dict/JSON
    insertion order) — chỉ phụ thuộc seed, tái lập được 100%.
    """
    keys = sorted(set(protein_ids))
    random.Random(seed).shuffle(keys)
    total = len(keys)
    train_size = int(total * train_ratio)
    valid_size = int(total * valid_ratio)
    train_keys = keys[:train_size]
    valid_keys = keys[train_size:train_size + valid_size]
    test_keys = keys[train_size + valid_size:]
    return train_keys, valid_keys, test_keys


def split_path(proc_dir: Path, branch: str) -> Path:
    return Path(proc_dir) / f"split_{branch}.json"


def save_split(
    proc_dir: Path,
    branch: str,
    train_keys,
    valid_keys,
    test_keys,
    seed: int,
    train_ratio: float = 0.7,
    valid_ratio: float = 0.2,
) -> Path:
    path = split_path(proc_dir, branch)
    payload = {
        "seed": seed,
        "train_ratio": train_ratio,
        "valid_ratio": valid_ratio,
        "train": sorted(train_keys),
        "valid": sorted(valid_keys),
        "test": sorted(test_keys),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return path


def load_split(proc_dir: Path, branch: str) -> dict | None:
    """Trả về dict {"train": [...], "valid": [...], "test": [...], ...} hoặc
    None nếu chưa chạy split_protein_ids.py cho branch này (pipeline cũ)."""
    path = split_path(proc_dir, branch)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def compute_label_vocab(protein_labels: dict, train_keys, min_count: int) -> list[str]:
    """Lọc GO term theo tần suất — CHỈ đếm trên protein thuộc TRAIN (train_keys),
    không đụng tới valid/test, để tránh rò rỉ thống kê nhãn của valid/test vào
    danh sách nhãn dùng để train + đánh giá (cùng nguyên tắc với label_network).

    Trước đây bộ lọc min-count (2_build_go_namespace.py) bị 3_build_graph_dataset.py
    ghi đè bằng vocab KHÔNG lọc (rebuild từ toàn bộ protein_labels) — hàm này
    thay thế cho cách làm cũ, và tính trên train thay vì toàn bộ dữ liệu.

    protein_labels : {protein_id: [GO_term, ...]} — toàn bộ (train+valid+test OK,
                      hàm tự lọc theo train_keys).
    train_keys      : danh sách protein_id thuộc tập train.
    min_count       : số protein train tối thiểu phải có term đó mới giữ lại.
    """
    train_set = set(train_keys)
    counts: dict[str, int] = {}
    for pid in train_set:
        for term in protein_labels.get(pid, []):
            counts[term] = counts.get(term, 0) + 1
    return sorted(term for term, c in counts.items() if c >= min_count)


def vocab_path(proc_dir: Path, branch: str) -> Path:
    return Path(proc_dir) / f"label_vocab_{branch}.json"


def save_vocab(proc_dir: Path, branch: str, vocab: list[str]) -> Path:
    path = vocab_path(proc_dir, branch)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(vocab), f, indent=2)
    return path


def load_vocab(proc_dir: Path, branch: str) -> list[str] | None:
    path = vocab_path(proc_dir, branch)
    if not path.is_file():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
