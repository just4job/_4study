"""
read_seqvec_features.py
=======================
Đọc ESM-2 embedding pickle (output của seq2vec.py) và tạo dict_sequence_feature.

Input : proceed_data/9606-avg-emb.pkl  ← {UniProtID → ndarray(1280,)} từ seq2vec.py
Output: proceed_data/dict_sequence_feature ← {UniProtID → ndarray(1280,)}

Chỉ giữ lại protein có trong tập GO annotation (valid_proteins từ *_ACS.json).
Protein không có embedding → padding bằng zero vector 1280-dim.
"""
import json
import pickle
from pathlib import Path

import numpy as np

BASE_DIR = Path("D:/CAFA6")
PROC_DIR = BASE_DIR / "proceed_data"

EMB_PATH = PROC_DIR / "9606-avg-emb.pkl"
OUT_PATH = PROC_DIR / "dict_sequence_feature"

EMB_DIM = 640  # ESM-2 esm2_t30_150M_UR50D (auto-detected from pkl below)

# Lấy tập protein hợp lệ từ GO annotation
valid_proteins: set[str] = set()
for ns in ("BP", "MF", "CC"):
    acs_path = PROC_DIR / f"human_{ns}_ACS.json"
    if acs_path.exists():
        with open(acs_path, "r", encoding="utf-8") as f:
            valid_proteins.update(json.load(f).keys())
print(f"Protein hợp lệ (có GO annotation): {len(valid_proteins):,}")

# Đọc ESM-2 embedding
if not EMB_PATH.exists():
    raise FileNotFoundError(
        f"Không tìm thấy {EMB_PATH}.\n"
        "Chạy: python data_processing/seq2vec.py -i D:/raw_data/seq.fasta -o proceed_data/9606-avg-emb.pkl"
    )
with open(EMB_PATH, "rb") as f:
    raw_emb: dict = pickle.load(f)
print(f"ESM-2 embedding: {len(raw_emb):,} protein")

# Auto-detect emb dim từ pkl (linh hoạt giữa các model size 8M/35M/150M/650M)
if raw_emb:
    sample_vec = next(iter(raw_emb.values()))
    detected_dim = int(np.asarray(sample_vec).shape[-1])
    if detected_dim != EMB_DIM:
        print(f"[INFO] Detected emb dim = {detected_dim} (override default {EMB_DIM})")
        EMB_DIM = detected_dim

# Xây dựng dict_sequence_feature — chỉ giữ protein hợp lệ
dict_sequence_feature: dict[str, np.ndarray] = {}
zero_vec = np.zeros(EMB_DIM, dtype=np.float32)
matched = 0
for protein in valid_proteins:
    if protein in raw_emb:
        dict_sequence_feature[protein] = np.asarray(raw_emb[protein], dtype=np.float32)
        matched += 1
    else:
        dict_sequence_feature[protein] = zero_vec.copy()

print(f"Matched: {matched:,} / {len(valid_proteins):,} (còn lại dùng zero vector)")

with open(OUT_PATH, "wb") as f:
    pickle.dump(dict_sequence_feature, f)
print(f"Đã lưu {len(dict_sequence_feature):,} protein → {OUT_PATH}")
