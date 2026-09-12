"""
4_build_ppi_graph.py
====================
Xây dựng PPI (Protein-Protein Interaction) Global Graph từ STRING database.

Input:
  raw_data/ppi.txt                          ← STRING PPI file (tab/space separated)
  proceed_data/uniprot_ensembl_mapping.csv  ← ENSP → UniProtKB_AC (từ bước uniprot_mapping.py)
  proceed_data/human_BP_ACS.json            ← để lấy tập protein hợp lệ

Output:
  proceed_data/ppi_graph_global             ← dgl.DGLGraph (nodes=protein, edges=PPI)
  proceed_data/ppi_protein_index            ← dict {UniProtKB_AC → node_id}
  proceed_data/ppi_graph_train_{bp,mf,cc}   ← bản đã ẩn cạnh valid/test của từng nhánh
                                               (chỉ tạo nếu đã chạy split_protein_ids.py
                                               trước — xem README mục 4.5 / 3.Bước 2b)

Node feature của PPI graph:
  - Mặc định: vector zero (1280-dim) — sẽ được cập nhật bởi GNN trong lúc train
  - Nếu có dict_sequence_feature → gắn seq embedding làm initial node feature.
    Số chiều được SUY RA từ chính file đó (5_build_seq_feature.py project về
    TARGET_DIM=1024), không hardcode — vì train_Struct2GO2.py bắt buộc
    ppi_feat_dim == seq_dim (nó tự đọc seq_dim từ dataset).

PPI leakage guard (build-time): ppi_graph_global LUÔN chứa toàn bộ protein (train+
valid+test) — bắt buộc, vì valid/test vẫn cần có mặt trong đồ thị để model dùng PPI
thật lúc suy luận. Cái được "chỉ tính từ train" là CẠNH nào PPIEncoder (GraphSAGE)
được thấy lúc train: ppi_graph_train_{ns} là 1 bản sao ppi_graph_global nhưng đã cắt
mọi cạnh chạm tới protein thuộc valid/test của nhánh đó (đọc từ split_{ns}.json, do
split_protein_ids.py sinh ra). Nếu chưa chạy split_protein_ids.py, bước này bị bỏ qua
(in cảnh báo) — train_Struct2GO2.py khi đó sẽ tự mask lúc runtime (chậm hơn nhưng vẫn
đúng, xem README mục 4.5).
"""

import csv
import json
import pickle
import sys
from pathlib import Path

import dgl
import numpy as np
import torch

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from data_processing.paths import resolve_data_dir, resolve_raw_dir
from data_processing.split_utils import build_train_only_ppi_graph, load_split

# ── Cấu hình ─────────────────────────────────────────────────────────────────
BASE_DIR   = resolve_data_dir()
RAW_DIR    = resolve_raw_dir()
PROC_DIR   = BASE_DIR / "proceed_data"

STRING_FILE   = RAW_DIR / "ppi.txt"
MAPPING_FILE  = PROC_DIR / "uniprot_ensembl_mapping.csv"
SEQ_FEAT_PATH = PROC_DIR / "dict_sequence_feature"

# Dùng bất kỳ ACS file nào để lấy tập protein hợp lệ
ACS_FILE = PROC_DIR / "human_BP_ACS.json"

# Ngưỡng lọc cạnh PPI (STRING combined_score: 0–1000)
# 400=medium, 700=high, 900=very high
PPI_SCORE_THRESHOLD = 700

# Chỉ dùng khi KHÔNG có dict_sequence_feature (node feature = zero vector).
# Khi có file đó, số chiều lấy thẳng từ vector đầu tiên trong file — xem Bước 6.
# Mặc định 1024 = TARGET_DIM của 5_build_seq_feature.py, khớp -seq_dim của
# train_Struct2GO2.py.
FALLBACK_NODE_FEAT_DIM = 1024

# ── Bước 1: Lấy tập protein hợp lệ từ GO annotation ─────────────────────────
print("=" * 60)
print("Bước 1 — Đọc tập protein hợp lệ từ GOA...")
valid_proteins: set[str] = set()
for ns in ("bp", "mf", "cc"):
    acs_path = PROC_DIR / f"human_{ns.upper()}_ACS.json"
    if acs_path.exists():
        with open(acs_path, "r", encoding="utf-8") as f:
            valid_proteins.update(json.load(f).keys())
print(f"  Tổng số protein hợp lệ (có GO annotation): {len(valid_proteins):,}")

# ── Bước 2: Đọc file mapping ENSP → UniProtKB ────────────────────────────────
print("\nBước 2 — Đọc ENSP → UniProtKB mapping...")
if not MAPPING_FILE.exists():
    raise FileNotFoundError(
        f"Không tìm thấy {MAPPING_FILE}.\n"
        "Chạy data_processing/3_uniprot_mapping.py trước."
    )

ensp2uniprot: dict[str, str] = {}
with open(MAPPING_FILE, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    for row in reader:
        ac = row["UniProtKB_AC"].strip()
        ensp = row["Ensembl_Protein"].strip()
        if ac and ensp:
            ensp2uniprot[ensp] = ac
print(f"  {len(ensp2uniprot):,} ENSP→UniProt mappings")

# ── Bước 3: Parse STRING file → tập cạnh PPI ─────────────────────────────────
print(f"\nBước 3 — Parse STRING file (ngưỡng score ≥ {PPI_SCORE_THRESHOLD})...")
if not STRING_FILE.exists():
    raise FileNotFoundError(f"Không tìm thấy {STRING_FILE}")

def parse_ensp(token: str) -> str | None:
    """Chuyển '9606.ENSP...' → 'ENSP...'."""
    if "." not in token:
        return None
    _, ensp = token.split(".", 1)
    return ensp if ensp.startswith("ENSP") else None


raw_edges: list[tuple[str, str]] = []
skipped_score = 0
skipped_map   = 0
skipped_valid = 0

with open(STRING_FILE, "r", encoding="utf-8") as f:
    header = next(f, None)  # bỏ qua header
    for line in f:
        parts = line.rstrip().split()
        if len(parts) < 3:
            continue
        score = int(parts[2])
        if score < PPI_SCORE_THRESHOLD:
            skipped_score += 1
            continue

        ensp1 = parse_ensp(parts[0])
        ensp2 = parse_ensp(parts[1])
        if ensp1 is None or ensp2 is None:
            skipped_map += 1
            continue

        ac1 = ensp2uniprot.get(ensp1)
        ac2 = ensp2uniprot.get(ensp2)
        if ac1 is None or ac2 is None:
            skipped_map += 1
            continue

        if ac1 not in valid_proteins or ac2 not in valid_proteins:
            skipped_valid += 1
            continue

        raw_edges.append((ac1, ac2))

print(f"  Cạnh giữ lại : {len(raw_edges):,}")
print(f"  Bỏ (score thấp)  : {skipped_score:,}")
print(f"  Bỏ (không map)   : {skipped_map:,}")
print(f"  Bỏ (ngoài GOA)   : {skipped_valid:,}")

# ── Bước 4: Xây dựng chỉ mục node ────────────────────────────────────────────
print("\nBước 4 — Xây dựng chỉ mục protein node...")
ppi_proteins: set[str] = set()
for ac1, ac2 in raw_edges:
    ppi_proteins.add(ac1)
    ppi_proteins.add(ac2)

# Thêm các protein hợp lệ chưa có cạnh PPI → isolated nodes
isolated = valid_proteins - ppi_proteins
ppi_proteins.update(valid_proteins)

protein_index: dict[str, int] = {ac: i for i, ac in enumerate(sorted(ppi_proteins))}
num_nodes = len(protein_index)
print(f"  Tổng node trong PPI graph : {num_nodes:,}")
print(f"  Protein isolated (không có cạnh PPI): {len(isolated):,}")

# ── Bước 5: Xây dựng DGL graph ───────────────────────────────────────────────
print("\nBước 5 — Xây dựng DGL graph...")
src_ids = [protein_index[ac1] for ac1, _ in raw_edges]
dst_ids = [protein_index[ac2] for _, ac2 in raw_edges]

# Đồ thị vô hướng: thêm cạnh ngược
src_ids = src_ids + dst_ids
dst_ids_full = dst_ids + [protein_index[ac1] for ac1, _ in raw_edges]

ppi_graph = dgl.graph(
    (torch.tensor(src_ids, dtype=torch.long),
     torch.tensor(dst_ids_full, dtype=torch.long)),
    num_nodes=num_nodes,
)
# Loại bỏ self-loop và cạnh trùng
ppi_graph = dgl.to_simple(ppi_graph)
ppi_graph = dgl.remove_self_loop(ppi_graph)
print(f"  Nodes: {ppi_graph.num_nodes():,} | Edges: {ppi_graph.num_edges():,}")

# ── Bước 6: Gắn node features ────────────────────────────────────────────────
print("\nBước 6 — Gắn initial node features...")
if SEQ_FEAT_PATH.exists():
    with open(SEQ_FEAT_PATH, "rb") as f:
        dict_seq_feature: dict = pickle.load(f)
    if not dict_seq_feature:
        raise ValueError(f"{SEQ_FEAT_PATH} rỗng — chạy lại 5_build_seq_feature.py")
    # Suy ra số chiều từ chính dữ liệu. Trước đây hằng số này bị hardcode 640
    # (ESM-2 150M raw dim) trong khi 5_build_seq_feature.py project về 1024 ->
    # gán node_feat[nid] = vector 1024 chiều làm RuntimeError ngay tại vòng lặp dưới.
    first = next(iter(dict_seq_feature.values()))
    node_feat_dim = len(torch.as_tensor(first, dtype=torch.float32).flatten())
    print(f"  Số chiều seq feature đọc từ {SEQ_FEAT_PATH.name}: {node_feat_dim}")
    node_feat = torch.zeros(num_nodes, node_feat_dim, dtype=torch.float32)
    filled = 0
    mismatched: list[str] = []
    for ac, nid in protein_index.items():
        if ac not in dict_seq_feature:
            continue
        vec = torch.as_tensor(dict_seq_feature[ac], dtype=torch.float32).flatten()
        if vec.numel() != node_feat_dim:
            mismatched.append(ac)
            continue
        node_feat[nid] = vec
        filled += 1
    print(f"  Gắn seq feature cho {filled:,} / {num_nodes:,} node")
    if mismatched:
        print(
            f"  [WARN] {len(mismatched):,} protein có seq feature lệch chiều "
            f"(bỏ qua, để zero). Ví dụ: {mismatched[:5]} — dấu hiệu "
            "dict_sequence_feature bị trộn từ 2 lần chạy với ESM model khác nhau."
        )
else:
    node_feat_dim = FALLBACK_NODE_FEAT_DIM
    node_feat = torch.zeros(num_nodes, node_feat_dim, dtype=torch.float32)
    print(
        f"  [WARN] Không tìm thấy dict_sequence_feature → node feature = zero "
        f"({node_feat_dim} chiều). PPIEncoder sẽ không có tín hiệu đầu vào nào "
        "ngoài cấu trúc đồ thị — chạy 5_build_seq_feature.py trước để có PPI thật."
    )

ppi_graph.ndata["feat"] = node_feat

# ── Bước 7: Lưu output ───────────────────────────────────────────────────────
print("\nBước 7 — Lưu output...")
ppi_graph_path  = PROC_DIR / "ppi_graph_global"
ppi_index_path  = PROC_DIR / "ppi_protein_index"

with open(ppi_graph_path, "wb") as f:
    pickle.dump(ppi_graph, f)
print(f"  Lưu PPI graph → {ppi_graph_path}")

with open(ppi_index_path, "wb") as f:
    pickle.dump(protein_index, f)
print(f"  Lưu protein index → {ppi_index_path}")


# ── Bước 8 (build-time PPI leakage guard) ────────────────────────────────────
# Với mỗi branch đã có split_{ns}.json (từ split_protein_ids.py — chạy TRƯỚC
# bước này), sinh sẵn 1 bản ppi_graph đã ẩn cạnh valid/test — để
# train_Struct2GO2.py không phải tự mask lúc runtime (nhanh + nhẹ RAM hơn,
# đặc biệt trên Kaggle vì không cần load {branch}_test_dataset chỉ để lấy
# ppi_node_id). Xem README mục 4.5.
print("\nBước 8 — Build ppi_graph_train_{bp,mf,cc} từ split_{ns}.json (nếu có)...")
for ns in ("bp", "mf", "cc"):
    split = load_split(PROC_DIR, ns)
    if split is None:
        print(f"  [SKIP] {ns}: chưa có split_{ns}.json — chạy split_protein_ids.py trước "
              f"(train_Struct2GO2.py sẽ tự mask lúc runtime thay thế)")
        continue
    hidden_acs = set(split["valid"]) | set(split["test"])
    hidden_ids = {protein_index[ac] for ac in hidden_acs if ac in protein_index}
    train_graph = build_train_only_ppi_graph(ppi_graph, hidden_ids)
    out_path = PROC_DIR / f"ppi_graph_train_{ns}"
    with open(out_path, "wb") as f:
        pickle.dump(train_graph, f)
    print(
        f"  {ns}: giữ {train_graph.num_edges():,}/{ppi_graph.num_edges():,} cạnh, "
        f"ẩn {len(hidden_ids):,} node valid/test → {out_path}"
    )

print("\n" + "=" * 60)
print("Hoàn thành! Tiếp theo: chạy lại 3_build_graph_dataset.py để thêm ppi_node_id")
