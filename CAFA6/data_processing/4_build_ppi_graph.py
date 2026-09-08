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

Node feature của PPI graph:
  - Mặc định: vector zero (1280-dim) — sẽ được cập nhật bởi GNN trong lúc train
  - Nếu có dict_sequence_feature → gắn seq embedding 1280-dim (ESM-2) làm initial node feature
"""

import csv
import json
import pickle
from pathlib import Path

import dgl
import numpy as np
import torch

# ── Cấu hình ─────────────────────────────────────────────────────────────────
BASE_DIR   = Path("D:/CAFA6")
RAW_DIR    = Path("D:/raw_data")
PROC_DIR   = BASE_DIR / "proceed_data"

STRING_FILE   = RAW_DIR / "ppi.txt"
MAPPING_FILE  = PROC_DIR / "uniprot_ensembl_mapping.csv"
SEQ_FEAT_PATH = PROC_DIR / "dict_sequence_feature"

# Dùng bất kỳ ACS file nào để lấy tập protein hợp lệ
ACS_FILE = PROC_DIR / "human_BP_ACS.json"

# Ngưỡng lọc cạnh PPI (STRING combined_score: 0–1000)
# 400=medium, 700=high, 900=very high
PPI_SCORE_THRESHOLD = 700

NODE_FEAT_DIM = 640    # ESM-2 esm2_t30_150M_UR50D output dim

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
node_feat = torch.zeros(num_nodes, NODE_FEAT_DIM, dtype=torch.float32)

if SEQ_FEAT_PATH.exists():
    with open(SEQ_FEAT_PATH, "rb") as f:
        dict_seq_feature: dict = pickle.load(f)
    filled = 0
    for ac, nid in protein_index.items():
        if ac in dict_seq_feature:
            feat = dict_seq_feature[ac]
            node_feat[nid] = torch.FloatTensor(feat)
            filled += 1
    print(f"  Gắn seq feature cho {filled:,} / {num_nodes:,} node")
else:
    print("  Không tìm thấy dict_sequence_feature → dùng zero vector")

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

print("\n" + "=" * 60)
print("Hoàn thành! Tiếp theo: chạy lại 3_build_graph_dataset.py để thêm ppi_node_id")
