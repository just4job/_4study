"""
3_build_graph_dataset.py
========================
Ghép dữ liệu từ nhiều nguồn thành Graph Dataset hoàn chỉnh cho từng nhánh GO.

Input cần có:
  proceed_data/proteins_edges/{ID}.txt   ← contact map (từ Bước 2)
  proceed_data/protein_node2vec          ← pickle {ID → ndarray (30,)}  (từ node2vec)
  proceed_data/protein_node2onehot       ← pickle {ID → ndarray (L×26)} (từ Bước 4)
  proceed_data/dict_sequence_feature     ← pickle {ID → list (1024,)}   (từ Bước 5)
  proceed_data/human_BP_ACS.json         ← {ID → [GO terms]} cho BP
  proceed_data/human_MF_ACS.json         ← {ID → [GO terms]} cho MF
  proceed_data/human_CC_ACS.json         ← {ID → [GO terms]} cho CC

Output (per branch bp / mf / cc):
  proceed_data/emb_graph_{ns}            ← {ID → dgl.DGLGraph}
  proceed_data/emb_seq_feature_{ns}      ← {ID → torch.FloatTensor (1024,)}
  proceed_data/emb_label_{ns}            ← {ID → torch.FloatTensor (num_labels,)}
  proceed_data/label_vocab_{ns}.json     ← list of GO terms (index → GO term)
  proceed_data/label_{ns}_network        ← dgl.DGLGraph của GO label co-occurrence
"""

import json
import pickle
import warnings
from pathlib import Path

import dgl
import numpy as np
import torch
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ── Cấu hình đường dẫn ────────────────────────────────────────────────────────
BASE_DIR    = Path("D:/CAFA6")
PROC_DIR    = BASE_DIR / "proceed_data"
EDGES_DIR   = PROC_DIR / "proteins_edges"

NODE2VEC_PATH   = PROC_DIR / "protein_node2vec"          # 30-dim PPI-based protein feature
ONEHOT_PATH     = PROC_DIR / "protein_node2onehot"       # 26-dim one-hot per residue
SEQ_FEAT_PATH   = PROC_DIR / "dict_sequence_feature"     # 1024-dim SeqVec
PPI_INDEX_PATH  = PROC_DIR / "ppi_protein_index"         # {UniProtKB_AC → node_id trong PPI graph}

ACS_FILES = {
    "bp": PROC_DIR / "human_BP_ACS.json",
    "mf": PROC_DIR / "human_MF_ACS.json",
    "cc": PROC_DIR / "human_CC_ACS.json",
}

# Chọn loại node feature: "node2vec" (30-dim) hoặc "onehot" (26-dim) hoặc "concat" (56-dim)
NODE_FEAT_MODE = "concat"   # Ghép onehot(26) + PPI node2vec(30) = 56

# ── Tải node features ──────────────────────────────────────────────────────────
print("=" * 60)
print("Đang tải các file feature...")

protein_node2vec    = {}
protein_node2onehot = {}
dict_seq_feature    = {}

if NODE2VEC_PATH.exists():
    with open(NODE2VEC_PATH, "rb") as f:
        protein_node2vec = pickle.load(f)
    print(f"  ✓ node2vec: {len(protein_node2vec):,} protein")
else:
    print("  ✗ Không tìm thấy protein_node2vec — node2vec feature bị bỏ qua")

if ONEHOT_PATH.exists():
    with open(ONEHOT_PATH, "rb") as f:
        protein_node2onehot = pickle.load(f)
    print(f"  ✓ onehot  : {len(protein_node2onehot):,} protein")
else:
    print("  ✗ Không tìm thấy protein_node2onehot — onehot feature bị bỏ qua")

if SEQ_FEAT_PATH.exists():
    with open(SEQ_FEAT_PATH, "rb") as f:
        dict_seq_feature = pickle.load(f)
    print(f"  ✓ SeqVec  : {len(dict_seq_feature):,} protein")
else:
    print("  ✗ Không tìm thấy dict_sequence_feature — sequence feature sẽ là zero vector")

ppi_protein_index: dict[str, int] = {}
if PPI_INDEX_PATH.exists():
    with open(PPI_INDEX_PATH, "rb") as f:
        ppi_protein_index = pickle.load(f)
    print(f"  ✓ PPI index: {len(ppi_protein_index):,} protein")
else:
    print("  ✗ Không tìm thấy ppi_protein_index — chạy 4_build_ppi_graph.py trước")


def build_node_feature(protein_id: str, num_residues: int) -> torch.Tensor:
    """
    Tạo node feature cho từng residue trong protein.
    - node2vec: broadcast vector 30-dim của cả protein cho mọi node
    - onehot  : lấy ma trận L×26 (per-residue)
    - concat  : ghép cả hai thành L×56
    Nếu thiếu feature → dùng zero vector.
    """
    if NODE_FEAT_MODE == "node2vec":
        if protein_id in protein_node2vec:
            feat = np.array(protein_node2vec[protein_id], dtype=np.float32)  # (30,)
            # Broadcast: lặp lại cho mọi node → shape (L, 30)
            feat = np.tile(feat, (num_residues, 1))
        else:
            feat = np.zeros((num_residues, 30), dtype=np.float32)

    elif NODE_FEAT_MODE == "onehot":
        if protein_id in protein_node2onehot:
            feat = np.array(protein_node2onehot[protein_id], dtype=np.float32)  # (L, 26)
            if feat.shape[0] != num_residues:
                # Cắt hoặc pad cho khớp với số node trong đồ thị
                if feat.shape[0] > num_residues:
                    feat = feat[:num_residues]
                else:
                    pad = np.zeros((num_residues - feat.shape[0], 26), dtype=np.float32)
                    feat = np.vstack([feat, pad])
        else:
            feat = np.zeros((num_residues, 26), dtype=np.float32)

    elif NODE_FEAT_MODE == "concat":
        # node2vec (broadcast) + onehot
        nv = np.zeros((num_residues, 30), dtype=np.float32)
        oh = np.zeros((num_residues, 26), dtype=np.float32)
        if protein_id in protein_node2vec:
            v = np.array(protein_node2vec[protein_id], dtype=np.float32)
            nv = np.tile(v, (num_residues, 1))
        if protein_id in protein_node2onehot:
            o = np.array(protein_node2onehot[protein_id], dtype=np.float32)
            l = min(o.shape[0], num_residues)
            oh[:l] = o[:l]
        feat = np.hstack([nv, oh])  # (L, 56)
    else:
        raise ValueError(f"NODE_FEAT_MODE không hợp lệ: {NODE_FEAT_MODE}")

    return torch.from_numpy(feat)


def load_contact_edges(protein_id: str):
    """Đọc file danh sách cạnh, trả về (src, dst) hoặc None nếu lỗi."""
    edge_file = EDGES_DIR / f"{protein_id}.txt"
    if not edge_file.exists():
        return None, None
    try:
        data = np.loadtxt(edge_file, dtype=np.int64)
        if data.ndim == 1:          # chỉ có 1 cạnh
            data = data.reshape(1, 2)
        if len(data) == 0:
            return None, None
        return data[:, 0], data[:, 1]
    except Exception:
        return None, None


def build_label_graph(go_terms: list[str], protein_labels: dict[str, list]) -> dgl.DGLGraph:
    """
    Xây dựng đồ thị co-occurrence của các GO label:
    Có cạnh giữa term_i và term_j nếu chúng cùng xuất hiện trong ≥ 1 protein.
    """
    n = len(go_terms)
    term_idx = {t: i for i, t in enumerate(go_terms)}
    co_occur = np.zeros((n, n), dtype=np.int32)

    for pid, terms in protein_labels.items():
        idxs = [term_idx[t] for t in terms if t in term_idx]
        for i in idxs:
            for j in idxs:
                if i != j:
                    co_occur[i, j] = 1

    np.fill_diagonal(co_occur, 0)
    src, dst = np.where(co_occur > 0)
    g = dgl.graph((src, dst), num_nodes=n)
    return g


# ── Vòng lặp chính: xử lý từng nhánh GO ─────────────────────────────────────
for ns_type, acs_path in ACS_FILES.items():
    print(f"\n{'=' * 60}")
    print(f"  Đang xử lý nhánh GO: {ns_type.upper()}  ({acs_path.name})")
    print(f"{'=' * 60}")

    if not acs_path.exists():
        print(f"  [SKIP] Không tìm thấy {acs_path}")
        continue

    # 1. Đọc GO annotation cho nhánh này
    with open(acs_path, "r", encoding="utf-8") as f:
        protein_labels: dict[str, list] = json.load(f)

    # 2. Xây dựng từ điển GO Term → index (label vocabulary)
    all_go_terms = sorted({t for terms in protein_labels.values() for t in terms})
    term2idx = {t: i for i, t in enumerate(all_go_terms)}
    num_labels = len(all_go_terms)
    print(f"  Số GO label: {num_labels}")
    print(f"  Số protein có annotation: {len(protein_labels):,}")

    # Lưu vocabulary
    vocab_path = PROC_DIR / f"label_vocab_{ns_type}.json"
    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(all_go_terms, f, indent=2)
    print(f"  Lưu vocabulary → {vocab_path.name}")

    # 3. Xây đồ thị label co-occurrence
    print("  Đang xây label network...")
    label_graph = build_label_graph(all_go_terms, protein_labels)
    label_net_path = PROC_DIR / f"label_{ns_type}_network"
    with open(label_net_path, "wb") as f:
        pickle.dump(label_graph, f)
    print(f"  Lưu label network → {label_net_path.name}  ({label_graph.num_nodes()} nodes, {label_graph.num_edges()} edges)")

    # 4. Duyệt từng protein → xây DGL graph + label vector + seq feature + ppi_node_id
    emb_graph       = {}
    emb_seq_feature = {}
    emb_label       = {}
    emb_ppi_node_id = {}   # {protein_id → int node index trong PPI global graph}

    skipped = 0
    for protein_id, go_terms in tqdm(protein_labels.items(), desc=f"  Building {ns_type} graphs"):

        # --- Contact map edges ---
        src_nodes, dst_nodes = load_contact_edges(protein_id)
        if src_nodes is None:
            skipped += 1
            continue

        num_residues = int(max(src_nodes.max(), dst_nodes.max())) + 1

        # --- DGL Graph ---
        g = dgl.graph((src_nodes, dst_nodes), num_nodes=num_residues)

        # --- Node feature ---
        node_feat = build_node_feature(protein_id, num_residues)
        g.ndata["h"] = node_feat
        g.ndata["feature"] = node_feat

        emb_graph[protein_id] = g

        # --- Sequence feature (protein-level) ---
        if protein_id in dict_seq_feature:
            seq_feat = torch.FloatTensor(dict_seq_feature[protein_id])
        else:
            seq_feat = torch.zeros(1024)
        emb_seq_feature[protein_id] = seq_feat

        # --- Multi-hot label vector ---
        label_vec = torch.zeros(num_labels, dtype=torch.float32)
        for t in go_terms:
            if t in term2idx:
                label_vec[term2idx[t]] = 1.0
        emb_label[protein_id] = label_vec

        # --- PPI node id (index trong PPI global graph) ---
        emb_ppi_node_id[protein_id] = ppi_protein_index.get(protein_id, -1)

    print(f"  ✓ {len(emb_graph):,} protein có đủ dữ liệu  |  ✗ {skipped} bị bỏ qua (thiếu contact map)")

    # 5. Lưu ra file pickle
    for obj, name in [
        (emb_graph,        f"emb_graph_{ns_type}"),
        (emb_seq_feature,  f"emb_seq_feature_{ns_type}"),
        (emb_label,        f"emb_label_{ns_type}"),
        (emb_ppi_node_id,  f"emb_ppi_node_id_{ns_type}"),
    ]:
        out_path = PROC_DIR / name
        with open(out_path, "wb") as f:
            pickle.dump(obj, f)
        print(f"  Lưu → {name}")

print("\n" + "=" * 60)
print("Hoàn thành! Kiểm tra thư mục proceed_data/ để xem các file emb_*")
print("Tiếp theo: chạy divide_data.py để chia train/valid/test")
