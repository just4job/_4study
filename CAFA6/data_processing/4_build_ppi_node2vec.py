"""
Build protein-level Node2Vec embeddings from a PPI network.

This script replaces the old notebook workflow and is the recommended way to
generate `proceed_data/protein_node2vec` when the PPI network changes.

Pipeline:
  1. Read a PPI edge list file (STRING format or similar).
  2. Map ENSP ids to UniProt accessions using the mapping CSV.
  3. Keep only proteins with valid structure ids.
  4. Train Node2Vec on the protein-protein graph.
  5. Save `{UniProt_ID: np.ndarray(dim,)}` to a pickle file.
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
from node2vec import Node2Vec


def parse_ensp(token: str) -> str | None:
    """Parse STRING token like '9606.ENSP...' into an ENSP id."""
    if "." not in token:
        return None
    _, ensp = token.split(".", 1)
    if not ensp.startswith("ENSP"):
        return None
    return ensp


def load_mapping(mapping_file: Path) -> dict[str, str]:
    mapping_df = pd.read_csv(mapping_file)
    return dict(zip(mapping_df["Ensembl_Protein"], mapping_df["UniProtKB_AC"]))


def load_valid_ids(valid_ids_file: Path) -> set[str]:
    return set(pd.read_csv(valid_ids_file)["Protein_ID"].tolist())


def build_ppi_graph(ppi_file: Path, ensp2uniprot: dict[str, str], valid_ids: set[str], min_score: int) -> nx.Graph:
    graph = nx.Graph()

    with ppi_file.open("r", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue

            p1_raw, p2_raw = parts[0], parts[1]

            if len(parts) >= 3:
                try:
                    score = int(parts[2])
                except ValueError:
                    score = 0
                if score < min_score:
                    continue

            ensp1 = parse_ensp(p1_raw)
            ensp2 = parse_ensp(p2_raw)
            if not ensp1 or not ensp2:
                continue

            uid1 = ensp2uniprot.get(ensp1)
            uid2 = ensp2uniprot.get(ensp2)
            if not uid1 or not uid2:
                continue

            if uid1 in valid_ids and uid2 in valid_ids:
                graph.add_edge(uid1, uid2)

    return graph


def estimate_precompute_bytes(graph: nx.Graph) -> tuple[int, int]:
    """Ước lượng RAM mà Node2Vec.precompute_probabilities() sẽ chiếm.

    Với p != 1 hoặc q != 1, node2vec phải dùng random walk BẬC 2: với mỗi cạnh
    (v, u) nó lưu 1 vector xác suất dài deg(u). Tổng số float vì thế là
    Σ_v Σ_{u ∈ N(v)} deg(u) = Σ_u deg(u)^2 — tăng theo BÌNH PHƯƠNG bậc, nên
    vài hub PPI bậc ngàn là đủ làm nổ RAM. Cộng thêm overhead dict cho mỗi cặp.

    Trả về (bytes ước lượng, Σ deg^2).
    """
    sum_deg_sq = sum(d * d for _, d in graph.degree())
    overhead = 2 * graph.number_of_edges() * 120  # dict/np object cho mỗi cặp có hướng
    return sum_deg_sq * 8 + overhead, sum_deg_sq


def train_node2vec(
    graph: nx.Graph,
    dimensions: int,
    walk_length: int,
    num_walks: int,
    p: float,
    q: float,
    window: int,
    min_count: int,
    epochs: int,
    workers: int,
    temp_folder: Path | None = None,
):
    kwargs = {}
    if temp_folder is not None:
        # joblib memmap ra đĩa thay vì giữ toàn bộ trong RAM của từng worker.
        temp_folder.mkdir(parents=True, exist_ok=True)
        kwargs["temp_folder"] = str(temp_folder)
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
        **kwargs,
    )
    model = node2vec.fit(window=window, min_count=min_count, epochs=epochs)
    embeddings = {str(node): np.asarray(model.wv[str(node)], dtype=np.float32) for node in graph.nodes()}
    return embeddings


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--ppi-file", type=Path, required=True, help="STRING or other PPI edge list file")
    parser.add_argument("--mapping-file", type=Path, required=True, help="CSV with columns UniProtKB_AC, Ensembl_Protein")
    parser.add_argument("--valid-ids-file", type=Path, required=True, help="CSV with valid structure ids")
    parser.add_argument("--output-file", type=Path, required=True, help="Pickle output file for protein_node2vec")
    parser.add_argument("--dimensions", type=int, default=30)
    parser.add_argument("--walk-length", type=int, default=30)
    parser.add_argument("--num-walks", type=int, default=10)
    parser.add_argument("--p", type=float, default=0.8)
    parser.add_argument("--q", type=float, default=1.2)
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--min-count", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Để 1 trừ khi chắc chắn đủ RAM: joblib NHÂN BẢN bảng xác suất bậc 2 "
             "cho mỗi worker, nên workers=8 tốn gần 8 lần RAM",
    )
    parser.add_argument(
        "--temp-folder", type=Path, default=None,
        help="Thư mục cho joblib memmap ra đĩa (giảm RAM khi --workers > 1)",
    )
    parser.add_argument(
        "--max-memory-gb", type=float, default=8.0,
        help="Dừng trước khi chạy nếu ước lượng RAM vượt ngưỡng này (0 = tắt kiểm tra)",
    )
    parser.add_argument(
        "--min-score", type=int, default=700,
        help="Ngưỡng combined_score. Mặc định 700 để KHỚP với 4_build_ppi_graph.py "
             "(PPI_SCORE_THRESHOLD=700); hạ xuống 400 làm đồ thị dày lên nhiều lần "
             "và RAM tăng theo bình phương bậc",
    )
    args = parser.parse_args()

    start = time.time()
    print("Step 1 - Loading mapping and valid ids ...")
    ensp2uniprot = load_mapping(args.mapping_file)
    valid_ids = load_valid_ids(args.valid_ids_file)
    print(f"  Mapping pairs: {len(ensp2uniprot):,}")
    print(f"  Valid protein ids: {len(valid_ids):,}")

    print("Step 2 - Building PPI graph ...")
    graph = build_ppi_graph(args.ppi_file, ensp2uniprot, valid_ids, args.min_score)
    print(f"  PPI graph: {graph.number_of_nodes():,} nodes, {graph.number_of_edges():,} edges")

    est_bytes, sum_deg_sq = estimate_precompute_bytes(graph)
    degrees = sorted((d for _, d in graph.degree()), reverse=True)
    est_gb = est_bytes / 1024**3 * max(args.workers, 1)
    print(
        f"  Bậc: lớn nhất {degrees[0] if degrees else 0:,}, "
        f"trung bình {2 * graph.number_of_edges() / max(graph.number_of_nodes(), 1):.1f}"
    )
    print(f"  Sum(deg^2) = {sum_deg_sq:,}")
    print(
        f"  RAM ước lượng cho precompute bậc 2: ~{est_bytes / 1024**3:.1f} GB"
        + (f" x {args.workers} worker = ~{est_gb:.1f} GB" if args.workers > 1 else "")
    )
    if args.max_memory_gb > 0 and est_gb > args.max_memory_gb:
        print(
            f"\n[FAIL] Ước lượng ~{est_gb:.1f} GB > --max-memory-gb {args.max_memory_gb}.\n"
            "  Chạy tiếp gần như chắc chắn làm treo máy. Cách giảm, theo thứ tự hiệu quả:\n"
            f"    1. Tăng --min-score (đang {args.min_score}; 700 = ngưỡng 4_build_ppi_graph.py dùng)\n"
            "    2. Đặt --workers 1 (mỗi worker giữ 1 bản sao bảng xác suất)\n"
            "    3. Thêm --temp-folder /duong/dan (joblib memmap ra đĩa)\n"
            "    4. Dùng -p 1 -q 1 (walk bậc 1, tương đương DeepWalk)\n"
            "    5. Nới --max-memory-gb nếu bạn thật sự có đủ RAM",
            file=sys.stderr,
        )
        raise SystemExit(1)
    if args.p == 1.0 and args.q == 1.0:
        print("  (p=q=1 -> random walk bậc 1)")
    if graph.number_of_nodes() == 0:
        raise ValueError("PPI graph is empty. Check the input file, mapping file, and min-score filter.")

    print("Step 3 - Training Node2Vec ...")
    print(f"  dimensions={args.dimensions}, walk_length={args.walk_length}, num_walks={args.num_walks}, workers={args.workers}")
    embeddings = train_node2vec(
        graph,
        dimensions=args.dimensions,
        walk_length=args.walk_length,
        num_walks=args.num_walks,
        p=args.p,
        q=args.q,
        window=args.window,
        min_count=args.min_count,
        epochs=args.epochs,
        workers=args.workers,
        temp_folder=args.temp_folder,
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("wb") as fh:
        pickle.dump(embeddings, fh)

    elapsed = time.time() - start
    print(f"Step 4 - Saved {len(embeddings):,} embeddings to {args.output_file}")
    print(f"  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()