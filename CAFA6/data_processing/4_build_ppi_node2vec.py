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
):
    node2vec = Node2Vec(
        graph,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        p=p,
        q=q,
        workers=workers,
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
    parser.add_argument("--workers", type=int, default=1, help="Keep this at 1 on Windows")
    parser.add_argument("--min-score", type=int, default=400, help="Minimum confidence score used when the PPI file includes a score column")
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
    )

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("wb") as fh:
        pickle.dump(embeddings, fh)

    elapsed = time.time() - start
    print(f"Step 4 - Saved {len(embeddings):,} embeddings to {args.output_file}")
    print(f"  Done in {elapsed:.1f}s")


if __name__ == "__main__":
    main()