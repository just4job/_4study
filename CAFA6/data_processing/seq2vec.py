"""
seq2vec.py — ESM-2 sequence embedding
======================================
Thay thế SeqVec (allennlp) bằng ESM-2 của Meta AI.
  - Model: esm2_t33_650M_UR50D  →  embedding 1280-dim
  - Cài đặt: pip install fair-esm

Input : file FASTA (UniProt format hoặc bất kỳ FASTA nào)
Output: dict pickle {UniProtID → ndarray(1280,)}

Cách lấy UniProtID từ FASTA header:
  >sp|P12345|PROT_HUMAN ... → P12345
  >P12345 ...               → P12345
  >P12345|...               → P12345
"""

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

# Redirect torch hub cache to D: drive to avoid filling C: drive (model is ~1.3 GB)
os.environ.setdefault("TORCH_HOME", "D:/torch_cache")


# ── Đọc FASTA ────────────────────────────────────────────────────────────────
def read_fasta(fasta_path: str) -> dict[str, str]:
    """Trả về {protein_id: sequence}."""
    sequences: dict[str, str] = {}
    current_id = None
    with open(fasta_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                header = line[1:]
                # UniProt format: sp|P12345|NAME hoặc tr|P12345|NAME
                parts = header.split("|")
                if len(parts) >= 2 and parts[0] in ("sp", "tr"):
                    current_id = parts[1]
                else:
                    current_id = parts[0].split()[0]
                sequences[current_id] = ""
            elif current_id is not None:
                sequences[current_id] += line
    return sequences


# ── Cấu hình model theo tên ───────────────────────────────────────────────────
ESM2_CONFIGS = {
    # name          : (loader_fn_name,       num_layers, emb_dim, file_size)
    "8M"            : ("esm2_t6_8M_UR50D",    6,   320, "~31 MB  — rất nhanh trên CPU"),
    "35M"           : ("esm2_t12_35M_UR50D",  12,  480, "~140 MB — nhanh trên CPU"),
    "150M"          : ("esm2_t30_150M_UR50D", 30,  640, "~600 MB — trung bình"),
    "650M"          : ("esm2_t33_650M_UR50D", 33, 1280, "~1.3 GB — chất lượng cao, cần GPU"),
    "3B"            : ("esm2_t36_3B_UR50S",   36, 2560, "~5.5 GB — cần GPU mạnh"),
}


# ── Tạo embedding bằng ESM-2 ─────────────────────────────────────────────────
def embed_with_esm2(sequences: dict[str, str], batch_size: int = 8,
                    device: str = "cuda",
                    model_size: str = "650M") -> dict[str, np.ndarray]:
    """
    Encode từng protein thành vector ESM-2 (mean pooling qua chiều dài).
    Protein dài hơn 1022 residue sẽ bị cắt bớt (giới hạn ESM-2).

    model_size: "8M" | "35M" | "150M" | "650M" | "3B"
      - CPU không có GPU → dùng "8M" hoặc "35M" cho tốc độ hợp lý
      - Có GPU          → dùng "650M" để chất lượng cao nhất
    """
    try:
        import esm
    except ImportError:
        raise ImportError("Chưa cài fair-esm. Chạy: pip install fair-esm")

    cfg = ESM2_CONFIGS.get(model_size)
    if cfg is None:
        raise ValueError(f"model_size phải là một trong: {list(ESM2_CONFIGS)}")
    loader_name, num_layers, emb_dim, size_hint = cfg

    print(f"Đang tải model ESM-2 ({loader_name}, {size_hint})...")
    loader = getattr(esm.pretrained, loader_name)
    model, alphabet = loader()
    batch_converter = alphabet.get_batch_converter()
    model = model.eval().to(device)

    MAX_LEN = 1022  # giới hạn context của ESM-2

    items = list(sequences.items())
    emb_dict: dict[str, np.ndarray] = {}

    for i in tqdm(range(0, len(items), batch_size), desc=f"ESM-2 ({loader_name})"):
        batch = items[i: i + batch_size]
        batch_data = [(pid, seq[:MAX_LEN]) for pid, seq in batch]

        try:
            _, _, tokens = batch_converter(batch_data)
            tokens = tokens.to(device)

            with torch.no_grad():
                results = model(tokens, repr_layers=[num_layers], return_contacts=False)

            representations = results["representations"][num_layers]  # (B, L+2, D)

            for j, (pid, seq) in enumerate(batch_data):
                seq_len = min(len(seq), MAX_LEN)
                emb = representations[j, 1: seq_len + 1].mean(0).cpu().numpy()
                emb_dict[pid] = emb

        except RuntimeError as e:
            print(f"\n[WARN] Batch OOM, chuyển single mode: {e}")
            for pid, seq in batch_data:
                try:
                    single_data = [(pid, seq)]
                    _, _, tokens = batch_converter(single_data)
                    tokens = tokens.to(device)
                    with torch.no_grad():
                        results = model(tokens, repr_layers=[num_layers])
                    representations = results["representations"][num_layers]
                    emb = representations[0, 1: len(seq) + 1].mean(0).cpu().numpy()
                    emb_dict[pid] = emb
                except Exception as e2:
                    print(f"[ERROR] Bỏ qua {pid}: {e2}")

    return emb_dict, emb_dim


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("ESM-2 model sizes:")
    for k, (fn, _, dim, hint) in ESM2_CONFIGS.items():
        print(f"  --model {k:4s}  →  {dim}-dim  {hint}")
    print()

    parser = argparse.ArgumentParser(description="ESM-2 sequence embedding")
    parser.add_argument("-i", "--input",  required=True, help="Path to FASTA file")
    parser.add_argument("-o", "--output", required=True, help="Path to output pickle")
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--cpu", action="store_true", help="Force CPU")
    parser.add_argument("--model", default="650M",
                        choices=list(ESM2_CONFIGS),
                        help="ESM-2 model size (default: 650M; dùng 8M/35M nếu chỉ có CPU)")
    args = parser.parse_args()

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    print(f"Device: {device}")
    if device == "cpu" and args.model in ("650M", "3B"):
        print(f"[WARN] Model {args.model} trên CPU sẽ rất chậm (~50h cho 20K seq).")
        print("       Dùng --model 8M hoặc --model 35M để nhanh hơn nhiều.\n")

    print(f"Đọc FASTA: {args.input}")
    sequences = read_fasta(args.input)
    print(f"  {len(sequences):,} sequence")

    emb_dict, emb_dim = embed_with_esm2(
        sequences, batch_size=args.batch_size,
        device=device, model_size=args.model,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(emb_dict, f)

    print(f"\nĐã lưu {len(emb_dict):,} embedding (dim={emb_dim}) → {out_path}")


if __name__ == "__main__":
    main()
