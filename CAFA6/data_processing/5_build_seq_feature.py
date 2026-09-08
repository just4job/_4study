"""
5_build_seq_feature.py
======================
Tạo dict_sequence_feature: {UniProt_ID → ndarray (1024,)}
bằng mô hình ESM-2 (Facebook AI).

Input:
    D:/raw_data/UP000005640_9606.fasta  ← FASTA file protein người
    D:/CAFA6/proceed_data/valid_protein_ids.csv  ← danh sách ID hợp lệ

Output:
    D:/CAFA6/proceed_data/dict_sequence_feature  ← pickle dict

Đặc điểm:
    - Dùng ESM-2 esm2_t30_150M_UR50D (150M params, dim=640) hoặc
      esm2_t33_650M_UR50D (650M, dim=1280) — xem CONFIG bên dưới
    - Tự động dùng GPU nếu có, CPU nếu không
    - Hỗ trợ CHECKPOINT: nếu bị tắt giữa chừng, chạy lại sẽ tiếp tục
    - Bỏ qua protein có sequence quá dài (>2000 residues) để tránh OOM
    - Output vector 1024-dim (bằng linear projection từ ESM dim)

Yêu cầu:
    pip install fair-esm biopython torch
"""

import gc
import gzip
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from Bio import SeqIO
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
FASTA_PATH   = Path("D:/raw_data/seq.fasta")
VALID_IDS    = Path("D:/CAFA6/proceed_data/valid_protein_ids.csv")
OUTPUT_PATH  = Path("D:/CAFA6/proceed_data/dict_sequence_feature")
CKPT_PATH    = Path("D:/CAFA6/proceed_data/dict_sequence_feature.ckpt")

# Chọn model ESM-2:
#   "esm2_t6_8M_UR50D"     →  dim=320  (nhỏ nhất, nhanh nhất, ~31MB)
#   "esm2_t12_35M_UR50D"   →  dim=480
#   "esm2_t30_150M_UR50D"  →  dim=640  (cân bằng tốc độ & chất lượng)
#   "esm2_t33_650M_UR50D"  →  dim=1280 (tốt nhất, ~2.5GB RAM GPU)
ESM_MODEL_NAME = "esm2_t30_150M_UR50D"

TARGET_DIM   = 1024    # Dimension output cuối cùng (dùng linear layer)
MAX_SEQ_LEN  = 2000    # Bỏ qua sequence dài hơn mức này (tránh OOM)
BATCH_SIZE   = 1       # Mỗi lần encode 1 protein (an toàn cho RAM)
SAVE_EVERY   = 200     # Lưu checkpoint mỗi N protein
# ──────────────────────────────────────────────────────────────────────────────

ESM_LAYER_MAP = {
    "esm2_t6_8M_UR50D":     (6,  320),
    "esm2_t12_35M_UR50D":   (12, 480),
    "esm2_t30_150M_UR50D":  (30, 640),
    "esm2_t33_650M_UR50D":  (33, 1280),
}


def load_fasta(fasta_path: Path) -> dict[str, str]:
    """Đọc FASTA (có thể là .gz), trả về {uniprot_id: sequence}."""
    sequences = {}
    open_fn = gzip.open if str(fasta_path).endswith(".gz") else open
    mode = "rt"
    with open_fn(fasta_path, mode) as fh:
        for record in SeqIO.parse(fh, "fasta"):
            # Header dạng: sp|P12345|GENE_HUMAN ... → lấy trường 1
            parts = record.id.split("|")
            uid = parts[1] if len(parts) >= 2 else parts[0]
            sequences[uid] = str(record.seq)
    return sequences


def load_checkpoint(ckpt_path: Path) -> dict:
    if ckpt_path.exists():
        with open(ckpt_path, "rb") as f:
            ckpt = pickle.load(f)
        print(f"  ✓ Tiếp tục từ checkpoint: {len(ckpt):,} protein đã xử lý")
        return ckpt
    return {}


def save_checkpoint(embeddings: dict, ckpt_path: Path) -> None:
    with open(ckpt_path, "wb") as f:
        pickle.dump(embeddings, f)


def build_projector(esm_dim: int, target_dim: int, device) -> nn.Module:
    """Linear projection từ ESM dim → target_dim (1024)."""
    if esm_dim == target_dim:
        return nn.Identity().to(device)
    proj = nn.Linear(esm_dim, target_dim, bias=False)
    # Khởi tạo bằng PCA-like: orthogonal init để giữ thông tin
    nn.init.orthogonal_(proj.weight)
    proj.eval()
    return proj.to(device)


def main():
    print("=" * 60)
    print("  5_build_seq_feature.py — ESM-2 Sequence Embeddings")
    print("=" * 60)

    # ── Kiểm tra file đầu vào ──────────────────────────────────────────────
    for name, p in [("FASTA", FASTA_PATH), ("VALID_IDS", VALID_IDS)]:
        status = "✓" if p.exists() else "✗ KHÔNG TÌM THẤY"
        print(f"  {status} {name}: {p}")
    if not FASTA_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy FASTA: {FASTA_PATH}")

    # ── Load danh sách ID hợp lệ ──────────────────────────────────────────
    if VALID_IDS.exists():
        import pandas as pd
        valid_set = set(pd.read_csv(VALID_IDS)["Protein_ID"].tolist())
        print(f"\n  Số protein hợp lệ (có .pdb.gz): {len(valid_set):,}")
    else:
        print("\n  [WARN] Không có valid_protein_ids.csv → xử lý tất cả protein trong FASTA")
        valid_set = None

    # ── Đọc FASTA ─────────────────────────────────────────────────────────
    print(f"\nĐọc FASTA: {FASTA_PATH.name} ...")
    all_seqs = load_fasta(FASTA_PATH)
    print(f"  Tổng số protein trong FASTA: {len(all_seqs):,}")

    # Lọc chỉ lấy protein hợp lệ
    if valid_set:
        seqs = {uid: seq for uid, seq in all_seqs.items() if uid in valid_set}
        print(f"  Sau khi lọc theo valid_ids: {len(seqs):,} protein")
    else:
        seqs = all_seqs

    # ── Load checkpoint ────────────────────────────────────────────────────
    embeddings_dict = load_checkpoint(CKPT_PATH)
    remaining = {uid: seq for uid, seq in seqs.items() if uid not in embeddings_dict}
    print(f"\n  Còn lại cần xử lý: {len(remaining):,} protein")

    if not remaining:
        print("\n✅ Tất cả protein đã được xử lý! Lưu output cuối...")
        with open(OUTPUT_PATH, "wb") as f:
            pickle.dump(embeddings_dict, f)
        print(f"✅ Lưu {len(embeddings_dict):,} embeddings → {OUTPUT_PATH}")
        return

    # ── Load ESM-2 model ───────────────────────────────────────────────────
    try:
        import esm as esm_lib
    except ImportError:
        raise ImportError(
            "Chưa cài fair-esm! Chạy: pip install fair-esm"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    print(f"Load model ESM-2: {ESM_MODEL_NAME} ...")
    t0 = time.time()

    # Tải model (tự download nếu chưa có, ~100-2500MB tuỳ model)
    model, alphabet = esm_lib.pretrained.__dict__[ESM_MODEL_NAME]()
    model.eval()
    model = model.to(device)

    num_layers, esm_dim = ESM_LAYER_MAP[ESM_MODEL_NAME]
    print(f"  ✓ ESM-2 dim={esm_dim}, repr_layer={num_layers}  ({time.time()-t0:.1f}s)")

    # Projector để map về TARGET_DIM
    projector = build_projector(esm_dim, TARGET_DIM, device)

    batch_converter = alphabet.get_batch_converter()

    # ── Vòng lặp chính ─────────────────────────────────────────────────────
    print(f"\nBắt đầu encode {len(remaining):,} protein ...")
    skipped_long = 0
    error_count  = 0
    done = 0

    items = list(remaining.items())
    # Sắp xếp theo chiều dài tăng dần để tăng tốc batch processing
    items.sort(key=lambda x: len(x[1]))

    with tqdm(total=len(items), desc="ESM-2 encoding", unit="prot") as pbar:
        for uid, seq in items:

            # Bỏ qua sequence quá dài
            if len(seq) > MAX_SEQ_LEN:
                skipped_long += 1
                pbar.update(1)
                continue

            try:
                # Chuẩn bị batch (1 protein)
                batch_data = [(uid, seq)]
                batch_labels, batch_strs, batch_tokens = batch_converter(batch_data)
                batch_tokens = batch_tokens.to(device)
                batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)

                with torch.no_grad():
                    results = model(batch_tokens, repr_layers=[num_layers])

                token_repr = results["representations"][num_layers]  # (1, L+2, D)

                # Mean pooling qua các residue (bỏ token BOS, EOS)
                tokens_len = batch_lens[0].item()
                embedding = token_repr[0, 1: tokens_len - 1].mean(0)  # (D,)

                # Project về TARGET_DIM
                with torch.no_grad():
                    embedding = projector(embedding)  # (1024,)

                embeddings_dict[uid] = embedding.cpu().numpy().astype(np.float32)
                done += 1

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    torch.cuda.empty_cache()
                    skipped_long += 1
                    tqdm.write(f"  [OOM] Bỏ qua {uid} (len={len(seq)})")
                else:
                    error_count += 1
                    tqdm.write(f"  [ERR] {uid}: {e}")

            finally:
                # Giải phóng VRAM
                if "batch_tokens" in dir():
                    del batch_tokens
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()

            pbar.update(1)

            # Lưu checkpoint định kỳ
            if done > 0 and done % SAVE_EVERY == 0:
                save_checkpoint(embeddings_dict, CKPT_PATH)
                tqdm.write(f"  💾 Checkpoint lưu: {len(embeddings_dict):,} protein")

    # ── Lưu kết quả cuối ──────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"  ✓ Đã encode:   {done:,} protein")
    print(f"  ✗ Bỏ qua OOM/dài: {skipped_long:,} protein")
    print(f"  ✗ Lỗi khác:   {error_count:,} protein")
    print(f"  Tổng trong dict: {len(embeddings_dict):,}")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "wb") as f:
        pickle.dump(embeddings_dict, f)
    print(f"\n✅ Đã lưu → {OUTPUT_PATH}")

    # Xoá checkpoint nếu hoàn thành
    if CKPT_PATH.exists() and error_count == 0 and skipped_long == 0:
        CKPT_PATH.unlink()
        print("  🗑️  Đã xoá checkpoint (hoàn thành)")

    # Quick sanity check
    sample_id = next(iter(embeddings_dict))
    vec = embeddings_dict[sample_id]
    print(f"\n  Kiểm tra nhanh:")
    print(f"    Protein mẫu : {sample_id}")
    print(f"    Shape vector : {vec.shape}  (phải là (1024,))")
    print(f"    Min / Max    : {vec.min():.4f} / {vec.max():.4f}")


if __name__ == "__main__":
    main()
