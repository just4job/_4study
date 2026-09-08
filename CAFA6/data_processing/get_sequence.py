import gzip
import warnings
import numpy as np
import pandas as pd
from Bio import SeqIO, BiopythonParserWarning
import os
import pickle
from pathlib import Path
from tqdm import tqdm

warnings.filterwarnings("ignore", category=BiopythonParserWarning)

def _load_sequence(filename):
        if filename.endswith('.pdb') or filename.endswith('.pdb.gz'):
            seq = load_predicted_PDB(filename)
        else:
            raise ValueError(f"Unsupported structure format: {filename}")
        S = seq2onehot(seq)

        return S, seq


def load_predicted_PDB(pdbfile):
    # sequence from atom lines
    if pdbfile.endswith('.gz'):
        with gzip.open(pdbfile, 'rt') as f:
            records = SeqIO.parse(f, 'pdb-atom')
            seqs = [str(r.seq) for r in records]
    else:
        records = SeqIO.parse(pdbfile, 'pdb-atom')
        seqs = [str(r.seq) for r in records]
    return seqs[0]

def seq2onehot(seq):
    """Create 26-dim embedding"""
    chars = ['-', 'D', 'G', 'U', 'L', 'N', 'T', 'K', 'H', 'Y', 'W', 'C', 'P',
             'V', 'S', 'O', 'I', 'E', 'F', 'X', 'Q', 'A', 'B', 'Z', 'R', 'M']
    vocab_size = len(chars)
    vocab_embed = dict(zip(chars, range(vocab_size)))

    # Convert vocab to one-hot
    vocab_one_hot = np.zeros((vocab_size, vocab_size), int)
    for _, val in vocab_embed.items():
        vocab_one_hot[val, val] = 1

    embed_x = [vocab_embed[v] for v in seq]
    seqs_x = np.array([vocab_one_hot[j, :] for j in embed_x])

    return seqs_x


BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "proceed_data"

STRUCT_DIR = Path("D:/raw_data/struct_feature")
VALID_IDS_CSV = PROC_DIR / "valid_protein_ids.csv"
OUT_ONEHOT = PROC_DIR / "protein_node2onehot"
OUT_SEQ = PROC_DIR / "protein_sequence"


def load_valid_ids(csv_path: Path) -> set[str]:
    if not csv_path.exists():
        return set()
    df = pd.read_csv(csv_path)
    if "Protein_ID" in df.columns:
        return set(df["Protein_ID"].dropna().astype(str).tolist())
    # Fallback for files without header
    return set(df.iloc[:, 0].dropna().astype(str).tolist())


if __name__ == "__main__":
    if not STRUCT_DIR.exists():
        raise FileNotFoundError(f"Structure folder not found: {STRUCT_DIR}")

    valid_ids = load_valid_ids(VALID_IDS_CSV)

    protein_node2one_hot = {}
    protein_sequence = {}

    file_list = sorted(f for f in os.listdir(STRUCT_DIR)
                       if f.endswith(".pdb") or f.endswith(".pdb.gz"))
    print(f"Tìm thấy {len(file_list)} file PDB — bắt đầu trích xuất sequence...")

    err_count = 0
    for file_name in tqdm(file_list, desc="Extracting sequences"):
        parts = file_name.split("-")
        if len(parts) < 2:
            continue
        protein_id = parts[1]

        if valid_ids and protein_id not in valid_ids:
            continue

        try:
            S, seqres = _load_sequence(str(STRUCT_DIR / file_name))
            protein_node2one_hot[protein_id] = S
            protein_sequence[protein_id] = seqres
        except Exception:
            err_count += 1
            continue

    print(f"Xong! {len(protein_node2one_hot)} protein thành công, {err_count} lỗi bị bỏ qua.")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUT_ONEHOT, 'wb') as f:
        pickle.dump(protein_node2one_hot, f)
    with open(OUT_SEQ, 'wb') as f:
        pickle.dump(protein_sequence, f)

    print(f"Saved onehot features: {len(protein_node2one_hot)} proteins -> {OUT_ONEHOT}")
    print(f"Saved sequences: {len(protein_sequence)} proteins -> {OUT_SEQ}")