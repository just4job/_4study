import gzip
import numpy as np
from pathlib import Path
from Bio.PDB.PDBParser import PDBParser
from scipy.spatial import distance_matrix

def get_contact_map_from_gz(pdb_gz_path, distance_threshold=8.0):
    """
    Đọc file .pdb.gz, trích xuất tọa độ C-alpha và trả về Ma trận kề (Adjacency Matrix).
    """
    parser = PDBParser(QUIET=True) # QUIET=True để tắt các cảnh báo lặt vặt của PDB
    
    # 1. Đọc trực tiếp từ file .gz không cần giải nén ra ổ cứng
    with gzip.open(pdb_gz_path, 'rt') as f:
        structure = parser.get_structure('protein', f)
        
    # 2. Rút trích tọa độ (X, Y, Z) của các nguyên tử Carbon-Alpha (CA)
    ca_coords = []
    
    # Lặp qua các mô hình, chuỗi và gốc axit amin
    for model in structure:
        for chain in model:
            for residue in chain:
                # Bỏ qua các phân tử nước (HOH) hoặc các gốc dị thể, chỉ lấy Axit Amin chuẩn
                if residue.has_id('CA') and residue.id[0] == ' ':
                    coord = residue['CA'].get_coord()
                    ca_coords.append(coord)
                    
    ca_coords = np.array(ca_coords)
    
    # 3. Tính Bản đồ khoảng cách (Distance Matrix)
    # distance_matrix tính khoảng cách Euclide giữa mọi cặp điểm với nhau
    dist_matrix = distance_matrix(ca_coords, ca_coords)
    
    # 4. Chuyển thành Bản đồ tiếp xúc (Contact Map / Adjacency Matrix)
    # Nếu khoảng cách < 8.0 Angstrom thì cho giá trị 1 (có cạnh), ngược lại là 0
    contact_map = (dist_matrix < distance_threshold).astype(int)
    
    # Xóa các cạnh tự nối chính nó (đường chéo chính = 0)
    np.fill_diagonal(contact_map, 0)
    
    return ca_coords, dist_matrix, contact_map

if __name__ == "__main__":
    from tqdm import tqdm

    # ── Cấu hình đường dẫn ──────────────────────────────────────────────────
    STRUCT_DIR = Path("D:/raw_data/struct_feature")
    OUTPUT_DIR = Path("D:/CAFA6/proceed_data/proteins_edges")
    THRESHOLD  = 8.0  # Angstrom

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    pdb_files = sorted(STRUCT_DIR.glob("*.pdb.gz"))
    print(f"Tìm thấy {len(pdb_files)} file .pdb.gz — bắt đầu xử lý...")

    ok_count  = 0
    err_count = 0

    for pdb_path in tqdm(pdb_files, desc="Extracting contact maps"):
        try:
            # Lấy UniProt ID từ tên file: AF-{ID}-F1-model_v6.pdb.gz
            protein_id = pdb_path.name.split("-")[1]
            out_file   = OUTPUT_DIR / f"{protein_id}.txt"

            # Bỏ qua nếu đã xử lý (cho phép chạy lại mà không mất công)
            if out_file.exists():
                ok_count += 1
                continue

            coords, _, contact_map = get_contact_map_from_gz(pdb_path, THRESHOLD)

            if len(coords) < 2:
                print(f"\n[WARN] {pdb_path.name}: quá ít residue ({len(coords)}), bỏ qua.")
                continue

            # Lưu danh sách cạnh: mỗi dòng "node_i node_j"
            edges = np.argwhere(contact_map == 1)
            np.savetxt(out_file, edges, fmt="%d", delimiter=" ")
            ok_count += 1

        except Exception as e:
            err_count += 1
            print(f"\n[ERROR] {pdb_path.name}: {e}")

    print(f"\nHoàn thành! ✓ {ok_count} protein | ✗ {err_count} lỗi")
    print(f"Edge files lưu tại: {OUTPUT_DIR}")