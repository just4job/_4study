import os
from pathlib import Path
import pandas as pd

BASE_DIR = Path("D:/CAFA6")
STRUCT_DIR = Path("D:/raw_data/struct_feature")
PROCEED_DIR = BASE_DIR / "proceed_data"

PROCEED_DIR.mkdir(parents=True, exist_ok=True)

print("Đang quét thư mục lấy ID...")
valid_ids = set() # Dùng Set để tự động loại bỏ ID trùng lặp
for file_path in STRUCT_DIR.glob("*.pdb.gz"):
    filename = file_path.name
    try:
        protein_id = filename.split('-')[1]
        valid_ids.add(protein_id)
    except IndexError:
        print(f"Bỏ qua file sai định dạng: {filename}")

# 3. Lưu ra file CSV
valid_ids_list = sorted(list(valid_ids))
df = pd.DataFrame(valid_ids_list, columns=["Protein_ID"])

output_file = PROCEED_DIR / "valid_protein_ids.csv"
df.to_csv(output_file, index=False) # Lần này có header để dễ đọc bằng Pandas sau này

print(f"Hoàn thành! Đã trích xuất {len(valid_ids_list)} ID hợp lệ.")
print(f"File lưu tại: {output_file}")