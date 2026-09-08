# Luồng Xử Lý Dữ Liệu Chi Tiết (Data Processing Pipeline)

## Tổng Quan

Dự án **ESM-AlphaFold-GO** dự đoán chức năng protein (Gene Ontology) bằng cách kết hợp **3 nguồn thông tin chính**:

| Nhánh | Dữ liệu | Xử lý | Đầu ra |
|---|---|---|---|
| **Cấu trúc** | Contact map từ AlphaFold2 | GCN + SAGPool (Hierarchical) | Node features |
| **Trình tự** | FASTA → ESM-2 / SeqVec | Average pooling | 640-dim hoặc 1024-dim embedding |
| **Mạng PPI** | STRING database | GraphSAGE + Cross-Attention | Global PPI graph |

**Output cuối:** Xác suất cho từng GO term (3 nhánh: MFO, CCO, BPO)

---

## Dữ Liệu Đầu Vào (Raw Data)

Phải chuẩn bị trước tại `D:\raw_data\`:

```
D:\raw_data\
├── struct_feature\              ← Cấu trúc protein AlphaFold2
│   ├── AF-A0A0A0MRZ7-F1-model_v6.pdb.gz
│   ├── AF-A0A0A0MRZ7-F1-model_v6.cif.gz
│   └── ... (mỗi protein có cặp .pdb.gz + .cif.gz)
│
├── goa_human.gaf.gz             ← GO annotation (UniProt GOA)
├── ppi.txt                      ← PPI network (STRING: protein1 protein2 score)
└── seq.fasta                    ← Trình tự protein (FASTA format, UniProt ID)
```

### Chi tiết các file

**1. Cấu trúc PDB:**
- **Format:** `AF-{UniProtID}-F1-model_v6.pdb.gz`
  - Ví dụ: `AF-A0A0A0MRZ7-F1-model_v6.pdb.gz` → ID là `A0A0A0MRZ7`
- **Lấy ID:** `filename.split('-')[1]`
- **Mục đích:** Trích xuất tọa độ C-alpha → xây contact map

**2. GO Annotation (GAF):**
- **Format:** Gzip-compressed text, tab-separated
- **Cấu trúc dòng:** `... | UniProtID | GO:ID | ...`
  - Cột 2: UniProtKB AC (UniProt ID)
  - Cột 5: GO identifier (GO:0001234, vv.)
- **Ý nghĩa:** Liệt kê tất cả GO terms được gán cho từng protein

**3. PPI Network (STRING):**
- **Format:** Space/tab-separated
- **Cấu trúc:** `9606.ENSP... 9606.ENSP... combined_score`
  - Cột 1 & 2: Ensembl protein ID (dạng `9606.ENSP...`)
  - Cột 3: combined_score (0–1000, cao hơn = tin tưởng hơn)
- **Threshold:** Thường dùng ≥ 700 (high confidence)

**4. Sequence (FASTA):**
- **Format:** FASTA chuẩn
- **Header parsing:** `>sp|P12345|NAME_HUMAN ...` → ID = `P12345`
- **Mục đích:** Mã hóa trình tự protein thành embedding

---

## Cấu Trúc Output Trung Gian (Proceed Data)

Mỗi bước tạo ra dữ liệu trung gian trong `D:\CAFA6\proceed_data\`:

```
proceed_data/
├── valid_protein_ids.csv                ← Bước 1: danh sách protein
├── HUMAN_protein_info.json              ← Bước 2: {ID → [GO terms]}
├── proteins_edges/                      ← Bước 3: contact map per protein
│   ├── A0A0A0MRZ7.txt                   ← Danh sách cạnh (node_i node_j)
│   └── ...
├── protein_node2onehot                  ← Bước 4a: one-hot embedding (pickle)
├── protein_node2vec                     ← Bước 4b: node2vec từ PPI (pickle)
├── dict_sequence_feature                ← Bước 4c: ESM-2/SeqVec embedding (pickle)
├── uniprot_ensembl_mapping.csv          ← Bước 5: ENSP → UniProtKB mapping
├── ppi_graph_global                     ← Bước 6: DGL graph của PPI network
├── ppi_protein_index                    ← Bước 6: {UniProtID → node_id}
├── human_BP_ACS.json                    ← Bước 7: {ID → [GO terms]} cho BP
├── human_MF_ACS.json                    ← Bước 7: {ID → [GO terms]} cho MF
├── human_CC_ACS.json                    ← Bước 7: {ID → [GO terms]} cho CC
├── label_vocab_bp.json                  ← Bước 7: index → GO term mapping
├── label_vocab_mf.json
├── label_vocab_cc.json
├── emb_graph_bp                         ← Bước 7: {ID → DGL graph} cho BP
├── emb_graph_mf
├── emb_graph_cc
├── emb_seq_feature_bp                   ← Bước 7: {ID → embedding vector}
├── emb_seq_feature_mf
├── emb_seq_feature_cc
├── emb_label_bp                         ← Bước 7: {ID → multi-hot vector}
├── emb_label_mf
├── emb_label_cc
├── emb_ppi_node_id_bp                   ← Bước 7: {ID → PPI node index}
├── emb_ppi_node_id_mf
├── emb_ppi_node_id_cc
├── label_bp_network                     ← Bước 7: DGL graph của GO co-occurrence
├── label_mf_network
└── label_cc_network
```

---

## Chi Tiết 8 Bước Xử Lý

### **BƯỚC 1: Trích xuất danh sách protein ID hợp lệ**

**Script:** `data_processing/1_get_valid_ids.py`

**Mục đích:**  
Quét toàn bộ file `*.pdb.gz` trong `D:\raw_data\struct_feature\`, trích xuất UniProt ID từ tên file.

**Thuật toán:**
```python
1. Duyệt tất cả file *.pdb.gz
2. Tên file format: AF-{UniProtID}-F1-model_v6.pdb.gz
3. Split theo '-': parts = filename.split('-')
4. Lấy: protein_id = parts[1]
5. Dùng Set để loại bỏ trùng lặp tự động
6. Sắp xếp lại và lưu CSV
```

**Input:**
- `D:\raw_data\struct_feature\*.pdb.gz` (tất cả file PDB)

**Output:**
- `proceed_data/valid_protein_ids.csv` (1 cột: Protein_ID)

**Kết quả:**
- Số dòng = số protein khác nhau có cấu trúc PDB
- Ví dụ: ~17,000 protein được chọn từ ~20,000 file

**Thời gian:** Vài phút (tùy số file)

---

### **BƯỚC 2: Parse GO Annotation → JSON**

**Script:** `data_processing/go_anno.py`

**Mục đích:**  
Đọc file `.gaf.gz` từ UniProt GOA, trích xuất mapping `UniProtID → [GO terms]`.

**Thuật toán:**
```python
1. Mở file gzip: goa_human.gaf.gz
2. Bỏ qua dòng comment (bắt đầu bằng '!')
3. Đọc từng dòng, split theo tab (\t)
4. Cột 2: UniProtID
5. Cột 5: GO identifier (GO:0001234)
6. Xây dict: {UniProtID: [GO1, GO2, ...]}
7. Lưu JSON (indent=4 để dễ đọc)
```

**Input:**
- `D:\raw_data\goa_human.gaf.gz` (compressed GAF file)

**Output:**
- `proceed_data/HUMAN_protein_info.json`
  ```json
  {
    "P12345": ["GO:0001234", "GO:0005678", ...],
    "Q98765": ["GO:0001111", ...],
    ...
  }
  ```

**Thống kê:**
- Ví dụ: ~18,000 protein × ~4–30 GO terms/protein
- Tổng cộng: ~80,000–200,000 GO annotation entries

**Thời gian:** 5–15 phút (tùy kích thước file)

---

### **BƯỚC 3: Trích xuất Contact Map từ PDB**

**Script:** `data_processing/2_extract_struct_map.py`

**Mục đích:**  
Chuyển đổi file PDB (3D coordinates) → Contact Map (edge list) dựa trên khoảng cách C-alpha.

**Thuật toán:**
```
1. Mở file .pdb.gz (không cần giải nén ra ổ cứng, đọc trực tiếp)
2. Parse cấu trúc protein bằng BioPython.PDBParser
3. Trích xuất tọa độ (X, Y, Z) của tất cả C-alpha (CA) atoms
   - Bỏ qua: nước (HOH), các phân tử lạ, residue không chuẩn
   - Giữ lại: chỉ những AA có CA atom với residue.id[0] == ' '
4. Tính ma trận khoảng cách Euclidean (Distance Matrix)
   - dist = sqrt((x1-x2)² + (y1-y2)² + (z1-z2)²)
5. Tạo Contact Map (Adjacency Matrix)
   - if distance < threshold (8.0 Ångström):
       contact[i][j] = 1
   - else:
       contact[i][j] = 0
6. Xóa đường chéo (không có self-loop)
   - np.fill_diagonal(contact_map, 0)
7. Chuyển thành edge list (danh sách cạnh)
   - edges = np.argwhere(contact_map == 1)
   - Mỗi dòng: "node_i node_j"
8. Lưu thành file text
```

**Input:**
- `D:\raw_data\struct_feature\*.pdb.gz` (tất cả PDB file)

**Output:**
- `proceed_data/proteins_edges/{UniProtID}.txt`
  ```
  0 5
  0 12
  1 3
  2 7
  ...
  ```
  - Mỗi dòng: `node_i node_j` (cạnh giữa 2 residue)
  - Số dòng ≈ số cạnh = số residue cạnh nhau

**Tham số quan trọng:**
- **distance_threshold = 8.0 Ångström** (có thể điều chỉnh)
  - 8.0: contact map khá dày đặc
  - 6.0: sparse hơn (chỉ gần hơn)

**Kết quả:**
- Ví dụ protein 300 residue:
  - Mỗi residue ≈ 3–5 neighbor
  - Tổng cạnh ≈ 600–800
  - File size ≈ 5–10 KB

**Xử lý lỗi:**
- Skip file nếu quá ít residue (< 2)
- Print warning nếu parse PDB thất bại

**Thời gian:** Vài giờ (phụ thuộc số PDB, thường là hàng ngàn file)

---

### **BƯỚC 4A: Trích xuất One-Hot Sequence Feature**

**Script:** `data_processing/get_sequence.py`

**Mục đích:**  
Chuyển đổi PDB structure → amino acid sequence → one-hot encoding (26-dim per residue).

**Thuật toán:**
```
1. Parse file PDB.gz bằng BioPython
2. Trích xuất chuỗi amino acid từ structure
3. Map mỗi AA → one-hot vector 26 chiều
   - Bảng encoding: 20 standard AA + 5 special (N/C-terminus, gap, unknown, etc.)
   - Ví dụ: Alanine (A) → [1, 0, 0, ..., 0]
            Cysteine (C) → [0, 1, 0, ..., 0]
4. Stack tất cả vectors thành ma trận (L × 26)
   - L = số residue
5. Lưu thành pickle dict
```

**Input:**
- `D:\raw_data\struct_feature\*.pdb.gz`

**Output:**
- `proceed_data/protein_node2onehot` (pickle dict)
  - Key: UniProtID
  - Value: numpy array (L, 26) dtype=float32

**Kích thước:**
- Ví dụ: 17,000 protein × (300 residue avg) × 26 float32
- ≈ 50–100 MB dict trong memory

**Thời gian:** Vài giờ

---

### **BƯỚC 4B: ESM-2 Sequence Embedding**

**Script:** `data_processing/seq2vec.py`

**Mục đích:**  
Mã hóa trình tự protein thành embedding vector cao chiều (1280-dim từ ESM-2 hoặc 640-dim từ SeqVec).

**Thuật toán (ESM-2):**
```
1. Đọc file FASTA
   - Parse header: >sp|P12345|NAME ... → ID = P12345
   - Trích xuất sequence
2. Tải pre-trained ESM-2 model (lựa chọn kích thước)
   - 8M   (6 layer): 320-dim, rất nhanh (CPU ok)
   - 35M  (12 layer): 480-dim, nhanh (CPU ok)
   - 150M (30 layer): 640-dim, trung bình (GPU khuyến nghị)
   - 650M (33 layer): 1280-dim, tốt (GPU cần thiết)
   - 3B   (36 layer): 2560-dim, cao cấp (GPU mạnh)
3. Batch processing (batch_size=8 hoặc điều chỉnh tùy GPU memory)
4. Truncate sequence nếu quá 1022 residue (giới hạn ESM-2)
5. Tokenize → tensor
6. Forward pass → get hidden state từ layer được chọn
7. Mean pooling qua chiều dài (chuỗi)
   - output = mean(hidden_states[1:L+1])  # bỏ qua <bos>, <eos>
   - Shape: (emb_dim,)
8. Lưu dict pickle
```

**Thay thế SeqVec:**
- Cũ: AllennLP SeqVec → 1024-dim (chậm)
- Mới: ESM-2 → 640-dim hoặc 1280-dim (nhanh + tốt hơn)

**Input:**
- `D:\raw_data\seq.fasta` (FASTA file)

**Output:**
- `proceed_data/dict_sequence_feature` (pickle dict)
  - Key: UniProtID
  - Value: numpy array (emb_dim,) dtype=float32

**Cấu hình:**
```bash
python data_processing/seq2vec.py \
  -i D:/raw_data/seq.fasta \
  -o D:/CAFA6/proceed_data/dict_sequence_feature \
  --model 650M \        # Hoặc 150M, 35M, 8M
  --batch_size 8        # Giảm xuống 2–4 nếu OOM
```

**Thời gian:** 
- ESM2_650M trên GPU (V100): ~2–4 giờ cho ~17,000 protein
- ESM2_150M trên GPU: ~1 giờ (nhanh hơn, quality tạm được)
- ESM2_35M trên CPU: ~12–24 giờ (khá chậm)

**Handle OOM:**
- Nếu GPU memory không đủ → chuyển batch_size = 1 (automatic fallback)

---

### **BƯỚC 4C: Chuẩn hóa Sequence Feature**

**Script:** `data_processing/read_seqvec_features.py`

**Mục đích:**  
Tải ESM-2 embedding từ file output, chuẩn hóa (standardization), lưu thành pickle dict.

**Thuật toán:**
```
1. Đọc embedding từ bước 4B (thường là 9606-avg-emb.pkl hoặc dict_sequence_feature)
2. Tính mean và std từ tất cả embedding
   - mean = E[X]
   - std = sqrt(E[(X - mean)²])
3. Chuẩn hóa: X_norm = (X - mean) / std
4. Lưu dict pickle chuẩn hóa
```

**Input:**
- Output từ Bước 4B (embedding dict)

**Output:**
- `proceed_data/dict_sequence_feature` (pickle dict, chuẩn hóa)

**Thời gian:** Vài phút

---

### **BƯỚC 5: UniProt ID Mapping (ENSP → UniProtKB)**

**Script:** `data_processing/3_uniprot_mapping.py`

**Mục đích:**  
Chuyển đổi Ensembl Protein ID (ENSP...) từ file PPI → UniProtKB AC (P12345, Q98765, vv.) để match với GO annotation.

**Vấn đề:**
- STRING database dùng **Ensembl protein ID** (`9606.ENSP00000123456`)
- GO annotation dùng **UniProtKB accession** (`P12345`)
- Cần mapping để liên kết 2 nguồn

**Thuật toán:**
```
1. Extract unique ENSP ID từ file STRING
   - Format: 9606.ENSP... 9606.ENSP... score
   - Split: parts = line.split()
   - Parse: ensp1 = parts[0].split('.')[1]  # ENSP00000...
2. Gọi UniProt ID Mapping REST API
   - Input: ENSP ID list
   - From: Ensembl_Protein
   - To: UniProtKB
   - API: https://rest.uniprot.org/idmapping/run
3. Batch submit (batch_size ≤ 500)
   - Ví dụ: submit 500 ENSP → nhận job ID
4. Poll job status
   - Sleep 5 giây, check status
   - Max retries: 60 × 5 = 300 giây = 5 phút
5. Download kết quả (tsv format)
6. Parse: ENSP → UniProtKB_AC mapping
7. Lưu CSV
```

**Input:**
- `D:\raw_data\ppi.txt` (STRING file)

**Output:**
- `proceed_data/uniprot_ensembl_mapping.csv`
  ```
  UniProtKB_AC,Ensembl_Protein
  P12345,ENSP00000123456
  Q98765,ENSP00000234567
  ...
  ```

**Kết quả:**
- Thường cover ~70–90% ENSP IDs
- Một số ENSP không có mapping (ObsoleteEnsembl, vv.)

**Xử lý lỗi:**
- Network error → auto retry 3 lần
- HTTP 429/503 → wait 2–6 giây, retry
- Job timeout → skip + log warning

**Thời gian:** 10–30 phút (tùy tốc độ API)

---

### **BƯỚC 6: Xây Dựng PPI Global Graph**

**Script:** `data_processing/4_build_ppi_graph.py`

**Mục đích:**  
Chuyển đổi STRING PPI network → DGL graph (nodes=protein, edges=interaction).

**Thuật toán:**
```
1. Load dữ liệu:
   - Protein hợp lệ từ GO annotation (bước 2)
   - ENSP → UniProtKB mapping (bước 5)
   - STRING file (raw PPI)

2. Parse STRING file (tab/space separated)
   - Format: 9606.ENSP... 9606.ENSP... combined_score
   - Lọc: combined_score ≥ threshold (mặc định 700)
   - Extract ENSP từ 2 cột đầu

3. Map ENSP → UniProtKB AC
   - ensp1_mapped = ensp2uniprot.get(ensp1)
   - Bỏ qua nếu không có mapping

4. Filter: Chỉ giữ protein có GO annotation
   - Chỉ giữ edge (u, v) nếu u ∈ valid_proteins AND v ∈ valid_proteins
   - Bỏ qua isolated protein

5. Xây DGL graph
   - Node: UniProtKB_AC
   - Edge: PPI interaction (undirected)
   - Node feature: 
     * Default: zero vector (1280-dim)
     * Hoặc: ESM-2 embedding từ dict_sequence_feature nếu có

6. Tạo mapping node_id
   - ppi_protein_index: {UniProtKB_AC → node_id_trong_graph}

7. Lưu:
   - proceed_data/ppi_graph_global (pickle DGL graph)
   - proceed_data/ppi_protein_index (pickle dict)
```

**Input:**
- `D:\raw_data\ppi.txt` (STRING network)
- `proceed_data/uniprot_ensembl_mapping.csv` (ENSP mapping)
- `proceed_data/human_BP_ACS.json` (valid protein set)

**Output:**
- `proceed_data/ppi_graph_global` (DGL graph)
  - Nodes: ~16,000–17,000 protein
  - Edges: ~150,000–300,000 interaction
- `proceed_data/ppi_protein_index` (pickle dict)

**Tham số:**
- **PPI_SCORE_THRESHOLD = 700** (high confidence)
  - 400–700: medium
  - 700–900: high
  - > 900: very high

**Thống kê:**
- Ví dụ: 17,000 node × 200 edge/node avg
- Tổng edge ≈ 3.4M → phù hợp với đồ thị mạnh

**Thời gian:** 5–20 phút

---

### **BƯỚC 7: Xây Dựng Graph Dataset (Ghép tất cả)**

**Script:** `data_processing/3_build_graph_dataset.py`

**Mục đích:**  
Ghép dữ liệu từ 6 bước trước thành Dataset hoàn chỉnh (3 namespace: BP, MF, CC).

**Luồng xử lý chi tiết:**

#### **7.1 Load tất cả feature**
```
Tải từ pickle:
- protein_node2vec        → 30-dim PPI-based embedding
- protein_node2onehot     → 26-dim one-hot per residue
- dict_sequence_feature   → 1024-dim ESM-2
- ppi_protein_index       → {UniProtID → PPI node_id}
```

#### **7.2 Với mỗi namespace (BP, MF, CC):**

**a) Đọc GO annotation:**
```
- Load human_{NS}_ACS.json
- Format: {UniProtID: [GO:001, GO:002, ...]}
```

**b) Xây vocabulary:**
```
- Lấy tất cả GO term từ tất cả protein
- Sort + index: {GO_term: index}
- Lưu: label_vocab_{ns}.json
```

**c) Build contact map graph (per protein):**
```
Cho mỗi protein:
1. Load edge list từ proteins_edges/{ID}.txt
   - Format: node_i node_j (mỗi dòng)
   - Mỗi node = 1 residue

2. Tạo DGL graph từ edge list
   - nodes: 0, 1, ..., L-1 (L = số residue)
   - edges: từ file

3. Xây node feature (3 mode):
   - MODE="node2vec": broadcast 30-dim vector
   - MODE="onehot": per-residue 26-dim
   - MODE="concat": ghép cả 2 → 56-dim
   
   Công thức cho concat:
   nv = broadcast node2vec → (L, 30)
   oh = one-hot matrix → (L, 26)
   feat = hstack([nv, oh]) → (L, 56)

4. Set g.ndata["feature"] = torch.from_numpy(feat)

5. Lưu: emb_graph_{ns}[ID] = graph
```

**d) Extract sequence feature:**
```
- Lấy dict_sequence_feature[ID]
- Convert to torch.FloatTensor
- Lưu: emb_seq_feature_{ns}[ID]
```

**e) Build label (multi-hot vector):**
```
1. Lấy GO term list từ human_{NS}_ACS[ID]
2. Tạo zero vector (num_labels,)
3. Set index = 1 cho từng GO term
   - Ví dụ: label[vocab[GO:001]] = 1
4. Lưu: emb_label_{ns}[ID] = torch.FloatTensor
```

**f) Map PPI node ID:**
```
- Lấy ppi_protein_index[ID] nếu có
- Else: -1 (protein không trong PPI graph)
- Lưu: emb_ppi_node_id_{ns}[ID]
```

**g) Build GO co-occurrence network (optional):**
```
- Xây graph từ GO term co-occurrence
- Node: GO term
- Edge: (GO_i, GO_j) nếu xuất hiện cùng ≥ min_co_occur threshold
- Mục đích: capture semantic relation giữa GO terms
```

#### **7.3 Output per namespace:**
```
emb_graph_{ns}        ← {UniProtID → DGL graph}
emb_seq_feature_{ns}  ← {UniProtID → torch.Tensor (1024,)}
emb_label_{ns}        ← {UniProtID → torch.Tensor (num_labels,)}
emb_ppi_node_id_{ns}  ← {UniProtID → int}
label_vocab_{ns}.json ← [GO_term_0, GO_term_1, ...]
label_{ns}_network    ← DGL graph của GO co-occurrence
```

**Input:**
- Tất cả output từ bước 1–6
- Plus: `proteins_edges/`, `human_*_ACS.json`

**Output:**
- 3 × 5 = 15 file (3 namespace × 5 loại data)

**Xử lý missing data:**
- Nếu thiếu edge file → graph = empty graph
- Nếu thiếu seq feature → embedding = zero vector
- Nếu thiếu node2vec → use onehot only (MODE-dependent)

**Thời gian:** 30–90 phút (phụ thuộc số protein, số GO term)

---

### **BƯỚC 8: Chia Train / Valid / Test**

**Script:** `data_processing/divide_data.py`

**Mục đích:**  
Chia protein dataset thành 3 tập con (train, valid, test) để huấn luyện và đánh giá model.

**Thuật toán:**
```
1. Lấy danh sách protein từ emb_graph_{ns}
   - keys = list(emb_graph.keys())

2. Shuffle theo seed
   - random.Random(seed).shuffle(keys)
   - Seed mặc định: 42 (reproducible)

3. Chia tỷ lệ (mặc định):
   - train_ratio = 0.7 (70%)
   - valid_ratio = 0.2 (20%)
   - test_ratio  = 0.1 (10%)
   
   Tính size:
   - train_size = int(total * 0.7)
   - valid_size = int(total * 0.2)
   - test_size = total - train_size - valid_size

4. Split keys:
   - train_keys = keys[:train_size]
   - valid_keys = keys[train_size : train_size+valid_size]
   - test_keys = keys[train_size+valid_size:]

5. Subset data:
   Cho mỗi split:
   - emb_graph_train = {k: emb_graph[k] for k in train_keys}
   - (tương tự cho seq_feature, label, ppi_node_id)

6. Tạo MyDataSet object:
   train_dataset = MyDataSet(
       emb_graph=emb_graph_train,
       emb_seq_feature=seq_feature_train,
       emb_label=label_train,
       emb_ppi_node_id=ppi_node_id_train
   )

7. Serialize (pickle):
   - Lưu train_dataset → divided_data/{ns}_train_dataset
   - Lưu valid_dataset → divided_data/{ns}_valid_dataset
   - Lưu test_dataset → divided_data/{ns}_test_dataset
```

**MyDataSet class:**
```python
class MyDataSet(Dataset):
    def __init__(self, emb_graph, emb_seq_feature, emb_label, 
                 emb_ppi_node_id=None):
        self.list = list(emb_graph.keys())
        self.graphs = emb_graph
        self.seq_feature = emb_seq_feature
        self.label = emb_label
        self.ppi_node_id = emb_ppi_node_id or {}
    
    def __getitem__(self, idx):
        protein = self.list[idx]
        return (
            protein,
            self.graphs[protein],      # DGL graph
            self.label[protein],       # multi-hot vector
            self.seq_feature[protein], # embedding
            self.ppi_node_id.get(protein, -1)  # node index trong PPI
        )
    
    def __len__(self):
        return len(self.list)
```

**Input:**
- emb_graph_{ns}, emb_seq_feature_{ns}, emb_label_{ns}, emb_ppi_node_id_{ns}

**Output (per namespace):**
- `divided_data/{ns}_train_dataset` (pickle MyDataSet)
- `divided_data/{ns}_valid_dataset`
- `divided_data/{ns}_test_dataset`

**Cấu hình:**
```bash
# Split all 3 namespaces
python data_processing/divide_data.py

# Split chỉ BP (nhanh hơn để test)
python data_processing/divide_data.py --namespace bp

# Custom seed
python data_processing/divide_data.py --seed 123

# Ghi đè split cũ
python data_processing/divide_data.py --force
```

**Thống kê (ví dụ BP):**
- Tổng protein: ~17,000
- Train: ~11,900 (70%)
- Valid: ~3,400 (20%)
- Test: ~1,700 (10%)

**Thời gian:** Vài phút

---

## Tóm Tắt Luồng Tổng Thể

```
D:\raw_data\struct_feature\*.pdb.gz
    ↓ [Bước 1]
    └→ valid_protein_ids.csv
    
D:\raw_data\goa_human.gaf.gz
    ↓ [Bước 2]
    └→ HUMAN_protein_info.json
        ↓
        └→ human_BP/MF/CC_ACS.json
        
valid_protein_ids.csv + *.pdb.gz
    ↓ [Bước 3]
    └→ proteins_edges/*.txt (contact map)
    
D:\raw_data\seq.fasta
    ↓ [Bước 4A, 4B, 4C]
    └→ dict_sequence_feature (embedding)
    └→ protein_node2onehot (one-hot)
    
D:\raw_data\ppi.txt
    ↓ [Bước 5]
    ├→ uniprot_ensembl_mapping.csv
    │
    ├→ [Bước 6]
    └→ ppi_graph_global + ppi_protein_index
    
Tất cả output từ trên
    ↓ [Bước 7]
    └→ emb_graph_BP/MF/CC
    └→ emb_seq_feature_BP/MF/CC
    └→ emb_label_BP/MF/CC
    └→ emb_ppi_node_id_BP/MF/CC
    └→ label_vocab_BP/MF/CC.json
    └→ label_BP/MF/CC_network
    
    ↓ [Bước 8]
    └→ divided_data/BP_train/valid/test_dataset
    └→ divided_data/MF_train/valid/test_dataset
    └→ divided_data/CC_train/valid/test_dataset
    
    ↓ [Training]
    └→ save_models/bestmodel_*.pkl
```

---

## Chạy Toàn Bộ Pipeline

### Script runner (tự động):

Chạy tất cả bước một lúc:
```bash
cd D:\CAFA6

# Bước 1
python data_processing/1_get_valid_ids.py

# Bước 2
python data_processing/go_anno.py

# Bước 2b (build namespace + vocab)
python data_processing/2_build_go_namespace.py  # nếu có

# Bước 3
python data_processing/2_extract_struct_map.py

# Bước 4a (sequence to onehot)
python data_processing/get_sequence.py

# Bước 4b (ESM-2 embedding)
python data_processing/seq2vec.py \
  -i D:/raw_data/seq.fasta \
  -o D:/CAFA6/proceed_data/dict_sequence_feature \
  --model 650M

# Bước 4c (normalize)
python data_processing/read_seqvec_features.py

# Bước 5 (ENSP → UniProt mapping)
python data_processing/3_uniprot_mapping.py

# Bước 6 (build PPI graph)
python data_processing/4_build_ppi_graph.py

# Bước 7 (build graph dataset)
python data_processing/3_build_graph_dataset.py

# Bước 8 (split train/valid/test)
python data_processing/divide_data.py
```

### Parallelization hints:
- Bước 1, 2 có thể chạy song song
- Bước 3, 4A, 4B, 5 có thể chạy song song (độc lập)
- Bước 6 phụ thuộc vào kết quả bước 5
- Bước 7 phụ thuộc vào 1–6
- Bước 8 phụ thuộc vào bước 7

---

## Lưu Ý Quan Trọng

1. **Đường dẫn:**
   - Tất cả path cứng (hardcoded) `D:\CAFA6`, `D:\raw_data`
   - Nếu cần thay đổi → edit từng script

2. **Thứ tự:**
   - Phải chạy **đúng thứ tự**, không thể skip
   - Ví dụ: không thể chạy bước 7 trước bước 3

3. **Dung lượng disk:**
   - proceed_data: ~10–50 GB (tùy số protein)
   - divided_data: ~5–10 GB

4. **Memory:**
   - Tối thiểu 16 GB RAM (khuyến nghị 32 GB)
   - Nếu OOM → giảm batch_size hoặc split theo namespace

5. **GPU:**
   - Bước 4B (ESM-2) rất nhanh trên GPU
   - CPU có thể chạy nhưng mất 12–24 giờ

6. **Seed reproducibility:**
   - Bước 8 dùng seed=42 mặc định
   - Thay đổi seed → khác split (nhưng vẫn 70:20:10)

