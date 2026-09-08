# Thông số train — Baseline Struct2GO (gốc)

Tài liệu này ghi **cấu hình huấn luyện baseline** từ script gốc (paper / DSW), **không** bao gồm code.  
Đối chiếu với pipeline CAFA6 hiện tại: [`train_Struct2GO2.py`](../train_Struct2GO2.py).

---

## 1. Mục đích

Baseline Struct2GO dùng để so sánh trong bảng kết quả (Table 1):

- **Struct2GO (reproduced)**
- **with one-hot** (node2vec + one-hot, `in_dim = 56`)

---

## 2. Hyperparameters train (baseline gốc)

| Nhóm | Tham số | Giá trị | Ghi chú |
|------|---------|---------|---------|
| **Vòng lặp** | Số epoch | **20** | Cố định trong script gốc |
| | Validate | Mỗi **4** epoch | Epoch 3, 7, 11, 15, 19 |
| **Batch / LR** | `batch_size` | **64** | |
| | `learningrate` | **1e-4** | |
| | `dropout` | **0.3** | |
| **Optimizer** | Thuật toán | **Adam** | Không dùng AdamW |
| | LR schedule | Cosine + warmup | `num_warmup_steps = 100` |
| **Loss** | Hàm mất mát | **CrossEntropyLoss** | Khác BCEWithLogits (CAFA6) |
| **Kiến trúc** | `in_dim` | **56** | one-hot (26) + node2vec (30) |
| | `hid_dim` | **512** | Hidden GCN / attention |
| | `num_convs` | **6** | Số ConvPoolBlock |
| | `pool_ratio` | **0.75** | SAGPool |
| | `labels_num` | Theo nhánh | mf ≈ 328, cc/bp khác |
| **Đầu vào** | Cấu trúc | Graph + sequence | **Không** có nhánh PPI |
| | DataLoader | 4 thành phần | `pid, graph, label, seq` |
| **Đánh giá** | Ngưỡng F-max | **99** mức | 0.01, 0.02, …, 0.99 |
| | Metric báo cáo | F-max, AUC, AUPR, Recall, Precision | |
| **Lưu model** | Điều kiện | F-max validation tăng | |
| | Tên file | `bestmodel_{branch}_{batch}_{lr}_{dropout}.pkl` | Ví dụ: `bestmodel_mf_64_0.0001_0.3.pkl` |

---

## 3. Đường dẫn dữ liệu (môi trường gốc)

| Loại | Đường dẫn |
|------|-----------|
| Train set | `/etc/dsw/divided_data/{branch}_train_dataset` |
| Valid set | `/etc/dsw/divided_data/{branch}_valid_dataset` |
| Label network | `processed_data/label_{branch}_network` |

`{branch}` ∈ `mf`, `cc`, `bp`.

Trên CAFA6 / Kaggle thường dùng:

- `DATA_DIR/divided_data/`
- `DATA_DIR/proceed_data/` (hoặc `processed_data/`)

---

## 4. Kết quả paper (baseline — Table 1)

### 4.1 Struct2GO (reproduced)

| Ontology | F-max | AUC | AUPR |
|:--------:|:-----:|:---:|:----:|
| **MFO** (mf) | 0.302 | 0.808 | 0.256 |
| **CCO** (cc) | 0.531 | 0.886 | 0.601 |
| **BPO** (bp) | 0.303 | 0.749 | 0.261 |

### 4.2 Struct2GO with one-hot

| Ontology | F-max | AUC | AUPR |
|:--------:|:-----:|:---:|:----:|
| **MFO** (mf) | 0.399 | 0.836 | 0.421 |
| **CCO** (cc) | 0.557 | 0.887 | 0.600 |
| **BPO** (bp) | 0.334 | 0.767 | 0.305 |

*Đây là số trong bảng paper, không phải log train tự chạy.*

---

## 5. So sánh train: baseline vs CAFA6 (`train_Struct2GO2`)

| Tham số | Baseline gốc | CAFA6 (mặc định / `--kaggle`) |
|---------|--------------|-------------------------------|
| Epoch | 20 | 3–10 local; **4–5** preset Kaggle |
| `hid_dim` | 512 | 256 |
| `num_convs` | 6 | 3 |
| `pool_ratio` | 0.75 | 0.5 |
| `batch_size` | 64 | 64–96 |
| `dropout` | 0.3 | 0.1–0.3 (cc thường 0.2) |
| Optimizer | Adam | AdamW |
| Loss | CrossEntropyLoss | BCEWithLogitsLoss |
| PPI / Cross-attention | Không | Có |
| Mixed precision (`amp`) | Không | Có (Kaggle) |
| Cache PPI / epoch | Không | Có |
| Validate every | 4 epoch | 3–6 epoch (preset: cuối cùng) |
| Số ngưỡng F-max | 99 | 5 (preset Kaggle) |
| `ppi_node_ids` trong batch | Không | Có |

---

## 6. Ghi chú khi so sánh với kết quả của bạn

| Nội dung | Baseline paper | Run CAFA6 / Kaggle của bạn |
|----------|----------------|----------------------------|
| Kiến trúc | Nhỏ hơn (không PPI) hoặc one-hot | Struct2GO2 + PPI + ESM seq |
| Thời gian train | Đủ 20 epoch, model lớn (hid 512) | Ngắn hơn (2–15 epoch), hid 256 |
| Metric so sánh | Thường **test** trong paper | Log của bạn: **valid** / test copy valid |
| Ngưỡng suy luận | Quét 99 mức khi train | Train: 5 mức (Kaggle); test: quét lại trong eval |

Khi viết báo cáo, nên ghi rõ: *baseline theo Table 1 (Struct2GO / with one-hot)* và *cấu hình train CAFA6 (epoch, hid, PPI, preset Kaggle)*.

### So sánh công bằng (cùng hyperparameter baseline)

Dùng flag **`--baseline-parity`** trên train và eval (PPI vẫn bật). Lệnh và bảng đối chiếu: [`FAIR_COMPARISON.md`](FAIR_COMPARISON.md).

```bash
python train_Struct2GO2.py -branch mf --baseline-parity
python eval_Struct2GO2.py -branch mf --baseline-parity
```

---

## 7. Tham chiếu nhanh — train CAFA6 đã chạy (log)

| Nhánh | Epoch | Dropout | F-max (valid) | AUC | Model file |
|-------|-------|---------|---------------|-----|------------|
| CC | 12 | 0.2 | 0.575 | 0.937 | `bestmodel_cc_64_0.0001_0.2.pkl` |
| BP | 15 | 0.1 | 0.280 | 0.945 | `bestmodel_bp_64_0.0001_0.1.pkl` |
| MF | 2 | 0.1 | 0.481 | 0.863 | `bestmodel_mf_64_0.0001_0.1.pkl` |

Chi tiết bảng **test** so baseline: [`EVAL.md`](EVAL.md) (tham số eval + metric test).
