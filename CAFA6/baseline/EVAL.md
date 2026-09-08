# Thông số eval / test — Baseline vs CAFA6

Tài liệu ghi **cấu hình đánh giá (eval)** của script baseline gốc và đối chiếu với [`eval_Struct2GO2.py`](../eval_Struct2GO2.py) (CAFA6). **Không** chứa code.

Tham số train baseline: [`README.md`](README.md).

---

## 1. Tham số eval — Baseline (script gốc)

| Nhóm | Tham số | Giá trị | Ghi chú |
|------|---------|---------|---------|
| **Thiết bị** | `device` | `cuda:0` | Cố định |
| **CLI** | `-branch` | `mf` (mặc định) | `mf` / `cc` / `bp` |
| | `-thresh` | **0.71** (mặc định) | Ngưỡng ghi vào `*_result.json` |
| | `-batch` | `'1'` | Hậu tố file `pred_actual` |
| **Dữ liệu** | Tập eval | `{branch}_test_dataset` | Chỉ **test**, không fallback valid |
| | Label network | `processed_data/label_{branch}_network` | |
| | Từ điển GO | `processed_data/{branch}_term2idx.json` | Dùng **keys** làm danh sách term |
| **Model** | Đường dẫn mặc định | `save_models/bestmodel_{branch}_32_0.0001_0.2.pkl` | Train baseline: batch **32**, lr 1e-4, dropout **0.2** |
| **Inference** | `batch_size` | **32** | GraphDataLoader |
| | `shuffle` | `False` | |
| | Đầu vào batch | **4** phần tử | `pid, graph, label, seq` — **không PPI** |
| | Hàm forward | `model(graph, seq, label_network)` | |
| | Sau forward | `sigmoid(logits)` | |
| **Loss (log)** | | `CrossEntropyLoss` | Tính loss trên test (không train) |
| **Metric** | Quét F-max | **99** ngưỡng | 0.01 … 0.99 (`Thresholds`) |
| | Báo cáo log | loss, **F-max**, AUC, AUPR, R, P | F-max trong log = best sau quét 99 mức |
| | Ngưỡng JSON kết quả | `-thresh` CLI | Có thể ≠ ngưỡng F-max tốt nhất trong log |
| **Đầu ra** | Log | `log/test_{branch}.log` | |
| | Pickle | `test_result/{branch}{batch}_pred_actual.pkl` | |
| | JSON | `test_result/{branch}_result.json` | Term mới: `prob > thresh` và label thật = 0 |
| | Ảnh | `test_result/{branch}_roc_curve.png` | |

### Đường dẫn dữ liệu (baseline gốc)

| Loại | Path |
|------|------|
| Test set | `/etc/dsw/divided_data/{branch}_test_dataset` |
| Label network | `processed_data/label_{branch}_network` |
| Term index | `processed_data/{branch}_term2idx.json` |
| Model | `save_models/bestmodel_{branch}_32_0.0001_0.2.pkl` |
| Kết quả | `test_result/` (thư mục gốc cwd) |

### Ngưỡng `-thresh` mặc định (baseline script)

| Nhánh | `-thresh` mặc định trong script | Ghi chú README CAFA6 |
|-------|--------------------------------|----------------------|
| MF | 0.71 | Thường dùng sau valid train |
| CC | (cùng parser, đổi `-branch`) | ~0.50 |
| BP | (cùng parser, đổi `-branch`) | ~0.40 |

*Script baseline chỉ đặt default 0.71; cc/bp cần truyền tay khi chạy.*

---

## 2. Tham số eval — CAFA6 (`eval_Struct2GO2.py`)

| Nhóm | Tham số | Giá trị | Ghi chú |
|------|---------|---------|---------|
| **Thiết bị** | `device` | `cuda:0` nếu có GPU, else CPU | |
| **CLI** | `-branch` | `mf` | |
| | `-thresh` | **0.71** (mặc định) | Dùng cho `*_result.json` |
| | `-batch` | `'1'` | Hậu tố pickle |
| | `-model_path` | (tùy chọn) | Mặc định `DATA_DIR/save_models/bestmodel_{branch}_96_0.0001_0.2.pkl` |
| | `--split` | **`auto`** | Có `test` → test; không thì **valid** (Kaggle) |
| **Dữ liệu** | `DATA_DIR` | env, mặc định `D:/CAFA6` | |
| | Tập eval | `divided_data/{branch}_test_dataset` hoặc `_valid_dataset` | |
| | Label network | `proceed_data/label_{branch}_network` | |
| | PPI graph | `proceed_data/ppi_graph_global` | |
| | Từ điển GO | `proceed_data/label_vocab_{branch}.json` | Fallback: `human_{MF,CC,BP}_ACS.json` |
| **Model** | Load | `torch.load(..., weights_only=False)` | Hỗ trợ PyTorch 2.6+ |
| **Inference** | `batch_size` | **32** | |
| | Đầu vào batch | **5** phần tử | `+ ppi_node_ids` |
| | Forward | `model(..., ppi_graph, ppi_node_ids, ppi_node_emb)` | Cache PPI embedding 1 lần |
| | Collate | Pad/truncate seq & label theo model | |
| **Loss (log)** | | `CrossEntropyLoss` | `labels.float()` |
| **Metric** | Quét F-max | **99** ngưỡng | Giống baseline |
| **Đầu ra** | Log | `{DATA_DIR}/log/test_{branch}.log` | |
| | Kết quả | `{DATA_DIR}/test_result/` | |

### Model & ngưỡng bạn đã dùng (Kaggle, log mới)

| Nhánh | File model | `-thresh` train (valid) | `-thresh` test (log quét 99 mức) |
|-------|------------|-------------------------|----------------------------------|
| **CC** | `bestmodel_cc_64_0.0001_0.2.pkl` | 0.3 | **0.27** |
| **BP** | `bestmodel_bp_64_0.0001_0.1.pkl` | 0.3 | **0.19** |
| **MF** | `bestmodel_mf_64_0.0001_0.1.pkl` (local) | 0.7 | **0.26** |

Lệnh eval gợi ý:

```bash
python eval_Struct2GO2.py -branch cc -thresh 0.3 --split valid \
  -model_path save_models/bestmodel_cc_64_0.0001_0.2.pkl

python eval_Struct2GO2.py -branch bp -thresh 0.3 --split valid \
  -model_path save_models/bestmodel_bp_64_0.0001_0.1.pkl
```

---

## 3. Bảng so sánh tham số: Baseline eval vs CAFA6 eval

| Tham số | Baseline (gốc) | CAFA6 (`eval_Struct2GO2`) |
|---------|----------------|---------------------------|
| Tập dữ liệu | Chỉ **test** | **`auto`**: test → không có thì **valid** |
| Thư mục data | `/etc/dsw/`, `processed_data/` | `DATA_DIR/divided_data`, `proceed_data/` |
| Từ điển GO | `{branch}_term2idx.json` (keys) | `label_vocab_{branch}.json` hoặc ACS JSON |
| PPI | **Không** | **Có** (`ppi_graph_global` + node id) |
| Batch loader | 4 tuple | 5 tuple + `collate_fn` |
| `batch_size` infer | 32 | 32 |
| Model mặc định | `bestmodel_{branch}_32_0.0001_0.2.pkl` | `bestmodel_{branch}_96_0.0001_0.2.pkl` |
| `-thresh` JSON | CLI (vd. 0.71 mf) | CLI (theo valid của bạn) |
| F-max trong log | Quét 99 mức | Quét 99 mức (giống) |
| Loss test | CrossEntropyLoss | CrossEntropyLoss |
| Log file | `log/test_{branch}.log` | `{DATA_DIR}/log/test_{branch}.log` |
| DGL patch | Không | Có (`ensure_dgl_importable`) |

---

## 4. Kết quả test — Baseline paper vs của bạn

*Paper: Table 1 (Struct2GO reproduced / with one-hot). Số của bạn: log `test_*.log` (CC 24/05, MF 23/05, BP 4ep Kaggle 30p + run 15ep).*

### 4.0 BP — run Kaggle nhanh (dropout 0.1, batch 64, AMP)

| Run | Train | F-max | AUC | AUPR | Thresh F-max |
|-----|-------|:-----:|:---:|:----:|:------------:|
| **4 ep (~30p)** | `--kaggle -epochs 4` | **0.313** | **0.935** | **0.248** | 0.15 |
| **6 ep (~45p)** | `--kaggle -epochs 6 -validate_every 3` | **0.311** | **0.930** | **0.246** | 0.15 |

Recall ~0.331, Precision ~0.293 (cả hai run).

### 4.1 So với Struct2GO baseline (paper)

| Ontology | Metric | Paper baseline | **Bạn (test)** | Δ (điểm) | Tăng % |
|:--------:|:------:|:--------------:|:--------------:|:--------:|:------:|
| MF | F-max | 0.302 | 0.477 | +0.175 | +57.9% |
| MF | AUC | 0.808 | 0.858 | +0.050 | +6.2% |
| MF | AUPR | 0.256 | 0.441 | +0.185 | +72.3% |
| CC | F-max | 0.531 | 0.572 | +0.041 | +7.7% |
| CC | AUC | 0.886 | 0.933 | +0.047 | +5.3% |
| CC | AUPR | 0.601 | 0.612 | +0.011 | +1.8% |
| BP | F-max | 0.303 | **0.313** (4ep) / 0.311 (6ep) / 0.333 (15ep) | +0.010 / +0.008 / +0.030 | +3.4% / +2.6% / +9.9% |
| BP | AUC | 0.749 | **0.935** (4ep) / 0.930 (6ep) / 0.944 (15ep) | +0.186 | +24.8% |
| BP | AUPR | 0.261 | **0.248** (4ep) / 0.246 (6ep) / 0.281 (15ep) | −0.013 / −0.015 / +0.020 | −5.0% / −5.7% / +7.7% |

### 4.2 So với with one-hot (paper)

| Ontology | Metric | Paper one-hot | **Bạn (test)** | Δ (điểm) | Tăng % |
|:--------:|:------:|:-------------:|:--------------:|:--------:|:------:|
| MF | F-max | 0.399 | 0.477 | +0.078 | +19.5% |
| MF | AUC | 0.836 | 0.858 | +0.022 | +2.6% |
| MF | AUPR | 0.421 | 0.441 | +0.020 | +4.8% |
| CC | F-max | 0.557 | 0.572 | +0.015 | +2.7% |
| CC | AUC | 0.887 | 0.933 | +0.046 | +5.2% |
| CC | AUPR | 0.600 | 0.612 | +0.012 | +2.0% |
| BP | F-max | 0.334 | **0.313** (4ep) / 0.311 (6ep) | −0.021 / −0.023 | −6.3% / −6.9% |
| BP | AUC | 0.767 | **0.935** (4ep) / 0.930 (6ep) | +0.168 / +0.163 | +21.9% / +21.2% |
| BP | AUPR | 0.305 | **0.248** (4ep) / 0.246 (6ep) | −0.057 / −0.059 | −18.7% / −19.3% |

### 4.3 Hai loại ngưỡng (quan trọng khi đọc log)

| Loại | Baseline | CAFA6 / bạn |
|------|----------|-------------|
| **`-thresh` (CLI)** | Gán nhãn trong JSON (`*_result.json`) | Giống — nên lấy từ **valid train** |
| **F-max (log sau quét)** | Best trong 99 mức trên tập eval | Log in `thresh: 0.24` … — dùng báo cáo metric |

Ví dụ CC: ghi JSON có thể dùng `-thresh 0.3` (train), trong log test F-max đạt **0.572** tại ngưỡng quét **0.27**.

---

## 5. Khác biệt quan trọng khi so sánh công bằng

| Vấn đề | Baseline paper / script | Run của bạn |
|--------|-------------------------|-------------|
| Model train | hid 512, 6 conv, 20 epoch, batch 32/64 | hid 256, 3 conv, 4–15 epoch, PPI |
| Tập eval | Test set chuẩn paper | Kaggle: thường **valid** hoặc test = copy valid |
| Checkpoint eval | `*_32_0.0001_0.2.pkl` | `*_64_0.0001_0.2.pkl` / `0.1.pkl` |
| Kiến trúc infer | Không PPI | Struct2GO2 + PPI |

Khi viết báo cáo, nên ghi: *eval protocol CAFA6: split, model path, thresh train vs thresh F-max log, có PPI.*

---

## 6. Tóm tắt

- **Giống baseline:** batch 32, sigmoid, quét 99 threshold, ROC/AUPR/F-max, format JSON + ROC png.  
- **Khác baseline:** CAFA6 thêm PPI, `DATA_DIR`, fallback valid, vocab khác, model path và train config khác.  
- **Kết quả test của bạn:** vượt paper baseline cả 3 nhánh; CC/BP cạnh tranh **with one-hot** (BP F-max ≈ 0.33).
