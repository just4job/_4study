# So sánh công bằng với baseline (cùng hyperparameter train/eval)

Model CAFA6 (**Struct2GO2**) vẫn có **PPI + cross-attention** so với baseline gốc (chỉ graph + sequence).  
Preset `--baseline-parity` chỉ đồng bộ **protocol huấn luyện và đánh giá** với [`README.md`](README.md) / [`EVAL.md`](EVAL.md), không tắt PPI.

---

## Train (`train_Struct2GO2.py`)

| Tham số | `--baseline-parity` | Ghi chú |
|---------|---------------------|---------|
| epoch | 20 | |
| batch_size | 64 | Tên checkpoint: `bestmodel_{branch}_64_0.0001_0.3.pkl` |
| lr | 1e-4 | |
| dropout | 0.3 (paper); **mf 0.1, cc 0.2, bp 0.1** (`scripts/baseline_config.py`) | MF `0.3` + PPI hay train fail |
| hid_dim | 512 | |
| num_convs | 6 | |
| pool_ratio | 0.75 | |
| validate_every | 4 | Epoch 3, 7, 11, 15, 19 |
| F-max thresholds | 99 | 0.01 … 0.99 |
| optimizer | Adam | Baseline; CAFA6 mặc định AdamW |
| loss | BCEWithLogitsLoss | Giữ BCE (multi-label GO); baseline paper dùng CE |
| amp | tắt | |
| PPI | **bật** | Khác biệt kiến trúc so với Table 1 |

### Local

**Mặc định:** `--baseline-parity` + **PPI bật** (không cần thêm flag).

```bash
set DATA_DIR=D:\CAFA6
python train_Struct2GO2.py -branch mf
python train_Struct2GO2.py -branch cc
python train_Struct2GO2.py -branch bp
```

Train nhanh (Kaggle / thử nghiệm): `--kaggle` hoặc `--no-baseline-parity`. Tắt PPI (ablation): `--no-ppi`.

### Kaggle (T4 — ~vài giờ/nhánh, có thể OOM; giảm batch nếu cần)

```bash
export DATA_DIR=/kaggle/working/CAFA6
export DGL_CUDA=1
python train_Struct2GO2.py -branch mf --baseline-parity
```

Nếu OOM: vẫn giữ preset nhưng override `-batch_size 32` (ghi rõ trong báo cáo).

**Không** kết hợp `--kaggle` với `--baseline-parity` cho cùng một lần chạy — hai preset trái nhau (`hid` 256 vs 512).

---

## Eval (`eval_Struct2GO2.py`)

| Tham số | `--baseline-parity` |
|---------|---------------------|
| split | **test** (bắt buộc; không fallback valid) |
| batch_size | 32 (cố định trong script) |
| model mặc định | `save_models/bestmodel_{branch}_64_0.0001_0.3.pkl` |
| `-thresh` (JSON) | mf **0.71**, cc **0.5**, bp **0.4** |
| F-max trong log | quét 99 ngưỡng |

```bash
set DATA_DIR=D:\CAFA6
python eval_Struct2GO2.py -branch mf
python eval_Struct2GO2.py -branch cc
python eval_Struct2GO2.py -branch bp
```

Cần có `divided_data/{branch}_test_dataset` (pack bằng `pack_for_kaggle.py` hoặc `divide_data.py`).

---

## Cách trình bày trong báo cáo

1. **Baseline paper:** Struct2GO / with one-hot — Table 1, không PPI.  
2. **Bạn (fair protocol):** Struct2GO2 + PPI, train/eval cùng epoch, hid, conv, pool, validate, 99 ngưỡng.  
3. **Bạn (Kaggle nhanh):** `--kaggle` — không so trực tiếp với Table 1 về protocol.

Khác biệt còn lại sau `--baseline-parity`: PPI, ESM sequence (640-d), fusion cross-attention, BCE thay CE.
