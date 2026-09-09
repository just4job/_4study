# Kaggle notebook — hướng dẫn chạy (copy từng cell)

Bật **GPU T4 x1** + **Internet**. Nếu lỗi DGL/numpy → **Restart session** → chạy lại từ Cell 1.

---

## 0. Điều kiện tiên quyết — dataset phải là bản ĐÃ ĐÓNG GÓI LẠI

Pipeline đã đổi (thêm Bước 2b, sửa vocab + PPI leakage guard — xem
[README mục 3](README.md#3-pipeline-xử-lý-dữ-liệu--chi-tiết-từng-script) và
[mục 4.5](README.md#45-chống-rò-rỉ-dữ-liệu-qua-ppi-ppi-leakage-guard)), nên
dataset Kaggle phải được build lại **trên máy local** rồi pack lại:

```bash
# Trên máy local (D:\CAFA6)
python data_processing/split_protein_ids.py --force        # Bước 2b (mới)
python data_processing/4_build_ppi_graph.py                # -> ppi_graph_train_{ns}
python data_processing/3_build_graph_dataset.py            # -> label_vocab/label_network chỉ từ train
python data_processing/divide_data.py --force              # đọc lại split_{ns}.json
python pack_for_kaggle.py                                  # -> kaggle_data.zip
```

Upload `kaggle_data.zip` lên [Kaggle Datasets](https://www.kaggle.com/datasets) →
**New Dataset** (vd. tên `cafa6-data`) → gắn vào notebook.

**Dataset đúng phải có đủ các file sau** (Cell 5 sẽ tự kiểm tra):

```
divided_data/{mf,cc,bp}_{train,valid,test}_dataset
proceed_data/ppi_graph_global          ← đầy đủ cạnh, dùng cho validate/test
proceed_data/ppi_graph_train_{mf,cc,bp} ← đã ẩn cạnh valid/test, dùng khi train
proceed_data/ppi_protein_index
proceed_data/split_{mf,cc,bp}.json      ← phân vùng train/valid/test chuẩn
proceed_data/label_vocab_{mf,cc,bp}.json ← vocab đã lọc min-count TRÊN TRAIN
proceed_data/label_{mf,cc,bp}_network   ← co-occurrence chỉ tính từ train
proceed_data/human_{MF,CC,BP}_ACS.json
```

> **Thiếu `ppi_graph_train_*` / `split_*.json` / `label_vocab_*.json`** → vẫn chạy
> được nhưng rơi về đường fallback cũ (mask PPI lúc runtime — chậm và tốn RAM hơn;
> vocab không lọc tần suất — mất cân bằng nặng). Xem cảnh báo `[WARN]` trong log.
>
> **Checkpoint cũ (train trước các fix này) KHÔNG dùng lại được** — `num_labels`
> đã đổi sau khi vocab được lọc đúng. Phải train lại từ đầu.

---

## Cell 1 — Clone repo

```python
%cd /kaggle/working
!rm -rf CAFA6
!git clone https://github.com/PNTLinh/CAFA6.git
%cd CAFA6
```

*(Đã clone rồi, chỉ cập nhật code: `%cd /kaggle/working/CAFA6` rồi `!git pull`)*

---

## Cell 2 — Cài thư viện + DGL CUDA

```python
!pip install -q "numpy>=1.26,<2.4" "scipy>=1.11,<1.16"
!pip install -q packaging fair-esm transformers biopython tqdm scikit-learn pandas networkx requests psutil
!python /kaggle/working/CAFA6/scripts/kaggle_fix_dgl.py
```

---

## Cell 3 — Kiểm tra GPU

```python
import torch
import dgl

g = dgl.graph(([0, 1], [1, 2]))
if torch.cuda.is_available():
    g = g.to("cuda")
print("OK", torch.__version__, dgl.__version__, g.device)
```

---

## Cell 4 — Nối dữ liệu từ /kaggle/input

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py
```

Chỉ dùng 1 nhánh cho nhanh (bỏ validate pickle nặng của nhánh khác):

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py --branches cc
```

---

## Cell 5 — Khoá `DATA_DIR` + kiểm tra dataset có đủ file mới

Chạy trước train/eval để tránh notebook cũ quay về `D:/CAFA6`, đồng thời xác nhận
dataset là bản **đã đóng gói lại** (có `split_*`, `label_vocab_*`, `ppi_graph_train_*`).

```python
import json
import os
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
root = Path(os.environ["DATA_DIR"])
print("DATA_DIR =", root)

BRANCHES = ["mf", "cc", "bp"]
ok = True

for br in BRANCHES:
    row = []
    for split in ("train", "valid", "test"):
        p = root / "divided_data" / f"{br}_{split}_dataset"
        row.append(f"{split}={'OK' if p.is_file() else 'MISSING'}")
    print(f"[{br}] divided_data: " + "  ".join(row))

print()
for br in BRANCHES:
    split_p = root / "proceed_data" / f"split_{br}.json"
    vocab_p = root / "proceed_data" / f"label_vocab_{br}.json"
    ppi_tr = root / "proceed_data" / f"ppi_graph_train_{br}"
    n_vocab = len(json.loads(vocab_p.read_text())) if vocab_p.is_file() else None
    print(
        f"[{br}] split_{br}.json={'OK' if split_p.is_file() else 'MISSING'}  "
        f"label_vocab={n_vocab if n_vocab is not None else 'MISSING'} label  "
        f"ppi_graph_train={'OK' if ppi_tr.is_file() else 'MISSING'}"
    )
    if not (split_p.is_file() and vocab_p.is_file() and ppi_tr.is_file()):
        ok = False

print("\nppi_graph_global:", (root / "proceed_data/ppi_graph_global").is_file())
print(
    "\n=> Dataset ĐÚNG bản mới" if ok else
    "\n=> [WARN] Thiếu file của pipeline mới — sẽ chạy fallback (xem mục 0)"
)
```

---

## Cell 6 — Train + eval cả 3 nhánh

`kaggle_run_branches.py` train theo thứ tự cc → mf → bp, eval sau mỗi nhánh, và
cập nhật `/kaggle/working/cafa6_output.zip` ngay sau mỗi nhánh (an toàn nếu
session bị ngắt giữa chừng). ~1–2 giờ tổng (BP chậm nhất).

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py
```

| Tình huống | Lệnh |
|---|---|
| Chỉ train, không eval | `!python .../kaggle_run_branches.py --no-eval` |
| Bỏ qua nhánh đã xong (chỉ copy + zip) | `!python .../kaggle_run_branches.py --skip mf` |
| Chỉ 1 nhánh | `!python .../kaggle_run_branches.py --branches cc` |
| Preset Table 1 (20 ep, hid 512 — chậm) | `!python .../kaggle_run_branches.py --baseline-parity` |
| Thêm tham số cho `train_Struct2GO2.py` | `!python .../kaggle_run_branches.py -- -batch_size 64 -epochs 5` |

> ⚠️ **Tham số thêm phải đặt sau `--`** (vd. `-- -batch_size 64`). Không có `--`
> thì argparse hiểu nhầm `-batch_size` là option của chính `kaggle_run_branches.py`
> và báo lỗi *unrecognized arguments*.

**OOM trên T4 16GB** → giảm batch:

```python
!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py -- -batch_size 64
```

Train riêng 1 nhánh (kiểm soát đầy đủ tham số):

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/train_Struct2GO2.py \
  -branch bp --no-baseline-parity --kaggle \
  -epochs 6 -dropout 0.1 -batch_size 64 -validate_every 3 --amp

!python /kaggle/working/CAFA6/eval_Struct2GO2.py -branch bp --baseline-parity --split test
```

---

## Cell 7 — Đọc log: 4 dòng cần kiểm tra

```python
!tail -40 /kaggle/working/CAFA6/log/cc.log
!tail -20 /kaggle/working/CAFA6/log/test_cc.log
```

| Dòng trong log | Ý nghĩa | Cần thấy gì |
|---|---|---|
| `[ppi-leak-guard] dùng ppi_graph_train_{br} đã build sẵn: giữ X/Y cạnh (build-time…)` | PPI leakage guard đang dùng graph đã ẩn cạnh valid/test | Phải là **build-time**. Nếu thấy *"mask lúc runtime (fallback…)"* → dataset thiếu `ppi_graph_train_*` (mục 0) |
| `loss=bce_pos_weight (per-label, cap=100.0, min=… max=… mean=…)` | Loss đang dùng + dải trọng số per-label | `max` cao hơn `min` nhiều = đang bù đúng cho label hiếm |
| `macro_f1=… \| rare(n=…)_f1=… \| medium(…)_f1=… \| common(…)_f1=…` | Chẩn đoán mất cân bằng | `rare_f1` quá thấp so với `common_f1` → thử `--loss focal` ([mục 4.6](README.md#46-loss-ablation-bce--bce_pos_weight--focal)) |
| `[thresh-select] threshold=… chọn từ VALID … KHÔNG quét lại trên 'test'` | Chống threshold-leak khi eval | Nếu thấy `[thresh-select][WARN] … (LEAK nếu 'test')` → thiếu `{branch}_valid_dataset` |

> F-max/AUPR đo bằng code hiện tại **thường thấp hơn** số cũ trong
> `baseline/EVAL.md` — vì số cũ có threshold-leak (chọn ngưỡng trên chính test).
> Đây là dấu hiệu fix đúng, không phải model kém đi.

---

## Cell 8 — Ablation (tuỳ chọn)

### 8.1 Fusion — 4 hướng

Script: [`scripts/run_fusion_ablation.py`](scripts/run_fusion_ablation.py)

| Hướng | PPI | Fusion | Ghi chú |
|---|:---:|:---:|---|
| `ppi_concat` | Có | concat | nối vector, không attention |
| `no_ppi_attn` | Không | attention | attention chỉ giữa struct/seq |
| `ppi_attn` | Có | attention | **1 chiều** — struct+seq là Query, PPI chỉ là Key/Value tĩnh |
| `ppi_bi_attn` | Có | bi_attention | **2 chiều** — struct/seq/PPI self-attend đối xứng, PPI cũng được cập nhật |

| Profile | MF (ep/lr) | CC (ep/lr) | BP (ep/lr) | ~Tổng 4 hướng × 3 nhánh |
|---|---|---|---|---|
| `fast` | 8 / 1e-4 | 8 / 1e-4 | 6 / 1e-4 | ~4–6 h |
| `balanced` (mặc định) | 10 / 1e-4 | 10 / 1e-4 | 8 / 1e-4 | ~6–8 h |
| `quality` | 15 / 1e-4 | 12 / 1e-4 | 12 / 1e-4 | ~10–14 h |

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

# Đủ 4 hướng (~6–8 h) — dễ vượt giới hạn 1 session T4
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --profile balanced

# Gọn hơn: chỉ so 1 chiều vs 2 chiều (~3–4 h)
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --profile balanced --configs ppi_attn ppi_bi_attn

# Chỉ 1 nhánh, override epoch
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --profile balanced --branches mf -epochs 12

# Chỉ eval lại (checkpoint đã có)
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --eval-only --profile balanced
```

### 8.2 Loss — `bce` / `bce_pos_weight` / `focal`

`--loss` áp dùng chung cho mọi config fusion trong một lần chạy (không nhân chéo
fusion × loss để tránh nổ số run). Checkpoint được thêm hậu tố `_{loss}` nên
không đè lên run mặc định.

```python
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --configs ppi_attn --loss bce_pos_weight
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --configs ppi_attn --loss focal
```

Checkpoint: `bestmodel_{branch}_{config}_{batch}_{lr}_{dropout}[_{loss}].pkl`
Kết quả: `log/fusion_ablation_summary.json`, `log/test_{branch}_{config}.log`

---

## Cell 9 — Lưu kết quả + tải về

```python
%env DATA_DIR=/kaggle/working/CAFA6

BRANCHES = "cc mf bp"   # đổi nếu mới xong 1–2 nhánh
!python /kaggle/working/CAFA6/scripts/kaggle_save_results.py \
  --branches {BRANCHES} --zip --per-branch-zip --split-mb 75
```

Gom `log/`, `save_models/`, `test_result/` về `/kaggle/working/` và tạo:
`cafa6_output.zip` (hoặc `cafa6_output_part1.zip`, `part2.zip`, … nếu > 75 MB) +
`cafa6_{cc,mf,bp}.zip` từng nhánh.

Kiểm tra + tải file nhỏ trực tiếp trong trình duyệt:

```python
import base64
from pathlib import Path
from IPython.display import HTML, FileLink, display

MAX_AUTO_MB = 75
work = Path("/kaggle/working")

for folder in ("log", "save_models", "test_result"):
    p = work / folder
    print(f"\n=== {folder}/ ===")
    for f in sorted(p.glob("*")) if p.is_dir() else []:
        size = f.stat().st_size
        print(f"  {f.name}  ({size/1e6:.1f} MB)" if size > 1e6 else f"  {f.name}  ({size} B)")

def download(path: Path, delay_ms: int = 0) -> None:
    if not path.is_file():
        return
    mb = path.stat().st_size / 1e6
    if mb > MAX_AUTO_MB:
        display(FileLink(str(path), result_html_suffix="?download=1"))
        print(f"  ⚠ {path.name} ({mb:.0f} MB) — bấm link, hoặc Save Version → tab Output")
        return
    uid = "dl_" + path.name.replace(".", "_")
    b64 = base64.b64encode(path.read_bytes()).decode()
    js = f"setTimeout(function(){{document.getElementById('{uid}').click();}},{delay_ms});" if delay_ms else ""
    display(HTML(
        f'<a id="{uid}" download="{path.name}" href="data:application/zip;base64,{b64}"></a>'
        f"<script>{js}</script><p>⬇️ <b>{path.name}</b> ({mb:.1f} MB)</p>"
    ))

print("\n=== Tải về ===")
for i, z in enumerate(sorted(work.glob("cafa6_*.zip"))):
    download(z, delay_ms=500 * i)
```

**Zip lớn (đủ model 3 nhánh, ~300 MB):** Save Version (Commit) → chờ chạy xong →
tab **Output** → tải `cafa6_output_part*.zip`.

**Giải nén về local:** copy `log/`, `save_models/`, `test_result/` vào `D:\CAFA6\`.

---

## Xử lý sự cố

**`[WARN] Thiếu file của pipeline mới` ở Cell 5**
Dataset đang là bản cũ. Build lại theo mục 0 trên máy local rồi upload dataset mới
(hoặc tạo version mới cho dataset cũ và gắn lại vào notebook).

**`mf_train_dataset` lỗi EOF / không đọc được**
`kaggle_link_data.py` tự gọi `scripts/repair_mf_train.py` để tìm bản `mf_train`
đọc được trong `/kaggle/input` và kiểm tra `label_dim` khớp
`proceed_data/label_vocab_mf.json`. Nếu vẫn lỗi, pack lại riêng MF ở local:
```bash
python pack_for_kaggle.py --branch mf --splits train
```

**`label_dim` train ≠ valid (F-max ~0.002)**
Đang trộn dataset cũ và mới. Kiểm tra nhanh:
```python
!python /kaggle/working/CAFA6/scripts/diag_mf_train.py
```
Phải thấy `label_dim OK (train == valid == N)` với `N` = số dòng trong
`label_vocab_mf.json`.

**`CUDA out of memory` (T4 16GB)**
```python
!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py -- -batch_size 64
# hoặc nhẹ hơn nữa:
!python /kaggle/working/CAFA6/train_Struct2GO2.py -branch mf --kaggle -batch_size 48 -hid_dim 256 -num_convs 3
```

**RAM đầy khi load pickle (Kaggle 30GB)**
Preset `--kaggle` đã đặt `num_workers=0`. Nếu vẫn đầy: chạy từng nhánh một
(`--branches cc`) và Restart session giữa các nhánh.

**DGL không thấy CUDA**
Chạy lại Cell 2 (`kaggle_fix_dgl.py`) → **Restart session** → chạy lại từ Cell 1.
