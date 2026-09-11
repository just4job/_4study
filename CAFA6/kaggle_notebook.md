# Kaggle notebook — hướng dẫn chạy (copy từng cell)

Bật **GPU T4 x1** + **Internet**. Nếu lỗi DGL/numpy → **Restart session** → chạy lại từ Cell 1.

---

## 0. Điều kiện tiên quyết — dataset phải là bản ĐÃ ĐÓNG GÓI LẠI

Pipeline đã đổi (thêm Bước 2b, sửa vocab + PPI leakage guard — xem
[README mục 3](README.md#3-pipeline-xử-lý-dữ-liệu--chi-tiết-từng-script) và
[mục 4.5](README.md#45-chống-rò-rỉ-dữ-liệu-qua-ppi-ppi-leakage-guard)).

- **Đang có dataset cũ** (pack theo pipeline trên `main`) → xem
  [mục migrate ngay bên dưới](#đang-dùng-dataset-cũ-pack-theo-pipeline-trên-main)
  (~vài phút, không phải build lại, giữ nguyên `divided_data`).
- **Chỉ còn file zip cũ, không còn `proceed_data` gốc ở local?** Vẫn sửa được
  cả vocab min-count bằng `migrate_old_data.py --relabel` — xem
  [mục migrate](#đang-dùng-dataset-cũ-pack-theo-pipeline-trên-main).
- **Muốn build lại hoàn toàn** (đổi cả split / dữ liệu nguồn đổi) → chạy ở **máy
  local** rồi pack lại:

```bash
# Trên máy local, trong thư mục CAFA6
python data_processing/split_protein_ids.py --force        # Bước 2b (mới)
python data_processing/4_build_ppi_graph.py                # -> ppi_graph_train_{ns}
python data_processing/3_build_graph_dataset.py            # -> label_vocab/label_network chỉ từ train
python data_processing/divide_data.py --force              # đọc lại split_{ns}.json
python scripts/audit_data.py --deep                        # KIỂM TRA trước khi pack
python pack_for_kaggle.py                                  # -> kaggle_data.zip
```

> Chạy `audit_data.py` **ở local trước khi pack** — phát hiện lỗi ở đây rẻ hơn
> nhiều so với sau khi đã upload vài GB lên Kaggle. Cell 5 sẽ chạy lại nó trên
> Kaggle để chắc chắn dữ liệu không hỏng trong lúc upload/giải nén.

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

### Đang dùng dataset CŨ (pack theo pipeline trên `main`)?

Không phải build lại từ đầu. Chạy migrate ngay trên Kaggle (~vài phút):

```python
%env DATA_DIR=/kaggle/working/_4study/CAFA6
# --materialize: proceed_data đang là symlink tới /kaggle/input (read-only)
!python /kaggle/working/_4study/CAFA6/scripts/migrate_old_data.py --materialize
!python /kaggle/working/_4study/CAFA6/scripts/audit_data.py
```

Script suy ra `split_{ns}.json` **từ chính `divided_data` đang có** (giữ nguyên
phân vùng train/valid/test cũ), rồi dựng `ppi_graph_train_{ns}` và dựng lại
`label_{ns}_network` chỉ từ protein train.

Mặc định script sửa 2 trong 3 vấn đề: PPI leak (guard build-time) và leak
`label_{ns}_network`. Chiều nhãn không đổi → checkpoint cũ vẫn load được.

**Sửa nốt vocab min-count bằng `--relabel`** (không cần contact map, không cần
`raw_data` — chỉ cần chính bản pack):

```python
%env DATA_DIR=/kaggle/working/_4study/CAFA6
# Xem trước vocab mới còn bao nhiêu nhãn, chưa ghi gì:
!python /kaggle/working/_4study/CAFA6/scripts/migrate_old_data.py --relabel --dry-run

# Chạy thật (ghi đè ds.label của divided_data theo vocab mới):
!python /kaggle/working/_4study/CAFA6/scripts/migrate_old_data.py --materialize --relabel
!python /kaggle/working/_4study/CAFA6/scripts/audit_data.py
```

Vector nhãn chỉ phụ thuộc `human_{NS}_ACS.json` + vocab (đều có trong bản pack),
nên tính lại được mà không đụng tới graph cấu trúc / seq feature / `ppi_node_id`.
Split giữ nguyên.

Lưu ý khi chạy `--relabel` trên Kaggle:

- `divided_data/*` mặc định là **symlink** tới `/kaggle/input`. Script ghi qua
  file tạm rồi `os.replace`, nên nó **thay symlink** bằng file thật trong
  `/kaggle/working` — cần chỗ trống ≈ kích thước dataset của nhánh đó (`mf_train`
  là file nặng nhất). Hết chỗ → chạy `--relabel` cho từng nhánh
  (`--branch mf`), hoặc chạy ở local rồi `pack_for_kaggle.py` upload lại.
- `num_labels` đổi → **train lại từ đầu**, checkpoint cũ vô dụng.
- Chạy lại lần 2 với cùng ngưỡng là no-op (script tự nhận ra vocab không đổi).
- Đổi ngưỡng: `--min-bp 250 --min-other 100` (mặc định, khớp
  `split_protein_ids.py`).

> **Thiếu `ppi_graph_train_*` / `split_*.json` / `label_vocab_*.json`** → vẫn chạy
> được nhưng rơi về đường fallback cũ (mask PPI lúc runtime — chậm và tốn RAM hơn;
> vocab không lọc tần suất — mất cân bằng nặng). Xem cảnh báo `[WARN]` trong log.
>
> **Checkpoint cũ:** nếu **build lại** hoặc chạy `--relabel` thì `num_labels` đổi →
> checkpoint cũ không load được, phải train lại từ đầu. Nếu chỉ **migrate** (không
> `--relabel`) thì chiều nhãn không đổi nên checkpoint cũ vẫn load được — nhưng nó
> được train khi còn leak, nên vẫn nên train lại để có số liệu sạch.

---

## Cell 1 — Clone repo

```python
!rm -rf /kaggle/working/_4study
!git clone --depth 1 --filter=blob:none --no-checkout \
    -b claude/cafa6-folder-summary-8j9xch \
    https://github.com/just4job/_4study.git /kaggle/working/_4study
!cd /kaggle/working/_4study && git sparse-checkout set --no-cone '/*' '!/CAFA6/save_models' \
    && git checkout claude/cafa6-folder-summary-8j9xch
!du -sh /kaggle/working/_4study && ls /kaggle/working/_4study/CAFA6
```

Chờ vài giây, `du -sh` ra khoảng **25 MB**.

Clone thường mất khoảng **314 MB** vì `CAFA6/save_models/` chứa ~290 MB checkpoint
của các lần train cũ. Chúng vô dụng ở đây — pipeline mới đổi `num_labels` nên
checkpoint cũ không load được, phải train lại từ đầu — nên ba cờ dưới đây bỏ hẳn
việc tải chúng về:

| Cờ | Tác dụng |
|---|---|
| `--depth 1` | chỉ lấy commit mới nhất, bỏ toàn bộ lịch sử |
| `--filter=blob:none` | hoãn tải nội dung file, chỉ tải file nào thật sự checkout |
| `sparse-checkout ... '!/CAFA6/save_models'` | loại thư mục checkpoint ra khỏi checkout |

`train_Struct2GO2.py` tự `makedirs("save_models", exist_ok=True)` nên thiếu thư mục
đó không ảnh hưởng gì tới việc train.

`rm -rf` trước khi clone: nếu lần chạy trước clone dở dang, thư mục đã tồn tại và
`git clone` báo *"already exists and is not an empty directory"*.

`/kaggle/working` **chỉ sống trong một session**. Session kết thúc là mất cả repo
đã clone lẫn gói đã `pip install` — nên với job train 1–2 giờ, hãy chạy bằng
**Save Version → Save & Run All (Commit)** thay vì ngồi canh tab (xem Cell 6).

**Cập nhật code mà không clone lại** — clone shallow thì `git pull` KHÔNG chạy
được (không đủ lịch sử chung, git báo *"divergent branches"*), phải ép con trỏ:

```python
!cd /kaggle/working/_4study \
    && git fetch --depth=1 origin claude/cafa6-folder-summary-8j9xch \
    && git reset --hard FETCH_HEAD
```

An toàn vì bản clone trên Kaggle không có thay đổi cục bộ, và `reset` giữ nguyên
cấu hình sparse-checkout. Nếu đã `import` module cũ trước đó thì nhớ
`importlib.reload(...)` — Python nhớ kết quả import cũ, kể cả lần import hỏng.

---

## Cell 2 — Cài thư viện + DGL CUDA

```python
!pip install -q "numpy>=1.26,<2.4" "scipy>=1.11,<1.16"
!pip install -q packaging fair-esm transformers biopython tqdm scikit-learn pandas networkx requests psutil
!python /kaggle/working/_4study/CAFA6/scripts/kaggle_fix_dgl.py
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
!python /kaggle/working/_4study/CAFA6/scripts/kaggle_link_data.py
```

Chỉ dùng 1 nhánh cho nhanh (bỏ validate pickle nặng của nhánh khác):

```python
!python /kaggle/working/_4study/CAFA6/scripts/kaggle_link_data.py --branches cc
```

---

## Cell 5 — Khoá `DATA_DIR` + audit dữ liệu

Chạy trước train/eval: khoá `DATA_DIR` (tránh notebook cũ quay về `D:/CAFA6`) rồi
audit toàn bộ dữ liệu đã chuẩn bị.

```python
%env DATA_DIR=/kaggle/working/_4study/CAFA6
!python /kaggle/working/_4study/CAFA6/scripts/audit_data.py
```

`scripts/audit_data.py` kiểm tra 6 nhóm (exit code 1 nếu có `FAIL`):

| # | Kiểm tra | Bắt được lỗi gì |
|---|---|---|
| 1 | Đủ artifact pipeline mới (`split_*`, `label_vocab_*`, `ppi_graph_train_*`) | Dataset còn là bản cũ → sẽ chạy fallback |
| 2 | Split rời nhau, phủ hết protein trong ACS | Trùng protein giữa train/valid/test |
| 3 | Vocab đã lọc min-count trên train | Bug vocab không lọc (MF 5136 label), label không có positive ở train |
| 4 | **`ppi_graph_train_{ns}` thực sự không còn cạnh chạm valid/test** | PPI leakage guard không hoạt động (build khi chưa có `split_*.json`) |
| 5 | Chiều nhãn khớp giữa vocab / label_network / dataset | Trộn dữ liệu cũ và mới → F-max ~0.002 |
| 6 | Label không có positive nào ở valid/test | Mất cân bằng do random split |

> Kiểm tra #4 là quan trọng nhất và **chỉ chạy được ở nơi có torch/dgl** (Kaggle
> sau Cell 2, hoặc máy local có env đầy đủ) — nó mở graph ra đếm thật, không chỉ
> xem file có tồn tại.

Đối chiếu sâu hơn (load cả `divided_data`, vài GB — chậm nhưng chắc chắn):

```python
!python /kaggle/working/_4study/CAFA6/scripts/audit_data.py --deep
```

Kiểm tra thêm: protein ID trong `{ns}_{train,valid,test}_dataset` có đúng nhóm
theo `split_{ns}.json` không, 3 tập có rời nhau không.

---

## Cell 6 — Train + eval cả 3 nhánh

`kaggle_run_branches.py` train theo thứ tự cc → mf → bp, eval sau mỗi nhánh, và
cập nhật `/kaggle/working/cafa6_output.zip` ngay sau mỗi nhánh (an toàn nếu
session bị ngắt giữa chừng). ~1–2 giờ tổng (BP chậm nhất).

```python
%env DATA_DIR=/kaggle/working/_4study/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/_4study/CAFA6/scripts/kaggle_run_branches.py
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
!python /kaggle/working/_4study/CAFA6/scripts/kaggle_run_branches.py -- -batch_size 64
```

Train riêng 1 nhánh (kiểm soát đầy đủ tham số):

```python
%env DATA_DIR=/kaggle/working/_4study/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/_4study/CAFA6/train_Struct2GO2.py \
  -branch bp --no-baseline-parity --kaggle \
  -epochs 6 -dropout 0.1 -batch_size 64 -validate_every 3 --amp

!python /kaggle/working/_4study/CAFA6/eval_Struct2GO2.py -branch bp --baseline-parity --split test
```

---

## Cell 7 — Đọc log: 4 dòng cần kiểm tra

```python
!tail -40 /kaggle/working/_4study/CAFA6/log/cc.log
!tail -20 /kaggle/working/_4study/CAFA6/log/test_cc.log
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
%env DATA_DIR=/kaggle/working/_4study/CAFA6
%env DGL_CUDA=1

# Đủ 4 hướng (~6–8 h) — dễ vượt giới hạn 1 session T4
!python /kaggle/working/_4study/CAFA6/scripts/run_fusion_ablation.py --profile balanced

# Gọn hơn: chỉ so 1 chiều vs 2 chiều (~3–4 h)
!python /kaggle/working/_4study/CAFA6/scripts/run_fusion_ablation.py --profile balanced --configs ppi_attn ppi_bi_attn

# Chỉ 1 nhánh, override epoch
!python /kaggle/working/_4study/CAFA6/scripts/run_fusion_ablation.py --profile balanced --branches mf -epochs 12

# Chỉ eval lại (checkpoint đã có)
!python /kaggle/working/_4study/CAFA6/scripts/run_fusion_ablation.py --eval-only --profile balanced
```

### 8.2 Loss — `bce` / `bce_pos_weight` / `focal`

`--loss` áp dùng chung cho mọi config fusion trong một lần chạy (không nhân chéo
fusion × loss để tránh nổ số run). Checkpoint được thêm hậu tố `_{loss}` nên
không đè lên run mặc định.

```python
!python /kaggle/working/_4study/CAFA6/scripts/run_fusion_ablation.py --configs ppi_attn --loss bce_pos_weight
!python /kaggle/working/_4study/CAFA6/scripts/run_fusion_ablation.py --configs ppi_attn --loss focal
```

Checkpoint: `bestmodel_{branch}_{config}_{batch}_{lr}_{dropout}[_{loss}].pkl`
Kết quả: `log/fusion_ablation_summary.json`, `log/test_{branch}_{config}.log`

---

## Cell 9 — Tải kết quả về máy

```python
import sys
sys.path.insert(0, "/kaggle/working/_4study/CAFA6")
from scripts.kaggle_download import download

download("cc")                  # log + test_result — vài trăm KB, tải ngay
# download("cc", models=True)   # kèm checkpoint .pkl (~40-60 MB/nhánh)
# download()                    # mọi nhánh đang có trong /kaggle/working
```

Phải `import` chứ không chạy bằng `!python`: trình duyệt chỉ tải được khi
notebook phát ra thẻ `<a download>`, mà muốn phát HTML thì code phải chạy trong
chính tiến trình notebook — `!python` là tiến trình con, output chỉ là văn bản.

Mặc định **không kèm checkpoint**. `log/` + `test_result/` là thứ cần đọc ngay
và chỉ vài trăm KB; checkpoint chiếm gần hết 60+ MB của bản zip đầy đủ. Cần
checkpoint để train tiếp thì `models=True`, hoặc lấy từ tab **Output** của
Version — đường đó không giới hạn dung lượng.

Script phát ra một **nút bấm**, không tự tải hộ: Kaggle render output trong
iframe sandbox nên trình duyệt chặn mọi download không đến từ thao tác người
dùng — `.click()` bằng script chỉ im lặng không làm gì. Một cú bấm thật thì luôn
được phép.

Hai ngưỡng an toàn (nhúng base64 làm 1 MB zip phình thành ~1,37 MB text trong
DOM): file > **25 MB** và tổng > **120 MB** thì không nhúng nữa mà đưa link thường.

Không thấy nút — đang chạy chế độ **Commit** (không có trình duyệt) hoặc output
bị nuốt — thì lấy file ở panel bên phải: **Output → `/kaggle/working` → 🔄
refresh → chuột phải file → Download**. Đường này luôn dùng được và không giới
hạn dung lượng.

Muốn gom file mà chưa tải: `download("cc", auto=False)` chỉ tạo zip rồi trả về
danh sách đường dẫn.

Xem nhanh có gì trong `/kaggle/working`:

```python
!ls -lh /kaggle/working/log /kaggle/working/test_result /kaggle/working/*.zip
```

**Zip lớn (đủ model 3 nhánh, ~300 MB):** Save Version (Commit) → chờ chạy xong →
tab **Output** → tải `cafa6_output_part*.zip`.

**Giải nén về local:** copy `log/`, `save_models/`, `test_result/` vào thư mục `CAFA6/` ở máy local.

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
!python /kaggle/working/_4study/CAFA6/scripts/diag_mf_train.py
```
Phải thấy `label_dim OK (train == valid == N)` với `N` = số dòng trong
`label_vocab_mf.json`.

**`CUDA out of memory` (T4 16GB)**
```python
!python /kaggle/working/_4study/CAFA6/scripts/kaggle_run_branches.py -- -batch_size 64
# hoặc nhẹ hơn nữa:
!python /kaggle/working/_4study/CAFA6/train_Struct2GO2.py -branch mf --kaggle -batch_size 48 -hid_dim 256 -num_convs 3
```

**RAM đầy khi load pickle (Kaggle 30GB)**
Preset `--kaggle` đã đặt `num_workers=0`. Nếu vẫn đầy: chạy từng nhánh một
(`--branches cc`) và Restart session giữa các nhánh.

**DGL không thấy CUDA**
Chạy lại Cell 2 (`kaggle_fix_dgl.py`) → **Restart session** → chạy lại từ Cell 1.
