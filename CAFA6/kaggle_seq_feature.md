# Chạy `5_build_seq_feature.py` trên Kaggle (ESM-2, cần GPU)

Đây là bước DUY NHẤT của tiền xử lý cần GPU. Máy local chạy được mọi bước khác;
ESM-2 trên ~20.500 protein bằng CPU mất nhiều giờ, trên T4 chỉ khoảng 30–60 phút.

**Vào / ra:**

| | |
|---|---|
| Input | `raw_data/seq.fasta` (~15 MB), `proceed_data/valid_protein_ids.csv` (~140 KB) |
| Output | `proceed_data/dict_sequence_feature` — `{UniProt_ID: ndarray(1024,)}`, ~85 MB |

Chỉ cần upload 2 file nhỏ, KHÔNG cần upload `proteins_edges/` hay bất cứ thứ gì khác.

---

## Bước 0 — Đóng gói input ở máy local

```bash
cd ~/Downloads/_4study
mkdir -p ~/cafa6_seq_input
cp raw_data/seq.fasta CAFA6/proceed_data/valid_protein_ids.csv ~/cafa6_seq_input/
ls -lh ~/cafa6_seq_input
```

Đặt trong `$HOME` chứ KHÔNG phải `/tmp`: trình duyệt cài bằng snap hoặc flatpak
(mặc định của Firefox/Chromium trên Ubuntu) chạy trong namespace riêng và thấy một
`/tmp` khác, nên hộp chọn file của nó không tìm ra thư mục vừa tạo.

Upload thư mục đó lên [Kaggle Datasets](https://www.kaggle.com/datasets) → **New
Dataset** → đặt tên ví dụ `cafa6-seq-input`.

---

## Bước 1 — Bật GPU **và Internet**

Trong notebook, mở panel bên phải (biểu tượng `<` ở góc phải nếu đang ẩn):

- **Accelerator** = `GPU T4 x2` (hoặc P100). Không bật thì chạy bằng CPU, rất chậm.
- **Internet** = `On`. Cell 1 phải `git clone` và `pip install fair-esm`; tắt Internet
  thì cả hai lệnh đó đều lỗi.

Cả hai tuỳ chọn đòi tài khoản Kaggle đã **xác minh số điện thoại**
(Settings → Phone Verification). Chưa xác minh thì mục Internet bị khoá mờ và
Accelerator chỉ có `None`.

---

## Cell 1 — Clone repo + cài fair-esm

```python
!git clone -b claude/cafa6-folder-summary-8j9xch \
    https://github.com/just4job/_4study.git /kaggle/working/_4study
!pip install -q fair-esm
```

`fair-esm` là gói DUY NHẤT phải cài — biopython, torch, pandas, tqdm Kaggle đã có sẵn.
Bước này KHÔNG cần dgl.

---

## Cell 2 — Trỏ đường dẫn

```python
import glob, os, shutil

REPO = "/kaggle/working/_4study/CAFA6"

# Tìm thư mục dataset chứa seq.fasta — không đoán mò theo vị trí, vì notebook
# có thể đang gắn nhiều dataset.
hits = glob.glob("/kaggle/input/**/seq.fasta", recursive=True)
assert hits, "Không thấy seq.fasta trong /kaggle/input — đã gắn dataset chưa?"
RAW = os.path.dirname(hits[0])

os.environ["DATA_DIR"] = REPO          # proceed_data nằm ngay trong repo
os.environ["RAW_DIR"]  = RAW

os.makedirs(f"{REPO}/proceed_data", exist_ok=True)
shutil.copy(f"{RAW}/valid_protein_ids.csv", f"{REPO}/proceed_data/")

print("DATA_DIR:", REPO)
print("RAW_DIR :", RAW)
!ls -la {RAW}
```

`data_processing/paths.py` đọc 2 biến này nên không phải sửa dòng nào trong script
(xem README mục 1). `seq.fasta` đọc thẳng từ `RAW_DIR`; `valid_protein_ids.csv` phải
được chép vào `proceed_data/` vì script tìm nó ở đó chứ không phải ở `RAW_DIR`.

---

## Cell 3 — Kiểm tra GPU

```python
import torch
print("CUDA:", torch.cuda.is_available())
if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0),
          f"{torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")
```

`CUDA: False` → quay lại Settings bật Accelerator, đừng chạy tiếp.

---

## Cell 4 — Chạy ESM-2

Ô này TỰ CHỨA — clone repo, cài gói, đặt biến môi trường rồi mới gọi script. Nhờ
vậy nó chạy đúng cả khi Kaggle khởi động một session hoàn toàn mới (chế độ
**Save & Run All** bên dưới luôn bắt đầu từ máy trắng).

```python
import glob, os, shutil

REPO = "/kaggle/working/_4study/CAFA6"

if not os.path.isdir(REPO):
    !git clone -b claude/cafa6-folder-summary-8j9xch https://github.com/just4job/_4study.git /kaggle/working/_4study
!pip install -q fair-esm

hits = glob.glob("/kaggle/input/**/seq.fasta", recursive=True)
assert hits, "Không thấy seq.fasta — dataset chưa gắn"
os.environ["DATA_DIR"] = REPO
os.environ["RAW_DIR"] = os.path.dirname(hits[0])
os.makedirs(f"{REPO}/proceed_data", exist_ok=True)
shutil.copy(f"{os.environ['RAW_DIR']}/valid_protein_ids.csv", f"{REPO}/proceed_data/")

!cd /kaggle/working/_4study/CAFA6 && python data_processing/5_build_seq_feature.py
```

Dòng cuối viết thẳng đường dẫn, KHÔNG dùng `!cd {REPO}`: cú pháp `{...}` lấy giá
trị từ bộ nhớ notebook, nên sau khi session ngắt nó truyền nguyên chuỗi `{REPO}`
cho bash và báo `cd: {REPO}: No such file or directory`. `os.environ` cũng được
set lại ngay trong cùng ô để tiến trình `python` con nhận đúng `DATA_DIR`/`RAW_DIR`.

### Chạy bằng "Save & Run All", đừng ngồi canh tab

`/kaggle/working` **chỉ tồn tại trong một session**. Session kết thúc (đóng tab quá
lâu, hết hạn mức, mạng rớt) là mất sạch: repo đã clone, gói đã `pip install`, và cả
file checkpoint. Checkpoint mỗi 200 protein chỉ cứu được trường hợp kernel restart
TRONG cùng session — không cứu được khi session chết.

Với job 30–60 phút, cách đúng là chạy nền:

1. Dán toàn bộ ô trên vào **một ô duy nhất** (nó đã tự chứa)
2. **Save Version** (góc trên phải) → chọn **Save & Run All (Commit)**, KHÔNG phải
   "Quick Save" → **Save**

Kaggle chạy notebook từ đầu đến cuối trên máy chủ của họ. Đóng tab, tắt máy đều
được. Xem tiến độ ở tab **Versions**.

Quan trọng: output của một Commit được **lưu vĩnh viễn** vào tab **Output** của
version đó, khác hẳn session tương tác.

**Cấu hình mặc định** (sửa ở đầu `5_build_seq_feature.py` nếu cần):

| Hằng số | Giá trị | Ý nghĩa |
|---|---|---|
| `ESM_MODEL_NAME` | `esm2_t30_150M_UR50D` | 150M tham số, dim 640 |
| `TARGET_DIM` | 1024 | Chiếu tuyến tính 640 → 1024 |
| `MAX_SEQ_LEN` | 5000 | **Protein dài hơn bị BỎ HẲN** (36/20.550 = 0,18%) |
| `BATCH_SIZE` | 1 | An toàn VRAM |

`MAX_SEQ_LEN` đáng lưu ý: protein bị bỏ sẽ không có trong output, và
`3_build_graph_dataset.py` lặng lẽ thay bằng zero vector cho toàn bộ nhánh sequence
của protein đó — không báo lỗi.

Ngưỡng 5000 chọn theo phân bố thật của proteome người: 2000 bỏ 465 protein (2,3%),
5000 chỉ còn 36 (0,18%). Nâng cao hơn lãi rất ít (8000 thêm 30 protein) trong khi
attention là O(L²) nên dễ OOM — chuỗi dài nhất là 34.350 residue (titin). Protein
nào vẫn OOM thì `except RuntimeError` bắt riêng, `empty_cache()` rồi chạy tiếp,
không làm hỏng cả job.

`python scripts/check_inputs.py` ở local in ra con số thật — nó đọc `MAX_SEQ_LEN`
thẳng từ script này nên luôn khớp.

---

## Cell 5 — Tải kết quả về

```python
import pickle, os
p = "/kaggle/working/_4study/CAFA6/proceed_data/dict_sequence_feature"
with open(p, "rb") as f:
    d = pickle.load(f)
print(f"{len(d):,} protein, {os.path.getsize(p)/1e6:.0f} MB")
first = next(iter(d.values()))
print("shape:", getattr(first, "shape", len(first)))
```

Kỳ vọng: khoảng 20.000+ protein, vector 1024 chiều, ~85 MB.

Chạy bằng **Save & Run All** thì vào notebook → tab **Output** của version vừa
chạy → tải `dict_sequence_feature` về, chép vào
`~/Downloads/_4study/CAFA6/proceed_data/`.

Xoá file checkpoint cho gọn (không cần giữ):

```python
!rm -f /kaggle/working/_4study/CAFA6/proceed_data/dict_sequence_feature.ckpt
```

---

## Về lại máy local

```bash
cd ~/Downloads/_4study/CAFA6
python scripts/check_inputs.py --skip-ppi
```

`dict_sequence_feature` phải hiện `OK` và dòng phủ đạt ≥ 90%. Sau đó chạy trọn
chuỗi build (xem README mục 3 — thứ tự chạy đầy đủ):

```bash
python data_processing/split_protein_ids.py --force
python data_processing/4_build_ppi_graph.py
python data_processing/3_build_graph_dataset.py
python data_processing/divide_data.py --force
python scripts/audit_data.py --deep
python pack_for_kaggle.py
```

Rồi upload `kaggle_data.zip` để train — xem [`kaggle_notebook.md`](kaggle_notebook.md).
