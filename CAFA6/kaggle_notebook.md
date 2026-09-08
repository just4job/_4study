# Kaggle notebook — copy từng cell

Bật **GPU T4** + **Internet**. Nếu lỗi DGL/numpy → **Restart session** → chạy lại từ Cell 1.

Gắn dataset `cafa6-data` (từ `pack_for_kaggle.py`) vào notebook.

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

## Cell 4 — Nối dữ liệu

```python
!python /kaggle/working/CAFA6/scripts/kaggle_link_data.py
!ls /kaggle/working/CAFA6/proceed_data/ppi_graph_global
!ls /kaggle/working/CAFA6/divided_data/*_train_dataset
```

## Cell 4.5 — Khóa `DATA_DIR` và kiểm tra split

Chạy cell này trước khi train/eval để tránh notebook cũ hoặc kernel cũ quay về `D:/CAFA6`.

```python
import os
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
print("DATA_DIR =", os.environ["DATA_DIR"])

data_root = Path(os.environ["DATA_DIR"])
for rel_path in ["proceed_data", "divided_data"]:
    print(rel_path, (data_root / rel_path).exists())

for rel_path in [
    "divided_data/cc_test_dataset",
    "divided_data/cc_valid_dataset",
    "divided_data/mf_test_dataset",
    "divided_data/mf_valid_dataset",
    "divided_data/bp_test_dataset",
    "divided_data/bp_valid_dataset",
]:
    print(rel_path, (data_root / rel_path).exists())
```

Nếu `cc_test_dataset` và `cc_valid_dataset` đều `False`, dữ liệu Kaggle chưa đủ để eval branch `cc`; cần re-run `kaggle_link_data.py` hoặc pack lại dataset.

## Cell 4.6b — Vá `mf_train_dataset` nếu bị EOFError

Chạy cell này chỉ khi train nhánh `mf` và file `mf_train_dataset` bị rỗng/hỏng. Cell sẽ tìm file `mf_train_dataset` trong `/kaggle/input`, copy bản hợp lệ sang thư mục làm việc, rồi kiểm tra lại kích thước file.

```python
from pathlib import Path
import shutil

work_file = Path("/kaggle/working/CAFA6/divided_data/mf_train_dataset")
if work_file.exists() and work_file.stat().st_size > 0:
    print("mf_train_dataset OK:", work_file)
else:
    candidates = [p for p in Path("/kaggle/input").rglob("mf_train_dataset") if p.is_file() and p.stat().st_size > 0]
    if not candidates:
        raise FileNotFoundError(
            "Không tìm thấy mf_train_dataset hợp lệ trong /kaggle/input. Hãy gắn dataset mf-train1 rồi chạy lại cell này."
        )

    source = max(candidates, key=lambda p: p.stat().st_size)
    work_file.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, work_file)
    print(f"copied {source} -> {work_file}")
    print("size(bytes)=", work_file.stat().st_size)
```

## Cell 4.6 — Sửa lại `eval_Struct2GO2.py` nếu notebook đang dùng bản cũ

Chạy cell này nếu bạn thấy traceback vẫn trỏ tới `D:/CAFA6` hoặc line 248 cũ sau khi clone repo vào Kaggle.

```python
from pathlib import Path

eval_path = Path("/kaggle/working/CAFA6/eval_Struct2GO2.py")
text = eval_path.read_text(encoding="utf-8")
old = '    data_dir = os.environ.get("DATA_DIR", "D:/CAFA6")\n'
new = '    data_dir = _resolve_data_dir()\n'

if old in text and new not in text:
    text = text.replace(old, new)
    if 'def _resolve_data_dir() -> str:' not in text:
        anchor = '_ACS_FILES = {\n    "mf": "human_MF_ACS.json",\n    "cc": "human_CC_ACS.json",\n    "bp": "human_BP_ACS.json",\n}\n\n'
        insert = anchor + '''
def _resolve_data_dir() -> str:
    """Pick the first usable CAFA6 data root for local or Kaggle runs."""
    candidates = []
    env_data_dir = os.environ.get("DATA_DIR")
    if env_data_dir:
        candidates.append(Path(env_data_dir))
    candidates.append(Path(__file__).resolve().parent)
    candidates.append(Path.cwd())
    candidates.append(Path("D:/CAFA6"))

    for candidate in candidates:
        if (candidate / "divided_data").exists() and (candidate / "proceed_data").exists():
            return str(candidate)

    return env_data_dir or str(Path(__file__).resolve().parent)

'''
        text = text.replace(anchor, insert, 1)
    eval_path.write_text(text, encoding="utf-8")
    print("patched stale eval_Struct2GO2.py")
else:
    print("eval_Struct2GO2.py already current")
```

---

## Cell 5 — Train CC → MF → BP, lưu từng nhánh, cập nhật zip

~1–2 giờ tổng (BP chậm nhất). Sau mỗi nhánh zip được ghi lại tại `/kaggle/working/cafa6_output.zip`.

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py
```

**MF đã train xong** (chỉ chạy cc + bp, vẫn copy mf cũ vào zip nếu có):

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py --branches cc mf bp --skip mf
```

**OOM** → thêm `-batch_size 64`:

```python
!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py -batch_size 64
```

**Chỉ train, không eval** (nhanh hơn):

```python
!python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py --no-eval
```

**Eval riêng nhánh CC** (nếu muốn chạy trực tiếp thay vì qua `kaggle_run_branches.py`):

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/eval_Struct2GO2.py -branch cc --no-baseline-parity --split auto
```

### Train BP ~45 phút (gần baseline, cải hơn 4 epoch)

Giữ **batch 64**, **dropout 0.1**, **lr 1e-4** như baseline; chỉ rút **epoch** và dùng preset `--kaggle` (hid 256 — tránh 99 ngưỡng valid của `--baseline-parity` vì BP rất chậm).

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

!python /kaggle/working/CAFA6/train_Struct2GO2.py \
  -branch bp \
  --no-baseline-parity \
  --kaggle \
  -epochs 6 \
  -dropout 0.1 \
  -batch_size 64 \
  -validate_every 3 \
  --amp
```

Eval (99 ngưỡng trên test — giống baseline paper):

```python
!python /kaggle/working/CAFA6/eval_Struct2GO2.py -branch bp --baseline-parity --split test
```

| | Baseline paper | Run 4ep (30p) | **Đề xuất 6ep (45p)** |
|--|----------------|---------------|------------------------|
| Epoch | 20 | 4 | **6** |
| batch | 64 | 64 | **64** |
| dropout | 0.3 (paper) / **0.1** BP | 0.1 | **0.1** |
| hid | 512 | 256 | 256 |
| Validate train | mỗi 4 ep, 99 ngưỡng | cuối | **mỗi 3 ep**, 5 ngưỡng |

### Ablation fusion — 3 nhánh × 2 hướng (epoch/lr theo profile)

Script: [`scripts/run_fusion_ablation.py`](scripts/run_fusion_ablation.py)

| Hướng | PPI | Fusion |
|-------|:---:|:------:|
| `ppi_concat` | Có | concat |
| `no_ppi_attn` | Không | attention |

**Profiles** (cùng epoch/lr cho 2 hướng trên mỗi nhánh — so sánh công bằng):

| Profile | MF (ep/lr) | CC (ep/lr) | BP (ep/lr) | ~Tổng 6 run |
|---------|------------|------------|------------|-------------|
| `fast` | 8 / 1e-4 | 8 / 1e-4 | 6 / 1e-4 | ~2–3 h |
| `balanced` | 10 / 1e-4 | 10 / 1e-4 | 8 / 1e-4 | ~3–4 h |
| `quality` | 15 / 1e-4 | 12 / 1e-4 | 12 / 1e-4 | ~5–7 h |

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

# Mặc định: balanced (~3–4 h)
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --profile balanced

# Nhanh thử (~2 h)
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --profile fast

# Chất lượng cao hơn
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --profile quality

# Chỉ MF, override epoch
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --profile balanced --branches mf -epochs 12

# Chỉ eval
!python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --eval-only --profile balanced
```

Checkpoint: `bestmodel_{branch}_{ppi_concat|no_ppi_attn}_{batch}_{lr}_{dropout}.pkl`  
Kết quả: `log/fusion_ablation_summary.json`, `log/test_{branch}_{config}.log`

### Train lại MF (baseline-parity, tránh dropout 0.3)

Run MF **24/05** với `dropout=0.3` cho F-max test **0.018**. Code mới dùng **dropout 0.1** cho MF và eval ưu tiên `bestmodel_mf_64_0.0001_0.1.pkl`.

```python
%env DATA_DIR=/kaggle/working/CAFA6
%env DGL_CUDA=1

# Train (20 ep, hid 512, dropout 0.1 — không cần -dropout tay)
!python /kaggle/working/CAFA6/train_Struct2GO2.py -branch mf --baseline-parity -batch_size 48

# Eval test (99 ngưỡng, thresh JSON 0.71)
!python /kaggle/working/CAFA6/eval_Struct2GO2.py -branch mf --baseline-parity --split test
```

Nếu đã có checkpoint tốt **22/05** (valid F-max ~0.63), chỉ eval:

```python
!python /kaggle/working/CAFA6/eval_Struct2GO2.py -branch mf --baseline-parity --split test \
  -model_path /kaggle/working/CAFA6/save_models/bestmodel_mf_64_0.0001_0.1.pkl
```

Kỳ vọng test: F-max **~0.47+**, AUC **~0.86**, AUPR **~0.44** (log 23/05). `best_fscore` valid nên **> 0.3** sau train.

---

## Cell 6 — Lưu log + model + test result + zip (copy-paste)

Chạy **sau train/eval** (mỗi nhánh hoặc cả 3). Output nằm dưới `/kaggle/working/` → tab **Output** của notebook.

```python
import os
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"

# Đổi danh sách nhánh nếu chỉ mới xong 1–2 nhánh: ví dụ ["cc"] hoặc ["cc", "mf"]
BRANCHES = ["cc", "mf", "bp"]
ZIP_NAME = "cafa6_output.zip"

branch_args = " ".join(BRANCHES)
!python /kaggle/working/CAFA6/scripts/kaggle_save_results.py --branches {branch_args} --zip --zip-name {ZIP_NAME}
```

---

## Cell 7 — Kiểm tra đã lưu đủ chưa

```python
from pathlib import Path

zip_path = Path("/kaggle/working/cafa6_output.zip")
print("zip:", zip_path, f"({zip_path.stat().st_size / 1e6:.1f} MB)" if zip_path.is_file() else "MISSING")

for folder in ["log", "save_models", "test_result"]:
    p = Path("/kaggle/working") / folder
    print(f"\n=== {folder}/ ===")
    if not p.is_dir():
        print("  (trống)")
        continue
    for f in sorted(p.iterdir()):
        if f.is_file():
            size = f.stat().st_size
            unit = f"{size / 1e6:.1f} MB" if size > 1e6 else f"{size} B"
            print(f"  {f.name}  ({unit})")

# Metric test (nếu đã eval)
import re
for log in sorted(Path("/kaggle/working/log").glob("test_*.log")):
    text = log.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"f_score\s+([\d.]+).*?auc\s+([\d.]+).*?aupr\s+([\d.]+)", text, re.S)
    if hits:
        f, a, u = hits[-1]
        print(f"\n{log.name}: F-max={f}, AUC={a}, AUPR={u}")
```

---

## Cell 8b — Tải toàn bộ CC + MF về local (log + save_models + test_result)

Dán **một cell** sau train/eval **cc** và **mf**. Gom mọi file từ `CAFA6/` và `/kaggle/working/`, zip rồi tải.

```python
# === TẢI TOÀN BỘ CC + MF: log + save_models (best/final) + test_result ===
import os
import shutil
import zipfile
import base64
from pathlib import Path
from IPython.display import HTML, FileLink, display

DATA_DIR = Path("/kaggle/working/CAFA6")
os.environ["DATA_DIR"] = str(DATA_DIR)
BRANCHES = ["cc", "mf"]
MAX_AUTO_MB = 70
SPLIT_MB = 70

WORK = Path("/kaggle/working")
OUT_LOG = WORK / "log"
OUT_MODELS = WORK / "save_models"
OUT_TEST = WORK / "test_result"


def is_branch_file(name: str) -> bool:
    n = name.lower()
    return any(
        n == f"{b}.log" or n == f"test_{b}.log" or n.startswith(f"{b}_")
        or f"_{b}_" in n or f"bestmodel_{b}_" in n or f"final_{b}_" in n
        for b in BRANCHES
    )


def copy_if_newer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)


# --- 1) LOG (train + test) ---
log_src = [DATA_DIR / "log", WORK / "log"]
for br in BRANCHES:
    for name in (f"{br}.log", f"test_{br}.log"):
        for root in log_src:
            src = root / name
            if src.is_file():
                copy_if_newer(src, OUT_LOG / name)
                print(f"log: {src} -> {OUT_LOG / name}")

# --- 2) SAVE_MODELS (bestmodel_* + final_* + mọi *cc*/*mf* .pkl) ---
model_src = [DATA_DIR / "save_models", WORK / "save_models"]
seen = set()
for root in model_src:
    if not root.is_dir():
        continue
    for f in sorted(root.glob("*.pkl")):
        if not is_branch_file(f.name):
            continue
        key = f.name
        if key in seen:
            continue
        seen.add(key)
        copy_if_newer(f, OUT_MODELS / f.name)
        print(f"model: {f.name} ({f.stat().st_size/1e6:.1f} MB)")

# --- 3) TEST_RESULT (json, pkl pred, roc, test_*.log) ---
test_src = [DATA_DIR / "test_result", WORK / "test_result", OUT_LOG]
for br in BRANCHES:
    patterns = (
        f"{br}_result.json",
        f"{br}*_pred_actual.pkl",
        f"{br}_roc_curve.png",
        f"test_{br}.log",
    )
    for root in test_src:
        if not root.is_dir():
            continue
        for pat in patterns:
            for f in root.glob(pat):
                if not is_branch_file(f.name):
                    continue
                if f.name.startswith("test_"):
                    copy_if_newer(f, OUT_LOG / f.name)
                    copy_if_newer(f, OUT_TEST / f.name)
                else:
                    copy_if_newer(f, OUT_TEST / f.name)
                print(f"result: {f.name}")

# --- 4) Manifest ---
manifest = OUT_LOG / "_kaggle_manifest.txt"
lines = ["# CAFA6 export cc+mf", ""]
for folder, label in ((OUT_LOG, "log"), (OUT_MODELS, "save_models"), (OUT_TEST, "test_result")):
    lines.append(f"## {label}/")
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.is_file() and (label != "log" or is_branch_file(f.name) or f.name == "_kaggle_manifest.txt"):
                if label == "log" and f.name.endswith(".log") and not is_branch_file(f.name):
                    continue
                lines.append(f"  {f.name}  ({f.stat().st_size} B)")
    lines.append("")
manifest.write_text("\n".join(lines), encoding="utf-8")

# --- 5) Zip ---
def collect_pairs(branches_only: list[str] | None = None) -> list[tuple[Path, str]]:
    pairs = []
    for folder, prefix in ((OUT_LOG, "log"), (OUT_MODELS, "save_models"), (OUT_TEST, "test_result")):
        if not folder.is_dir():
            continue
        for f in sorted(folder.rglob("*")):
            if not f.is_file():
                continue
            if branches_only and not is_branch_file(f.name) and f.name != "_kaggle_manifest.txt":
                continue
            if branches_only is None and not is_branch_file(f.name) and f.name != "_kaggle_manifest.txt":
                continue
            pairs.append((f, f"{prefix}/{f.relative_to(folder).as_posix()}"))
    return pairs


def write_zip(path: Path, pairs: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f, arc in pairs:
            zf.write(f, arc)
    print(f"zip: {path.name} ({path.stat().st_size/1e6:.1f} MB, {len(pairs)} files)")


def split_zip(base: Path, pairs: list[tuple[Path, str]], max_mb: float) -> list[Path]:
    max_b = int(max_mb * 1e6)
    parts, cur, sz, idx = [], [], 0, 1

    def flush():
        nonlocal cur, sz, idx
        if not cur:
            return
        p = WORK / f"{base.stem}_part{idx}{base.suffix}"
        write_zip(p, cur)
        parts.append(p)
        idx += 1
        cur, sz = [], 0

    for item in pairs:
        fs = item[0].stat().st_size
        if cur and sz + fs > max_b:
            flush()
        cur.append(item)
        sz += fs
    flush()
    return parts


all_pairs = collect_pairs()
for br in BRANCHES:
    br_pairs = [(f, a) for f, a in all_pairs if br in f.name.lower()]
    if br_pairs:
        write_zip(WORK / f"cafa6_{br}_full.zip", br_pairs)

full = WORK / "cafa6_cc_mf_full.zip"
if sum(f.stat().st_size for f, _ in all_pairs) > SPLIT_MB * 1e6:
    zips = split_zip(full, all_pairs, SPLIT_MB)
else:
    write_zip(full, all_pairs)
    zips = [full]

# report (không model)
report_pairs = [(f, a) for f, a in all_pairs if "/save_models/" not in a]
write_zip(WORK / "cafa6_cc_mf_report.zip", report_pairs)


def autodl(path: Path, delay_ms: int = 0) -> None:
    if not path.is_file():
        return
    mb = path.stat().st_size / 1e6
    uid = "dl_" + path.name.replace(".", "_")
    if mb > MAX_AUTO_MB:
        display(FileLink(str(path), result_html_suffix="?download=1"))
        print(f"  ⚠ {path.name} ({mb:.0f} MB) → Save Version → Output")
        return
    b64 = base64.b64encode(path.read_bytes()).decode()
    js = f"setTimeout(function(){{document.getElementById('{uid}').click();}},{delay_ms});" if delay_ms else ""
    display(HTML(
        f'<a id="{uid}" download="{path.name}" href="data:application/zip;base64,{b64}"></a>'
        f"<script>{js}</script><p>⬇️ <b>{path.name}</b> ({mb:.1f} MB)</p>"
    ))


print("\n=== Danh sách đã gom ===")
for folder in (OUT_LOG, OUT_MODELS, OUT_TEST):
    print(f"\n{folder.name}/")
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.is_file() and (folder.name != "log" or is_branch_file(f.name)):
                print(f"  {f.name}  ({f.stat().st_size/1e6:.2f} MB)" if f.suffix == ".pkl" else f"  {f.name}")

print("\n=== Tải về ===")
autodl(WORK / "cafa6_cc_mf_report.zip", 0)
for i, z in enumerate([WORK / "cafa6_cc_full.zip", WORK / "cafa6_mf_full.zip"] + zips):
    autodl(z, 800 * (i + 1))

print("\nZip lớn: Save Version → Output → cafa6_cc_mf_full_part1.zip, part2.zip, …")
```

**Giải nén:** copy `log/`, `save_models/`, `test_result/` vào `D:\CAFA6\`.

---

## Cell 8c — Tải TOÀN BỘ bp + cc + mf (đúng danh sách log / save_models / test_result)

Dán **một cell** khi đã có đủ file dưới `/kaggle/working/log`, `save_models/`, `test_result/` (như output Cell 7).

```python
# === TẢI TOÀN BỘ: log/ + save_models/ + test_result/ (bp, cc, mf) ===
import os
import shutil
import zipfile
import base64
from pathlib import Path
from IPython.display import HTML, FileLink, display

DATA_DIR = Path("/kaggle/working/CAFA6")
os.environ["DATA_DIR"] = str(DATA_DIR)
WORK = Path("/kaggle/working")
OUT_LOG = WORK / "log"
OUT_MODELS = WORK / "save_models"
OUT_TEST = WORK / "test_result"
MAX_AUTO_MB = 70
SPLIT_MB = 70

FOLDERS = ("log", "save_models", "test_result")


def copy_if_newer(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if not dst.exists() or src.stat().st_mtime > dst.stat().st_mtime:
        shutil.copy2(src, dst)


# Gom từ CAFA6 + /kaggle/working (lấy bản mới nhất)
for sub in FOLDERS:
    out = WORK / sub
    for root in (DATA_DIR / sub, WORK / sub):
        if not root.is_dir():
            continue
        for f in root.rglob("*"):
            if f.is_file():
                copy_if_newer(f, out / f.name if f.parent == root else out / f.relative_to(root))


def all_files() -> list[tuple[Path, str]]:
    pairs = []
    for sub in FOLDERS:
        folder = WORK / sub
        if not folder.is_dir():
            continue
        for f in sorted(folder.rglob("*")):
            if f.is_file():
                pairs.append((f, f"{sub}/{f.relative_to(folder).as_posix()}"))
    return pairs


def write_zip(path: Path, pairs: list[tuple[Path, str]]) -> None:
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        for f, arc in pairs:
            zf.write(f, arc)
    mb = path.stat().st_size / 1e6
    print(f"✅ {path.name}  ({mb:.1f} MB, {len(pairs)} files)")


def split_zip(stem: str, pairs: list[tuple[Path, str]], max_mb: float) -> list[Path]:
    max_b = int(max_mb * 1e6)
    parts, cur, sz, idx = [], [], 0, 1

    def flush():
        nonlocal cur, sz, idx
        if not cur:
            return
        p = WORK / f"{stem}_part{idx}.zip"
        write_zip(p, cur)
        parts.append(p)
        idx += 1
        cur, sz = [], 0

    for item in pairs:
        fs = item[0].stat().st_size
        if cur and sz + fs > max_b:
            flush()
        cur.append(item)
        sz += fs
    flush()
    return parts


pairs = all_files()
if not pairs:
    raise FileNotFoundError("Trống — chạy Cell 6/7 trước hoặc kiểm tra /kaggle/working/log")

print("=== Đã gom (giống Cell 7) ===\n")
for sub in FOLDERS:
    folder = WORK / sub
    print(f"=== {sub}/ ===")
    if folder.is_dir():
        for f in sorted(folder.iterdir()):
            if f.is_file():
                sz = f.stat().st_size
                unit = f"{sz/1e6:.1f} MB" if sz > 1e6 else f"{sz} B"
                print(f"  {f.name}  ({unit})")
    print()

# Report nhẹ (không .pkl model)
report_pairs = [(f, a) for f, a in pairs if not a.startswith("save_models/")]
write_zip(WORK / "cafa6_all_report.zip", report_pairs)

# Full (~300MB) → chia part
total_mb = sum(f.stat().st_size for f, _ in pairs) / 1e6
print(f"Tổng: {total_mb:.0f} MB → zip chia {SPLIT_MB}MB/part\n")
if total_mb > SPLIT_MB:
    zips = split_zip("cafa6_all_full", pairs, SPLIT_MB)
else:
    p = WORK / "cafa6_all_full.zip"
    write_zip(p, pairs)
    zips = [p]


def autodl(path: Path, delay_ms: int = 0) -> None:
    if not path.is_file():
        return
    mb = path.stat().st_size / 1e6
    uid = "x" + path.name.replace(".", "_")
    if mb > MAX_AUTO_MB:
        display(FileLink(str(path), result_html_suffix="?download=1"))
        print(f"⚠ {path.name} ({mb:.0f} MB) — bấm link hoặc Save Version → Output")
        return
    b64 = base64.b64encode(path.read_bytes()).decode()
    js = f"setTimeout(function(){{document.getElementById('{uid}').click();}},{delay_ms});" if delay_ms else ""
    display(HTML(
        f'<a id="{uid}" download="{path.name}" href="data:application/zip;base64,{b64}"></a>'
        f"<script>{js}</script><p>⬇️ <b>{path.name}</b> ({mb:.1f} MB)</p>"
    ))


print("=== Tải về ===")
autodl(WORK / "cafa6_all_report.zip", 0)
for i, z in enumerate(zips):
    autodl(z, 1000 * (i + 1))

print("\n📦 Zip model lớn: Save Version (Commit) → Output → cafa6_all_full_part1.zip, part2.zip, …")
print("   Giải nén vào D:\\CAFA6\\ giữ nguyên log/ save_models/ test_result/")
```

---

## Cell 8 — Tải zip về máy (đủ 3 nhánh, file lớn chia part)

**Lưu ý:** `BRANCHES` phải gồm **cả 3** nhánh đã train/eval. Zip 292MB **không** tự tải base64 — dùng zip từng nhánh hoặc tab **Output**.

```python
import os
import zipfile
import base64
from pathlib import Path
from IPython.display import HTML, FileLink, display

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
BRANCHES = ["cc", "mf", "bp"]   # đủ 3 ontology — đừng chỉ ["cc"]
MAX_AUTO_MB = 75

branch_args = " ".join(BRANCHES)
!python /kaggle/working/CAFA6/scripts/kaggle_save_results.py \
  --branches {branch_args} --zip --per-branch-zip --split-mb {MAX_AUTO_MB}

working = Path("/kaggle/working")

def autodownload(path: Path, delay_ms: int = 0) -> None:
    if not path.is_file():
        print(f"MISSING: {path}")
        return
    mb = path.stat().st_size / 1e6
    if mb > MAX_AUTO_MB:
        display(FileLink(str(path), result_html_suffix="?download=1"))
        print(f"⚠️ {path.name} ({mb:.0f} MB) — bấm link hoặc: Save Version → Output → {path.name}")
        return
    b64 = base64.b64encode(path.read_bytes()).decode()
    delay = f"setTimeout(function(){{ document.getElementById('{path.stem}').click(); }}, {delay_ms});" if delay_ms else ""
    display(HTML(f"""
<a id="{path.stem}" download="{path.name}" href="data:application/zip;base64,{b64}"></a>
<script>{delay}</script>
<p>⬇️ {path.name} ({mb:.1f} MB)</p>
"""))

# 1) Report nhẹ (log + test_result + manifest) — luôn tải được
report = working / "cafa6_report.zip"
with zipfile.ZipFile(report, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ["log", "test_result"]:
        root = working / folder
        if root.is_dir():
            for f in root.rglob("*"):
                if f.is_file():
                    zf.write(f, f"{folder}/{f.relative_to(root).as_posix()}")
    manifest = working / "log" / "_kaggle_manifest.txt"
    if manifest.is_file():
        zf.write(manifest, "log/_kaggle_manifest.txt")
print(f"Report: {report} ({report.stat().st_size/1e6:.1f} MB)")
autodownload(report)

# 2) Từng nhánh (cc / mf / bp) — mở zip thấy rõ từng ontology
for i, br in enumerate(BRANCHES):
    p = working / f"cafa6_{br}.zip"
    if p.is_file():
        autodownload(p, delay_ms=400 * (i + 1))

# 3) Zip đầy đủ hoặc các part (nếu > MAX_AUTO_MB)
full = working / "cafa6_output.zip"
parts = sorted(working.glob("cafa6_output_part*.zip"))
if parts:
    for i, p in enumerate(parts):
        autodownload(p, delay_ms=600 * (i + 1))
elif full.is_file():
    autodownload(full, delay_ms=800)

print("\n=== Kiểm tra trong zip (phải có cả 3 nhánh) ===")
if report.is_file():
    with zipfile.ZipFile(report) as zf:
        for br in BRANCHES:
            hits = [n for n in zf.namelist() if br in n.lower()]
            print(f"  {br}: {len(hits)} files", hits[:5], "..." if len(hits) > 5 else "")
```

**Cách tải zip 292MB (đủ model cả 3 nhánh):**

1. **Save Version** (Commit) → chờ chạy xong → tab **Output** bên phải → tải `cafa6_output.zip` hoặc `cafa6_output_part1.zip`, `part2.zip`, …
2. Hoặc tải lần lượt `cafa6_cc.zip`, `cafa6_mf.zip`, `cafa6_bp.zip` (cell trên tự tải nếu mỗi file &lt; 75MB).

**Nội dung zip:**

```
log/_kaggle_manifest.txt   ← danh sách file theo nhánh
log/mf.log  log/cc.log  log/bp.log
log/test_mf.log  log/test_cc.log  log/test_bp.log
save_models/bestmodel_*.pkl  final_*.pkl
test_result/mf_result.json  cc_result.json  bp_result.json  ...
```
