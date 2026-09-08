# Kaggle T4 — Ablation fusion (2 hướng × 3 nhánh)

**Hướng 1:** PPI + concat (`--fusion concat`)  
**Hướng 2:** Không PPI + attention (`--no-ppi --fusion attention`)

Profile **`balanced`** (~3–4 giờ tổng 6 run): epoch/lr tối ưu thời gian, **cùng tham số** cho 2 hướng trên mỗi nhánh → so sánh công bằng.

| Nhánh | Epoch | LR | Dropout | Validate | Batch | ~Thời gian/hướng |
|-------|:-----:|:--:|:-------:|:--------:|:-----:|:----------------:|
| **mf** | 10 | 1e-4 | 0.1 | 4 | 64 | ~28 ph |
| **cc** | 10 | 1e-4 | 0.2 | 3 | 64 | ~30 ph |
| **bp** | 8 | 1e-4 | 0.1 | 3 | 64 | ~32 ph |

**Baseline paper (Table 1):** MF 0.302 / CC 0.531 / BP 0.303 (F-max)

> **Bắt buộc:** Chạy **Cell 0** trước — nếu thiếu `mf_train_dataset` train fail; `archive_run` có thể copy log cũ (F-max 0.018).

### Retrain B/C chi tiết từng nhánh

→ Xem **[kaggle_retrain_bc.md](kaggle_retrain_bc.md)** (Cell 0→8: CC → BP → MF).

### Chạy tự động 1 lệnh

```python
import os
os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
os.environ["DGL_CUDA"] = "1"
!python /kaggle/working/CAFA6/scripts/retrain_fusion_bc.py --profile match_a
```

---

## Cell 0 — Nối dữ liệu + kiểm tra (BẮT BUỘC trước train)

**Add Data** trên Kaggle: dataset `kaggle_data.zip` (từ `pack_for_kaggle.py`) hoặc dataset CAFA6 có `divided_data/` + `proceed_data/`.

```python
import os, shutil
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
DATA = Path("/kaggle/working/CAFA6")

# 1) Link proceed_data + divided_data từ /kaggle/input
!python {DATA}/scripts/kaggle_link_data.py --branches mf cc bp

# 2) Sửa file thiếu/hỏng — copy từ input
def repair_dataset(branch: str, split: str, min_mb: float = 1.0):
    name = f"{branch}_{split}_dataset"
    dst = DATA / "divided_data" / name
    if dst.is_file() and dst.stat().st_size >= min_mb * 1e6:
        print(f"  OK {name} ({dst.stat().st_size/1e6:.1f} MB)")
        return True
    cands = [p for p in Path("/kaggle/input").rglob(name) if p.is_file() and p.stat().st_size >= min_mb * 1e6]
    if not cands:
        print(f"  FAIL {name} — không có trong /kaggle/input")
        return False
    src = max(cands, key=lambda p: p.stat().st_size)
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"  FIXED {name} <- {src} ({dst.stat().st_size/1e6:.1f} MB)")
    return True

print("=== Repair divided_data ===")
ok = True
for branch in ("mf", "cc", "bp"):
    for split in ("train", "valid", "test"):
        if not repair_dataset(branch, split, min_mb=0.5 if split != "train" else 1.0):
            if split in ("train", "valid"):
                ok = False

assert (DATA / "proceed_data/ppi_graph_global").exists(), "Thiếu proceed_data — upload kaggle_data.zip"
assert ok, "Thiếu train/valid — Add Data rồi chạy lại Cell 0"

# 3) test thiếu → dùng valid cho eval (ghi nhớ)
for branch in ("mf", "cc", "bp"):
    test_p = DATA / f"divided_data/{branch}_test_dataset"
    valid_p = DATA / f"divided_data/{branch}_valid_dataset"
    if not test_p.is_file() and valid_p.is_file():
        shutil.copy2(valid_p, test_p)
        print(f"  test={branch}: copy valid → test (eval tạm)")

print("\n=== Checklist ===")
for branch in ("mf", "cc", "bp"):
    for split in ("train", "valid", "test"):
        p = DATA / f"divided_data/{branch}_{split}_dataset"
        print(f"  {p.name}: {'OK' if p.is_file() else 'MISSING'}")
```

---

## Cell 1 — Setup

```python
import os, re, shutil, json
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
os.environ["DGL_CUDA"] = "1"

DATA = Path("/kaggle/working/CAFA6")
(DATA / "log").mkdir(parents=True, exist_ok=True)
(DATA / "save_models").mkdir(parents=True, exist_ok=True)

print("DATA:", DATA)
```

*(Chạy cell link data / DGL patch của notebook chính trước nếu chưa có.)*

---

## Cell 2 — Hyperparameter + helper (lưu log / so baseline)

```python
# === PROFILE match_a — cùng epoch phương pháp A + pos-weight + ckpt combo (B/C) ===
# FAST=True → balanced ngắn (~2h)
FAST = False

HP = {
    "mf": dict(epochs=8 if FAST else 20, lr=1e-4, dropout=0.1, validate_every=4, batch=64),
    "cc": dict(epochs=8 if FAST else 20, lr=1e-4, dropout=0.2, validate_every=3, batch=64),
    "bp": dict(epochs=6 if FAST else 15, lr=1e-4, dropout=0.1, validate_every=3, batch=64),
}
TRAIN_EXTRA = ["--pos-weight", "--ckpt-metric", "combo"]  # cải thiện AUPR cho B/C

BASELINE_FMAX = {"mf": 0.302, "cc": 0.531, "bp": 0.303}
BEST_PRIOR = {"mf": 0.477, "cc": 0.585, "bp": 0.333}  # PPI+attention (cũ)

CONFIGS = {
    "ppi_concat":    ["--fusion", "concat"],
    "no_ppi_attn":   ["--no-ppi", "--fusion", "attention"],
}

def require_data(branch: str):
    p = DATA / f"divided_data/{branch}_train_dataset"
    if not p.is_file():
        raise FileNotFoundError(f"Thiếu {p} — chạy Cell 0 trước")

def model_path(branch: str):
    h = HP[branch]
    dr, lr, b = f"{h['dropout']:g}", f"{h['lr']:g}", h["batch"]
    return DATA / f"save_models/bestmodel_{branch}_{b}_{lr}_{dr}.pkl"

def eval_split(branch: str) -> str:
    if (DATA / f"divided_data/{branch}_test_dataset").is_file():
        return "test"
    if (DATA / f"divided_data/{branch}_valid_dataset").is_file():
        return "valid"
    raise FileNotFoundError(f"Thiếu test/valid cho {branch}")

def archive_run(branch: str, cfg: str):
    """Copy checkpoint + log — CHỈ sau train+eval thành công."""
    h = HP[branch]
    dr, lr, b = f"{h['dropout']:g}", f"{h['lr']:g}", h["batch"]
    ckpt_src = model_path(branch)
    ckpt_dst = DATA / f"save_models/bestmodel_{branch}_{cfg}_{b}_{lr}_{dr}.pkl"
    if not ckpt_src.is_file():
        raise FileNotFoundError(f"Không archive — thiếu {ckpt_src.name} (train fail?)")
    shutil.copy2(ckpt_src, ckpt_dst)
    print("  ckpt:", ckpt_dst.name)
    for src_name, dst_name in (
        (f"{branch}.log", f"train_{branch}_{cfg}.log"),
        (f"test_{branch}.log", f"test_{branch}_{cfg}.log"),
    ):
        src = DATA / "log" / src_name
        if src.is_file():
            shutil.copy2(src, DATA / "log" / dst_name)
            print("  log:", dst_name)

def parse_metrics(branch: str, cfg: str):
    p = DATA / "log" / f"test_{branch}_{cfg}.log"
    if not p.is_file():
        return None
    text = p.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(r"f_score\s+([\d.]+).*?auc\s+([\d.]+).*?aupr\s+([\d.]+)", text, re.S)
    if not hits:
        return None
    f, a, u = hits[-1]
    th = re.findall(r"thresh:\s*([\d.]+),\s*f_score", text)
    return dict(fmax=float(f), auc=float(a), aupr=float(u), thresh=float(th[-1]) if th else None)

def print_metrics(branch: str, cfg: str):
    m = parse_metrics(branch, cfg)
    if not m:
        print("  (chưa có metric — train/eval chưa xong hoặc chưa archive)")
        return m
    bl, pr = BASELINE_FMAX[branch], BEST_PRIOR[branch]
    print(f"  F-max={m['fmax']:.4f} ({m['fmax']-bl:+.3f} vs baseline, {m['fmax']-pr:+.3f} vs PPI+attn)")
    print(f"  AUC={m['auc']:.4f}  AUPR={m['aupr']:.4f}  thresh={m['thresh']}")
    if m["fmax"] < 0.05:
        print("  ⚠ F-max < 0.05 — có thể log cũ hoặc train fail, xóa log và chạy lại")
    return m

print("Profile:", "FAST" if FAST else "match_a")
for br, v in HP.items():
    print(f"  {br}: ep={v['epochs']} lr={v['lr']:.0e} dropout={v['dropout']} val_every={v['validate_every']}")
```

---

## Cell 3–8 — Train / Eval / Archive (mẫu — đổi `branch`, `cfg`)

**Mỗi cell:** đổi 2 dòng đầu → chạy. Không chạy `archive_run` nếu train báo lỗi.

| Cell | `branch`, `cfg` |
|------|-----------------|
| 3 | `mf`, `ppi_concat` |
| 4 | `mf`, `no_ppi_attn` |
| 5 | `cc`, `ppi_concat` |
| 6 | `cc`, `no_ppi_attn` |
| 7 | `bp`, `ppi_concat` |
| 8 | `bp`, `no_ppi_attn` |

```python
branch, cfg = "mf", "ppi_concat"   # <<< ĐỔI MỖI CELL
require_data(branch)
h = HP[branch]
ex = " ".join(CONFIGS[cfg])
split = eval_split(branch)

print(f"=== TRAIN {branch} / {cfg} | ep={h['epochs']} lr={h['lr']} ===")
!python {DATA}/train_Struct2GO2.py -branch {branch} --no-baseline-parity --kaggle \
  -epochs {h['epochs']} -dropout {h['dropout']} -batch_size {h['batch']} \
  -learningrate {h['lr']} -validate_every {h['validate_every']} --amp \
  {' '.join(TRAIN_EXTRA)} {ex}

mp = model_path(branch)
assert mp.is_file(), f"Train FAIL — không có {mp.name}. Chạy Cell 0, không archive."

print(f"=== EVAL {branch} / {cfg} split={split} ===")
!python {DATA}/eval_Struct2GO2.py -branch {branch} --split {split} -model_path {mp} {ex}

archive_run(branch, cfg)
print_metrics(branch, cfg)
```

---

## Cell 9 — Bảng tổng hợp + so baseline (chạy sau cả 6 cell)

```python
rows = []
for branch in ("mf", "cc", "bp"):
    for cfg in ("ppi_concat", "no_ppi_attn"):
        m = parse_metrics(branch, cfg)
        h = HP[branch]
        rows.append({
            "branch": branch, "config": cfg,
            "epochs": h["epochs"], "lr": h["lr"],
            **(m or {}),
        })

print(f"{'br':<4} {'config':<14} {'ep':>3} {'F-max':>7} {'AUC':>7} {'AUPR':>7}  vs baseline  vs PPI+attn")
print("-" * 72)
for r in rows:
    if "fmax" not in r:
        print(f"{r['branch']:<4} {r['config']:<14}  (chưa có log)")
        continue
    bl = BASELINE_FMAX[r["branch"]]
    pr = BEST_PRIOR[r["branch"]]
    print(
        f"{r['branch']:<4} {r['config']:<14} {r['epochs']:>3} "
        f"{r['fmax']:>7.4f} {r['auc']:>7.4f} {r['aupr']:>7.4f}  "
        f"{r['fmax']-bl:>+7.3f}      {r['fmax']-pr:>+7.3f}"
    )

# So 2 hướng trong từng nhánh
for branch in ("mf", "cc", "bp"):
    sub = [r for r in rows if r["branch"] == branch and "fmax" in r]
    if len(sub) >= 2:
        w = max(sub, key=lambda x: x["fmax"])
        print(f"\n{branch.upper()} thắng ablation: {w['config']} (F-max={w['fmax']:.4f})")

out = DATA / "log/fusion_ablation_summary.json"
out.write_text(json.dumps({"profile": "fast" if FAST else "balanced", "hp": HP, "results": rows}, indent=2))
print("\nSaved", out)
```

---

## Cell 10 — Tải log + checkpoint về máy

```python
import zipfile
from IPython.display import FileLink, display

zip_path = Path("/kaggle/working/fusion_ablation_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ("log", "save_models"):
        root = DATA / folder
        if not root.is_dir():
            continue
        for f in root.iterdir():
            if f.is_file() and ("ppi_concat" in f.name or "no_ppi_attn" in f.name
                                or f.name == "fusion_ablation_summary.json"):
                zf.write(f, f"{folder}/{f.name}")
print(f"Zip: {zip_path} ({zip_path.stat().st_size/1e6:.1f} MB)")
display(FileLink(str(zip_path), result_html_suffix="?download=1"))
print("Hoặc: Save Version → Output → fusion_ablation_results.zip")
```

---

## Cell 0b — Xóa log/archive sai (F-max 0.018 từ run cũ)

Chạy nếu đã archive nhầm khi train fail:

```python
from pathlib import Path
DATA = Path("/kaggle/working/CAFA6")
for p in list((DATA/"log").glob("test_mf_*")) + list((DATA/"save_models").glob("*mf_ppi*")):
    if "ppi_concat" in p.name or "no_ppi" in p.name:
        p.unlink()
        print("removed", p.name)
print("Xong — chạy lại Cell 0 rồi Cell 3")
```

---

## Lưu ý T4

1. **OOM** → trong Cell 2 đổi `batch=48` hoặc `32` cho nhánh lỗi.
2. **Không** dùng `--baseline-parity` khi train (chậm ~4h/nhánh BP).
3. Eval `--split test`; metric so baseline lấy từ **F-max trong log** (quét ngưỡng).
4. File gốc `test_mf.log` bị ghi đè — luôn dùng bản đã archive `test_mf_ppi_concat.log`.
5. Đối chiếu thêm **PPI+attention cũ:** MF 0.477, CC 0.585, BP 0.333 (cột `vs PPI+attn` trong Cell 9).
