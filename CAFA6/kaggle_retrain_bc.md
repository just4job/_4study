# Kaggle — Retrain B/C từng nhánh (kết quả tốt nhất)

**B** = PPI + concat (`--fusion concat`)  
**C** = không PPI + attention (`--no-ppi --fusion attention`)

Profile **`match_a`**: mf/cc **20 epoch**, bp **15 epoch** + `--pos-weight` + `--ckpt-metric combo` + **ConcatFusion** mới.

| Cell | Nội dung | ~Thời gian |
|------|----------|:----------:|
| 0 | Data + sửa `mf_train` | 2 ph |
| 1 | Setup | — |
| 2 | Helper + `run_bc()` | — |
| **3** | **CC / B** ppi_concat | ~38 ph |
| **4** | **CC / C** no_ppi_attn | ~38 ph |
| **5** | **BP / B** ppi_concat | ~32 ph |
| **6** | **BP / C** no_ppi_attn | ~32 ph |
| **7** | **MF / B** ppi_concat | ~45 ph |
| **8** | **MF / C** no_ppi_attn | ~45 ph |
| 9 | Bảng tổng hợp | — |
| 10 | Zip tải về | — |

**Thứ tự khuyến nghị:** Cell 0→1→2 → **3→4** (CC) → **5→6** (BP) → **7→8** (MF).

---

## Cell 0 — Data (MF ablation: cần mf_train 422 labels)

**Add Data (notebook):**
- `final-data` — CC/BP/MF valid + proceed_data (bắt buộc)
- **`cafa6-mf-train`** — zip từ `pack_for_kaggle.py --branch mf --splits train` (~3.5 GB, **422 labels**)
- **KHÔNG** add `mf-train1` (5136 labels — sẽ làm F-max ~0.002)

**Trên máy local (trước khi upload):**
```bash
cd D:/CAFA6
python pack_for_kaggle.py --branch mf --splits train --output kaggle_mf_train.zip
# Kiểm tra: mf_train labels=422, n≈12782
```
Upload `kaggle_mf_train.zip` lên Kaggle → New Dataset → tên vd. `cafa6-mf-train`.

```python
import os, shutil
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
DATA = Path("/kaggle/working/CAFA6")

# CC/BP + mf_valid từ final-data (KHÔNG repair mf-train1)
!python {DATA}/scripts/kaggle_link_data.py --branches mf cc bp

# Chỉ copy mf_train 422 labels từ dataset upload mới
!python {DATA}/scripts/repair_mf_train.py --train-only --require-label-dim 422

# test thiếu → copy valid
for br in ("mf", "cc", "bp"):
    t, v = DATA/f"divided_data/{br}_test_dataset", DATA/f"divided_data/{br}_valid_dataset"
    if not t.is_file() and v.is_file():
        shutil.copy2(v, t)
        print(f"test {br}: copy valid")

!python {DATA}/scripts/diag_mf_train.py

print("\n=== Checklist ===")
for br in ("mf", "cc", "bp"):
    for sp in ("train", "valid", "test"):
        p = DATA / f"divided_data/{br}_{sp}_dataset"
        mb = p.stat().st_size/1e6 if p.is_file() else 0
        print(f"  {p.name}: {'OK' if p.is_file() else 'MISSING'} ({mb:.1f} MB)")
assert (DATA/"proceed_data/ppi_graph_global").exists()
print("DATA OK — mf_train phải labels=422 khớp mf_valid")
```

---

## Cell 1 — Setup

```python
import os, re, shutil, json, sys, subprocess
from pathlib import Path

os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
os.environ["DGL_CUDA"] = "1"

DATA = Path("/kaggle/working/CAFA6")
(DATA/"log").mkdir(parents=True, exist_ok=True)
(DATA/"save_models").mkdir(parents=True, exist_ok=True)
print("DATA:", DATA)
```

---

## Cell 2 — Helper + hàm `run_bc()` (gọi mỗi nhánh)

```python
import sys, subprocess

# match_a — cùng budget epoch phương pháp A
HP = {
    "mf": dict(epochs=20, lr=1e-4, dropout=0.1, validate_every=4, batch=64),
    "cc": dict(epochs=20, lr=1e-4, dropout=0.2, validate_every=3, batch=64),
    "bp": dict(epochs=15, lr=1e-4, dropout=0.1, validate_every=3, batch=64),
}
TRAIN_EXTRA = ["--pos-weight", "--ckpt-metric", "combo"]

CONFIGS = {
    "ppi_concat":    ["--fusion", "concat"],           # B
    "no_ppi_attn":   ["--no-ppi", "--fusion", "attention"],  # C
}

BASELINE = {"mf": 0.302, "cc": 0.531, "bp": 0.303}
PRIOR_A  = {"mf": 0.477, "cc": 0.585, "bp": 0.333}

def model_path(branch):
    h = HP[branch]
    return DATA / f"save_models/bestmodel_{branch}_{h['batch']}_{h['lr']:g}_{h['dropout']:g}.pkl"

def eval_split(branch):
    if (DATA/f"divided_data/{branch}_test_dataset").is_file():
        return "test"
    return "valid"

def archive_run(branch, cfg):
    h = HP[branch]
    src = model_path(branch)
    dst = DATA/f"save_models/bestmodel_{branch}_{cfg}_{h['batch']}_{h['lr']:g}_{h['dropout']:g}.pkl"
    if not src.is_file():
        raise FileNotFoundError(f"Train fail — thiếu {src.name}")
    shutil.copy2(src, dst)
    for s, d in [(f"{branch}.log", f"train_{branch}_{cfg}.log"),
                 (f"test_{branch}.log", f"test_{branch}_{cfg}.log")]:
        p = DATA/"log"/s
        if p.is_file():
            shutil.copy2(p, DATA/"log"/d)
    print("  archived:", dst.name)

def print_metrics(branch, cfg):
    p = DATA/f"log/test_{branch}_{cfg}.log"
    if not p.is_file():
        print("  (chưa có log)"); return
    t = p.read_text(errors="replace")
    hits = re.findall(r"f_score\s+([\d.]+).*?auc\s+([\d.]+).*?aupr\s+([\d.]+)", t, re.S)
    if not hits:
        print("  (chưa parse được metric)"); return
    f, a, u = hits[-1]
    print(f"  F-max={float(f):.4f}  AUC={float(a):.4f}  AUPR={float(u):.4f}")
    print(f"  vs baseline {BASELINE[branch]:.3f}: {float(f)-BASELINE[branch]:+.3f}")
    print(f"  vs PPI+attn A {PRIOR_A[branch]:.3f}: {float(f)-PRIOR_A[branch]:+.3f}")

import subprocess

def _run(cmd: list[str], step: str):
    print(">>>", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(DATA))
    if rc != 0:
        raise RuntimeError(f"{step} FAIL (exit {rc})")

def ensure_mf_train():
    """Gọi TRƯỚC mọi train MF — chỉ copy mf_train 422 labels, giữ valid final-data."""
    from scripts.repair_mf_train import mf_train_ok, repair_mf_train
    if mf_train_ok(DATA):
        print("  mf_train OK")
        return
    _run([
        sys.executable, str(DATA/"scripts/repair_mf_train.py"),
        "--train-only", "--require-label-dim", "422",
    ], "repair_mf_train")

def run_bc(branch: str, cfg: str):
    """Train → eval → archive một run B hoặc C."""
    assert cfg in CONFIGS, f"cfg phải là ppi_concat hoặc no_ppi_attn, got {cfg}"
    if branch == "mf":
        ensure_mf_train()

    h = HP[branch]
    ex = CONFIGS[cfg]
    split = eval_split(branch)
    train_p = DATA/f"divided_data/{branch}_train_dataset"
    if not train_p.is_file() or train_p.stat().st_size < 1e6:
        raise FileNotFoundError(f"Thiếu/hỏng {train_p} — chạy Cell 0")

    label = "B PPI+concat" if cfg == "ppi_concat" else "C no-PPI+attn"
    print(f"\n{'='*60}\n  {branch.upper()} / {label}  ep={h['epochs']}  split={split}\n{'='*60}")

    train_cmd = [
        sys.executable, str(DATA/"train_Struct2GO2.py"),
        "-branch", branch, "--no-baseline-parity", "--kaggle",
        "-epochs", str(h["epochs"]), "-dropout", str(h["dropout"]),
        "-batch_size", str(h["batch"]), "-learningrate", str(h["lr"]),
        "-validate_every", str(h["validate_every"]), "--amp",
        *TRAIN_EXTRA, *ex,
    ]
    _run(train_cmd, "TRAIN")

    mp = model_path(branch)
    if not mp.is_file():
        raise RuntimeError(f"TRAIN FAIL — không có {mp.name}")

    tail = (DATA/f"log/{branch}.log").read_text(errors="replace")[-4000:]
    if "saved checkpoint" not in tail:
        raise RuntimeError("Log không có 'saved checkpoint' — không eval (checkpoint cũ?)")

    eval_cmd = [
        sys.executable, str(DATA/"eval_Struct2GO2.py"),
        "-branch", branch, "--split", split,
        "-model_path", str(mp), "--no-baseline-parity", *ex,
    ]
    _run(eval_cmd, "EVAL")

    archive_run(branch, cfg)
    print_metrics(branch, cfg)
    print("DONE", branch, cfg)

print("Helper OK — gọi run_bc('cc', 'ppi_concat') hoặc chạy Cell 3–8")
for br, v in HP.items():
    print(f"  {br}: ep={v['epochs']} dropout={v['dropout']}")
```

---

## Cell 3 — CC / B (ppi_concat) ~38 ph

```python
run_bc("cc", "ppi_concat")
```

---

## Cell 4 — CC / C (no_ppi_attn) ~38 ph

```python
run_bc("cc", "no_ppi_attn")
```

---

## Cell 5 — BP / B (ppi_concat) ~32 ph

```python
run_bc("bp", "ppi_concat")
```

---

## Cell 6 — BP / C (no_ppi_attn) ~32 ph

```python
run_bc("bp", "no_ppi_attn")
```

---

## Cell 7 — MF / B (ppi_concat) ~45 ph

*`run_bc` tự gọi `repair_mf_train` trước train.*

```python
run_bc("mf", "ppi_concat")
```

---

## Cell 8 — MF / C (no_ppi_attn) ~45 ph

```python
run_bc("mf", "no_ppi_attn")
```

---

## Cell 9 — Bảng tổng hợp B/C vs baseline & phương pháp A

```python
rows = []
for branch in ("mf", "cc", "bp"):
    for cfg in ("ppi_concat", "no_ppi_attn"):
        p = DATA/f"log/test_{branch}_{cfg}.log"
        if not p.is_file():
            rows.append({"branch": branch, "cfg": cfg, "fmax": None})
            continue
        t = p.read_text(errors="replace")
        hits = re.findall(r"f_score\s+([\d.]+).*?auc\s+([\d.]+).*?aupr\s+([\d.]+)", t, re.S)
        if hits:
            f, a, u = hits[-1]
            rows.append({"branch": branch, "cfg": cfg,
                         "fmax": float(f), "auc": float(a), "aupr": float(u)})

print(f"{'br':<4} {'config':<14} {'F-max':>7} {'AUC':>7} {'AUPR':>7}  vs_bl  vs_A")
print("-" * 62)
for r in rows:
    if r["fmax"] is None:
        print(f"{r['branch']:<4} {r['cfg']:<14}  (chưa chạy)")
        continue
    bl, pa = BASELINE[r["branch"]], PRIOR_A[r["branch"]]
    print(f"{r['branch']:<4} {r['cfg']:<14} {r['fmax']:>7.4f} {r['auc']:>7.4f} {r['aupr']:>7.4f}  "
          f"{r['fmax']-bl:>+6.3f} {r['fmax']-pa:>+6.3f}")

for branch in ("mf", "cc", "bp"):
    sub = [r for r in rows if r["branch"]==branch and r.get("fmax")]
    if len(sub) >= 2:
        w = max(sub, key=lambda x: x["fmax"])
        print(f"\n{branch.upper()} tốt nhất B/C: {w['cfg']}  F-max={w['fmax']:.4f}")

out = DATA/"log/retrain_bc_summary.json"
out.write_text(json.dumps({"hp": HP, "results": rows}, indent=2))
print("\nSaved", out)
```

---

## Cell 10 — Zip checkpoint + log

```python
import zipfile
from IPython.display import FileLink, display

zip_path = Path("/kaggle/working/retrain_bc_results.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
    for folder in ("log", "save_models"):
        for f in (DATA/folder).iterdir():
            if f.is_file() and ("ppi_concat" in f.name or "no_ppi_attn" in f.name):
                zf.write(f, f"{folder}/{f.name}")
print("Wrote", zip_path, f"({zip_path.stat().st_size/1e6:.1f} MB)")
display(FileLink(str(zip_path)))
```

---

## Cell thay thế — 1 lệnh tự động (sau Cell 0)

```python
import os
os.environ["DATA_DIR"] = "/kaggle/working/CAFA6"
os.environ["DGL_CUDA"] = "1"
!python /kaggle/working/CAFA6/scripts/retrain_fusion_bc.py --profile match_a
```

---

## Troubleshooting

| Triệu chứng | Xử lý |
|-------------|--------|
| MF F-max ~0.002 | train 5136 vs valid 422 — upload mf_train 422, `--train-only` |
| `repair_mf_train` FAIL | Thiếu dataset cafa6-mf-train; không dùng mf-train1 |
| `saved checkpoint` không có | Train fail — không archive |
| OOM / 20 phút im lặng | Restart session; `HP['mf']['batch']=48`; `diag_mf_train.py` |
| Eval vocab truncate WARN | Bình thường — vẫn eval được |

## Kỳ vọng sau retrain (match_a)

| Nhánh | B ppi_concat | C no_ppi_attn |
|-------|:------------:|:-------------:|
| MF | F-max ~0.40+ | ~0.38+ |
| CC | ~0.55+ | ~0.57+ |
| BP | ~0.30+ | ~0.30+ |
