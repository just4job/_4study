#!/usr/bin/env python3
"""Kiểm tra vì sao train MF không có output (load pickle / RAM / log)."""
from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA = Path(os.environ.get("DATA_DIR", REPO))


def main() -> int:
    print("=== diag_mf_train ===", flush=True)
    print(f"DATA_DIR: {DATA}", flush=True)

    train_p = DATA / "divided_data/mf_train_dataset"
    valid_p = DATA / "divided_data/mf_valid_dataset"
    log_p = DATA / "log/mf.log"

    for p in (train_p, valid_p):
        if not p.is_file():
            print(f"MISSING {p}")
            return 1
        print(f"  {p.name}: {p.stat().st_size/1e6:.1f} MB", flush=True)

    if log_p.is_file():
        print(f"\nlog/mf.log ({log_p.stat().st_size} B), last 15 lines:", flush=True)
        lines = log_p.read_text(errors="replace").splitlines()
        for line in lines[-15:]:
            print(" ", line)
    else:
        print("\nlog/mf.log: (chưa có — train chưa start hoặc process chết sớm)", flush=True)

    if str(REPO) not in sys.path:
        sys.path.insert(0, str(REPO))
    import __main__

    from data_processing.divide_data import MyDataSet

    __main__.MyDataSet = MyDataSet

    print("\n=== label_dim ===", flush=True)
    try:
        with open(train_p, "rb") as f:
            ds_train = pickle.load(f)
        with open(valid_p, "rb") as f:
            ds_valid = pickle.load(f)
        import numpy as np

        dim_train = int(np.asarray(ds_train[0][2]).reshape(-1).shape[0])
        dim_valid = int(np.asarray(ds_valid[0][2]).reshape(-1).shape[0])
        print(f"  mf_train: n={len(ds_train)} labels={dim_train}", flush=True)
        print(f"  mf_valid: n={len(ds_valid)} labels={dim_valid}", flush=True)
        if dim_train != dim_valid:
            print(
                f"  [FAIL] train ({dim_train}) != valid ({dim_valid}) — "
                "F-max sẽ ~0.002. Upload mf_train 422 labels.",
                flush=True,
            )
            return 1
        print("  label_dim OK (422 expected for CAFA6)", flush=True)
    except Exception as exc:
        print(f"  label check FAIL: {exc}", flush=True)
        return 1

    print("\n=== Thử load mf_train (đo thời gian) ===", flush=True)
    t0 = time.time()
    try:
        ds = ds_train
        print(f"  OK n={len(ds)} trong {time.time()-t0:.1f}s", flush=True)
    except Exception as exc:
        print(f"  FAIL sau {time.time()-t0:.1f}s: {exc}", flush=True)
        return 1

    print("\nGợi ý: nếu load >120s hoặc RAM đầy → Restart session, batch=48, num_workers=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
