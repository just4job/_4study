#!/usr/bin/env python3
"""Train + test lại phương pháp B (PPI+concat) và C (no-PPI+attention).

Dùng sau khi cập nhật ConcatFusion, --pos-weight, --ckpt-metric combo.

Mặc định profile ``match_a`` (cùng epoch phương pháp A: mf/cc 20, bp 15).

Usage (Kaggle):
  %env DATA_DIR=/kaggle/working/CAFA6
  %env DGL_CUDA=1
  !python /kaggle/working/CAFA6/scripts/retrain_fusion_bc.py

  !python .../retrain_fusion_bc.py --branches mf --configs ppi_concat
  !python .../retrain_fusion_bc.py --eval-only --branches cc

Usage (local):
  python scripts/retrain_fusion_bc.py --data-dir D:/CAFA6
  python scripts/retrain_fusion_bc.py --branches mf cc bp --profile match_a
"""
from __future__ import annotations

import argparse
import os
import pickle
import shutil
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.run_fusion_ablation import (  # noqa: E402
    BASELINE_REPRO,
    BEST_PRIOR,
    FUSION_CONFIGS,
    FusionConfig,
    PROFILES,
    _compare_branch,
    _env,
    _parse_test_log,
    _print_profile_table,
    _resolve_hp,
    eval_one,
    train_one,
)


def _verify_branch_data(data_dir: Path, branch: str) -> None:
    """Đảm bảo train/valid pickle đọc được trước khi train."""
    import __main__

    from data_processing.divide_data import MyDataSet

    __main__.MyDataSet = MyDataSet

    for split in ("train", "valid"):
        path = data_dir / "divided_data" / f"{branch}_{split}_dataset"
        if not path.is_file():
            raise FileNotFoundError(f"Thiếu {path}")
        if path.stat().st_size < 500_000:
            raise RuntimeError(f"{path.name} quá nhỏ ({path.stat().st_size} B) — có thể hỏng")
        with open(path, "rb") as handle:
            ds = pickle.load(handle)
        n = len(ds) if hasattr(ds, "__len__") else "?"
        print(f"  OK {path.name} ({path.stat().st_size / 1e6:.1f} MB, n={n})")


def _archive_logs(data_dir: Path, branch: str, cfg: FusionConfig) -> None:
    log_dir = data_dir / "log"
    log_dir.mkdir(parents=True, exist_ok=True)
    for src_name, dst_name in (
        (f"{branch}.log", f"train_{branch}_{cfg.name}.log"),
        (f"test_{branch}.log", f"test_{branch}_{cfg.name}.log"),
    ):
        src = log_dir / src_name
        if src.is_file():
            shutil.copy2(src, log_dir / dst_name)
            print(f"  [log] {dst_name}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Retrain + test B (ppi_concat) và C (no_ppi_attn)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None, help="CAFA6 root (divided_data, proceed_data)")
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["mf", "cc", "bp"],
        choices=["mf", "cc", "bp"],
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[c.name for c in FUSION_CONFIGS],
        choices=[c.name for c in FUSION_CONFIGS],
        help="ppi_concat | no_ppi_attn",
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="match_a",
        help="match_a = cùng epoch phương pháp A (khuyến nghị)",
    )
    parser.add_argument("-epochs", type=int, default=None)
    parser.add_argument("-learningrate", type=float, default=None)
    parser.add_argument("-batch_size", type=int, default=None)
    parser.add_argument("-validate_every", type=int, default=None)
    parser.add_argument("--skip-data-check", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--baseline-eval",
        action="store_true",
        help="Eval với --baseline-parity (mặc định: --no-baseline-parity)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", str(REPO)))
    cwd = data_dir if (data_dir / "train_Struct2GO2.py").is_file() else REPO
    env = _env(data_dir)
    os.environ.update(env)

    selected = [c for c in FUSION_CONFIGS if c.name in args.configs]
    results: list[dict] = []
    failed: list[str] = []
    t0 = time.time()

    print("=" * 60)
    print("  RETRAIN fusion B/C — ConcatFusion + pos-weight + ckpt combo")
    print("=" * 60)
    print("DATA_DIR:", data_dir)
    print("Profile:", args.profile)
    _print_profile_table(args.profile, args.branches)
    print("Configs:", [c.name for c in selected])

    if not args.skip_data_check and not args.eval_only:
        print("\n=== Kiểm tra divided_data ===")
        try:
            for branch in args.branches:
                print(f"[{branch}]")
                _verify_branch_data(data_dir, branch)
        except Exception as exc:
            print(f"\n[FAIL] Data check: {exc}", file=sys.stderr)
            if "mf_train" in str(exc).lower() or args.branches == ["mf"]:
                print(
                    "\nGợi ý MF: upload kaggle_mf.zip (422 labels) từ pack_for_kaggle.py, "
                    "rồi chạy repair_mf_train.py --train-only",
                    file=sys.stderr,
                )
            return 1
        ppi = data_dir / "proceed_data/ppi_graph_global"
        if not ppi.exists():
            print(f"[FAIL] Thiếu {ppi}", file=sys.stderr)
            return 1
        print("Data OK\n")

    for branch in args.branches:
        if branch == "mf" and not args.eval_only:
            try:
                from scripts.repair_mf_train import mf_train_ok, repair_mf_train

                if mf_train_ok(data_dir):
                    print("[mf] mf_train OK — bỏ qua repair")
                else:
                    repair_mf_train(
                        data_dir,
                        Path("/kaggle/input"),
                        train_only=True,
                    )
            except Exception as exc:
                print(f"[FAIL] MF data chưa sẵn sàng: {exc}", file=sys.stderr)
                print(
                    "\nCần upload mf_train 422 labels (pack_for_kaggle.py --branch mf --splits train). "
                    "KHÔNG dùng mf-train1 (5136 labels).",
                    file=sys.stderr,
                )
                return 1

        hp = _resolve_hp(
            branch,
            args.profile,
            args.epochs,
            args.learningrate,
            args.batch_size,
            args.validate_every,
        )

        for cfg in selected:
            tag = f"{branch}/{cfg.name}"
            print("\n" + "=" * 60)
            print(f"=== {tag} | ep={hp.epochs} lr={hp.learningrate:.0e} ===")
            print("=" * 60)

            from scripts.run_fusion_ablation import _ckpt_tag

            ckpt = data_dir / "save_models" / _ckpt_tag(branch, cfg, hp)

            try:
                if not args.eval_only:
                    ckpt = train_one(branch, cfg, hp, cwd, data_dir, env)
                    _archive_logs(data_dir, branch, cfg)

                if not args.train_only:
                    if not ckpt.is_file():
                        raise FileNotFoundError(f"Thiếu checkpoint: {ckpt}")
                    rc = eval_one(branch, cfg, ckpt, cwd, env, args.baseline_eval)
                    if rc != 0:
                        raise RuntimeError(f"Eval exit {rc}")
                    _archive_logs(data_dir, branch, cfg)

                test_log = data_dir / "log" / f"test_{branch}_{cfg.name}.log"
                metrics = _parse_test_log(test_log) or _parse_test_log(
                    data_dir / "log" / f"test_{branch}.log"
                )
                row = {
                    "branch": branch,
                    "config": cfg.name,
                    "epochs": hp.epochs,
                    "lr": hp.learningrate,
                    "checkpoint": ckpt.name,
                    **(metrics or {}),
                }
                results.append(row)
                if metrics:
                    bl = BASELINE_REPRO[branch]["fmax"]
                    pr = BEST_PRIOR[branch]["fmax"]
                    print(
                        f"[metric] F-max={metrics['fmax']:.4f} "
                        f"({metrics['fmax']-bl:+.3f} vs baseline, {metrics['fmax']-pr:+.3f} vs PPI+attn)  "
                        f"AUC={metrics['auc']:.4f}  AUPR={metrics['aupr']:.4f}"
                    )
            except Exception as exc:
                print(f"[FAIL] {tag}: {exc}", file=sys.stderr)
                failed.append(tag)

    elapsed = (time.time() - t0) / 60
    print("\n" + "=" * 72)
    print("=== SUMMARY (B/C retrain) ===")
    print("=" * 72)
    print(f"{'br':<4} {'config':<14} {'ep':>3} {'F-max':>7} {'AUC':>7} {'AUPR':>7}")
    print("-" * 48)
    for r in results:
        fmax = f"{r.get('fmax', 0):.4f}" if "fmax" in r else "   -   "
        auc = f"{r.get('auc', 0):.4f}" if "auc" in r else "   -   "
        aupr = f"{r.get('aupr', 0):.4f}" if "aupr" in r else "   -   "
        print(
            f"{r['branch']:<4} {r['config']:<14} {r['epochs']:>3} "
            f"{fmax:>7} {auc:>7} {aupr:>7}"
        )

    for branch in args.branches:
        _compare_branch(
            [r for r in results if r["branch"] == branch and "fmax" in r],
            branch,
        )

    summary_path = data_dir / "log" / "retrain_fusion_bc_summary.json"
    import json

    summary_path.write_text(
        json.dumps(
            {
                "profile": args.profile,
                "elapsed_minutes": round(elapsed, 1),
                "results": results,
                "baseline": BASELINE_REPRO,
                "best_prior_ppi_attn": BEST_PRIOR,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nWrote {summary_path}  ({elapsed:.1f} min)")

    if failed:
        print("Failed:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
