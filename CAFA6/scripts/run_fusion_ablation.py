#!/usr/bin/env python3
"""Train + eval ablation fusion — 3 nhánh × 4 hướng, epoch/lr tối ưu theo profile.

Hướng 1: PPI + concat                    (--fusion concat)
Hướng 2: không PPI + attention            (--no-ppi --fusion attention)
Hướng 3: PPI + attention 1 CHIỀU          (--fusion attention)      — struct+seq
         làm Query cố định, PPI chỉ là Key/Value tĩnh (không được cập nhật).
Hướng 4: PPI + attention 2 CHIỀU          (--fusion bi_attention)   — struct/seq/PPI
         gộp thành 1 chuỗi token, self-attention đối xứng: PPI cũng được struct/seq
         cập nhật ngược lại (xem model/layer.py:BidirectionalCrossAttention).

Cả 4 hướng dùng **cùng** epoch / lr / batch / dropout trên mỗi nhánh → so sánh công bằng.
Muốn chạy ít hướng hơn (vd. chỉ so 1 chiều vs 2 chiều): `--configs ppi_attn ppi_bi_attn`.

Profiles (--profile):
  fast      ~15–25 ph/nhánh/hướng  (tổng ~2–3 h cho 6 run)
  balanced  ~25–35 ph/nhánh/hướng  (mặc định, ~3–4 h)
  quality   ~40–55 ph/nhánh/hướng  (~5–7 h)
  match_a   cùng epoch phương pháp A (mf/cc 20, bp 15) + pos-weight + ckpt combo

Usage (Kaggle):
  %env DATA_DIR=/kaggle/working/CAFA6
  %env DGL_CUDA=1
  !python /kaggle/working/CAFA6/scripts/run_fusion_ablation.py --profile balanced

  !python .../run_fusion_ablation.py --profile fast --branches mf
  !python .../run_fusion_ablation.py --eval-only --profile balanced
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.baseline_config import BRANCH_BASELINE_DROPOUT  # noqa: E402

BRANCHES_DEFAULT = ("cc", "mf", "bp")

# Baseline paper (Table 1) — đối chiếu trong summary
BASELINE_REPRO = {
    "mf": {"fmax": 0.302, "auc": 0.808, "aupr": 0.256},
    "cc": {"fmax": 0.531, "auc": 0.886, "aupr": 0.601},
    "bp": {"fmax": 0.303, "auc": 0.749, "aupr": 0.261},
}

# Run tốt nhất trước (PPI + attention) — tham chiếu
BEST_PRIOR = {
    "mf": {"fmax": 0.477, "auc": 0.858, "aupr": 0.441, "note": "PPI+attn, 20ep kaggle"},
    "cc": {"fmax": 0.585, "auc": 0.944, "aupr": 0.633, "note": "baseline-parity 20ep"},
    "bp": {"fmax": 0.333, "auc": 0.944, "aupr": 0.281, "note": "kaggle 15ep"},
}


@dataclass(frozen=True)
class BranchHP:
    epochs: int
    learningrate: float
    validate_every: int
    batch_size: int
    est_minutes: int  # ước lượng 1 hướng trên T4


@dataclass(frozen=True)
class FusionConfig:
    name: str
    use_ppi: bool
    fusion_mode: str

    def train_args(self) -> list[str]:
        out = ["--fusion", self.fusion_mode]
        if not self.use_ppi:
            out.append("--no-ppi")
        return out


FUSION_CONFIGS: tuple[FusionConfig, ...] = (
    FusionConfig("ppi_concat", use_ppi=True, fusion_mode="concat"),
    FusionConfig("no_ppi_attn", use_ppi=False, fusion_mode="attention"),
    FusionConfig("ppi_attn", use_ppi=True, fusion_mode="attention"),      # 1 chiều
    FusionConfig("ppi_bi_attn", use_ppi=True, fusion_mode="bi_attention"),  # 2 chiều
)

# epoch / lr / validate — tối ưu thời gian; **giống nhau** cho mọi hướng trên cùng nhánh
PROFILES: dict[str, dict[str, BranchHP]] = {
    "fast": {
        "mf": BranchHP(epochs=8, learningrate=1e-4, validate_every=4, batch_size=64, est_minutes=18),
        "cc": BranchHP(epochs=8, learningrate=1e-4, validate_every=4, batch_size=64, est_minutes=20),
        "bp": BranchHP(epochs=6, learningrate=1e-4, validate_every=3, batch_size=64, est_minutes=22),
    },
    "balanced": {
        "mf": BranchHP(epochs=10, learningrate=1e-4, validate_every=4, batch_size=64, est_minutes=28),
        "cc": BranchHP(epochs=10, learningrate=1e-4, validate_every=3, batch_size=64, est_minutes=30),
        "bp": BranchHP(epochs=8, learningrate=1e-4, validate_every=3, batch_size=64, est_minutes=32),
    },
    "quality": {
        "mf": BranchHP(epochs=15, learningrate=1e-4, validate_every=4, batch_size=64, est_minutes=42),
        "cc": BranchHP(epochs=12, learningrate=1e-4, validate_every=3, batch_size=64, est_minutes=45),
        "bp": BranchHP(epochs=12, learningrate=1e-4, validate_every=3, batch_size=64, est_minutes=48),
    },
    # Cùng budget epoch với phương pháp A (PPI+attention) — dùng cho B/C sau khi sửa fusion
    "match_a": {
        "mf": BranchHP(epochs=20, learningrate=1e-4, validate_every=4, batch_size=64, est_minutes=55),
        "cc": BranchHP(epochs=20, learningrate=1e-4, validate_every=3, batch_size=64, est_minutes=58),
        "bp": BranchHP(epochs=15, learningrate=1e-4, validate_every=3, batch_size=64, est_minutes=50),
    },
}


def _lr_tag(lr: float) -> str:
    return f"{lr:g}"


def _env(data_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["DATA_DIR"] = str(data_dir)
    env.setdefault("DGL_CUDA", "1")
    return env


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def _ckpt_tag(branch: str, cfg: FusionConfig, hp: BranchHP, loss: str | None = None) -> str:
    dr = f"{BRANCH_BASELINE_DROPOUT[branch]:g}"
    lr = _lr_tag(hp.learningrate)
    # loss=None (mặc định, không truyền --loss) giữ nguyên tên cũ — tương thích
    # ngược với checkpoint đã có; chỉ thêm hậu tố khi override --loss tường minh.
    suffix = f"_{loss}" if loss else ""
    return f"bestmodel_{branch}_{cfg.name}_{hp.batch_size}_{lr}_{dr}{suffix}.pkl"


def _train_ckpt(data_dir: Path, branch: str, hp: BranchHP) -> Path:
    dr = f"{BRANCH_BASELINE_DROPOUT[branch]:g}"
    lr = _lr_tag(hp.learningrate)
    return data_dir / "save_models" / f"bestmodel_{branch}_{hp.batch_size}_{lr}_{dr}.pkl"


def _parse_test_log(log_path: Path) -> dict[str, float] | None:
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    hits = re.findall(
        r"f_score\s+([\d.]+).*?auc\s+([\d.]+).*?aupr\s+([\d.]+)",
        text,
        re.S,
    )
    if not hits:
        return None
    f, a, u = hits[-1]
    thresh_hits = re.findall(r"thresh:\s*([\d.]+),\s*f_score", text)
    return {
        "fmax": float(f),
        "auc": float(a),
        "aupr": float(u),
        "thresh": float(thresh_hits[-1]) if thresh_hits else None,
    }


def train_one(
    branch: str,
    cfg: FusionConfig,
    hp: BranchHP,
    cwd: Path,
    data_dir: Path,
    env: dict[str, str],
    loss: str | None = None,
) -> Path:
    dropout = BRANCH_BASELINE_DROPOUT[branch]
    cmd = [
        sys.executable,
        "train_Struct2GO2.py",
        "-branch",
        branch,
        "--no-baseline-parity",
        "--kaggle",
        "-epochs",
        str(hp.epochs),
        "-dropout",
        str(dropout),
        "-batch_size",
        str(hp.batch_size),
        "-learningrate",
        str(hp.learningrate),
        "-validate_every",
        str(hp.validate_every),
        "--amp",
        "--pos-weight",
        "--ckpt-metric",
        "combo",
        *cfg.train_args(),
    ]
    if loss is not None:
        # --loss (nếu truyền) ưu tiên hơn --pos-weight ở trên — xem
        # _build_criterion() trong train_Struct2GO2.py. Áp DÙNG CHUNG cho mọi
        # config trong lần chạy này (không nhân chéo fusion × loss, tránh nổ số
        # run) — muốn so 2 loss thì chạy script 2 lần với --loss khác nhau.
        cmd += ["--loss", loss]
    rc = _run(cmd, cwd, env)
    if rc != 0:
        raise RuntimeError(f"Train failed: {branch} / {cfg.name} (exit {rc})")

    src = _train_ckpt(data_dir, branch, hp)
    if not src.is_file():
        raise FileNotFoundError(f"Không thấy checkpoint sau train: {src}")

    dst = data_dir / "save_models" / _ckpt_tag(branch, cfg, hp, loss=loss)
    shutil.copy2(src, dst)
    print(f"[save] {dst} ({dst.stat().st_size / 1e6:.1f} MB)")
    return dst


def eval_one(
    branch: str,
    cfg: FusionConfig,
    model_path: Path,
    cwd: Path,
    env: dict[str, str],
    baseline_eval: bool,
) -> int:
    cmd = [
        sys.executable,
        "eval_Struct2GO2.py",
        "-branch",
        branch,
        "--split",
        "test",
        "-model_path",
        str(model_path),
        *cfg.train_args(),
    ]
    if baseline_eval:
        cmd.append("--baseline-parity")
    else:
        cmd.append("--no-baseline-parity")
    return _run(cmd, cwd, env)


def _resolve_hp(
    branch: str,
    profile: str,
    epochs: int | None,
    lr: float | None,
    batch: int | None,
    validate_every: int | None,
) -> BranchHP:
    base = PROFILES[profile][branch]
    return BranchHP(
        epochs=epochs if epochs is not None else base.epochs,
        learningrate=lr if lr is not None else base.learningrate,
        validate_every=validate_every if validate_every is not None else base.validate_every,
        batch_size=batch if batch is not None else base.batch_size,
        est_minutes=base.est_minutes,
    )


def _print_profile_table(profile: str, branches: list[str]) -> None:
    print(f"\n=== Profile: {profile} ===")
    print(f"{'branch':<6} {'epochs':>6} {'lr':>10} {'batch':>6} {'val_every':>10} {'~min':>6}")
    print("-" * 48)
    total = 0
    for br in branches:
        hp = PROFILES[profile][br]
        print(
            f"{br:<6} {hp.epochs:>6} {hp.learningrate:>10.0e} {hp.batch_size:>6} "
            f"{hp.validate_every:>10} {hp.est_minutes:>6}"
        )
        total += hp.est_minutes * len(FUSION_CONFIGS)
    print(f"Ước lượng tổng ({len(FUSION_CONFIGS)} hướng × {len(branches)} nhánh): ~{total} phút\n")


def _compare_branch(results: list[dict], branch: str) -> None:
    rows = [r for r in results if r["branch"] == branch and "fmax" in r]
    if len(rows) < 2:
        return
    bl = BASELINE_REPRO[branch]
    prior = BEST_PRIOR[branch]
    print(f"\n--- So sánh {branch.upper()} ---")
    print(f"  Baseline paper:  F-max={bl['fmax']:.3f}  AUC={bl['auc']:.3f}  AUPR={bl['aupr']:.3f}")
    print(f"  PPI+attn (cũ):   F-max={prior['fmax']:.3f}  ({prior['note']})")
    best = max(rows, key=lambda r: r["fmax"])
    for r in sorted(rows, key=lambda x: x["config"]):
        d_bl = r["fmax"] - bl["fmax"]
        print(
            f"  {r['config']:<14} ep={r['epochs']} lr={r['lr']:.0e}  "
            f"F-max={r['fmax']:.4f} ({d_bl:+.3f} vs baseline)  "
            f"AUC={r['auc']:.4f}  AUPR={r['aupr']:.4f}"
        )
    print(f"  → Tốt nhất ablation: {best['config']} (F-max={best['fmax']:.4f})")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ablation fusion: concat vs attention 1 chiều vs attention 2 chiều vs no-PPI (3 nhánh)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument(
        "--branches", nargs="+", default=list(BRANCHES_DEFAULT), choices=["mf", "cc", "bp"]
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[c.name for c in FUSION_CONFIGS],
        choices=[c.name for c in FUSION_CONFIGS],
    )
    parser.add_argument(
        "--profile",
        choices=list(PROFILES.keys()),
        default="balanced",
        help="Bộ epoch/lr/batch (fast | balanced | quality | match_a)",
    )
    parser.add_argument("-epochs", type=int, default=None, help="Override epoch mọi nhánh")
    parser.add_argument("-learningrate", type=float, default=None, help="Override lr mọi nhánh")
    parser.add_argument("-batch_size", type=int, default=None)
    parser.add_argument("-validate_every", type=int, default=None)
    parser.add_argument("--baseline-eval", action="store_true")
    parser.add_argument("--train-only", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument(
        "--loss",
        choices=["bce", "bce_pos_weight", "focal"],
        default=None,
        help=(
            "Override loss cho MỌI config trong lần chạy này (không mặc định "
            "--pos-weight/bce_pos_weight của mỗi config nữa). Không nhân chéo "
            "fusion × loss — muốn so loss khác nhau thì chạy script nhiều lần."
        ),
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

    print("DATA_DIR:", data_dir)
    print("Profile:", args.profile)
    _print_profile_table(args.profile, args.branches)
    print("Configs:", [c.name for c in selected])

    for branch in args.branches:
        hp = _resolve_hp(
            branch,
            args.profile,
            args.epochs,
            args.learningrate,
            args.batch_size,
            args.validate_every,
        )
        dropout = BRANCH_BASELINE_DROPOUT[branch]

        for cfg in selected:
            tag = f"{branch}/{cfg.name}"
            print("\n" + "=" * 60)
            print(f"=== {tag} | ep={hp.epochs} lr={hp.learningrate:.0e} batch={hp.batch_size} ===")
            print("=" * 60)

            ckpt = data_dir / "save_models" / _ckpt_tag(branch, cfg, hp, loss=args.loss)

            try:
                if not args.eval_only:
                    ckpt = train_one(branch, cfg, hp, cwd, data_dir, env, loss=args.loss)

                if not args.train_only:
                    if not ckpt.is_file():
                        raise FileNotFoundError(f"Thiếu checkpoint: {ckpt}")
                    rc = eval_one(branch, cfg, ckpt, cwd, env, args.baseline_eval)
                    if rc != 0:
                        raise RuntimeError(f"Eval exit {rc}")

                    # Lưu log test riêng từng config
                    src_log = data_dir / "log" / f"test_{branch}.log"
                    dst_log = data_dir / "log" / f"test_{branch}_{cfg.name}.log"
                    if src_log.is_file():
                        shutil.copy2(src_log, dst_log)

                metrics = _parse_test_log(data_dir / "log" / f"test_{branch}_{cfg.name}.log")
                if not metrics:
                    metrics = _parse_test_log(data_dir / "log" / f"test_{branch}.log")
                row = {
                    "profile": args.profile,
                    "branch": branch,
                    "config": cfg.name,
                    "ppi": cfg.use_ppi,
                    "fusion": cfg.fusion_mode,
                    "loss": args.loss or "bce_pos_weight",
                    "epochs": hp.epochs,
                    "dropout": dropout,
                    "batch": hp.batch_size,
                    "lr": hp.learningrate,
                    "validate_every": hp.validate_every,
                    "checkpoint": ckpt.name,
                    **(metrics or {}),
                }
                results.append(row)
                if metrics:
                    bl = BASELINE_REPRO[branch]["fmax"]
                    print(
                        f"[metric] F-max={metrics['fmax']:.4f} ({metrics['fmax']-bl:+.3f} vs baseline), "
                        f"AUC={metrics['auc']:.4f}, AUPR={metrics['aupr']:.4f}"
                    )
            except Exception as exc:
                print(f"[FAIL] {tag}: {exc}", file=sys.stderr)
                failed.append(tag)

    elapsed = (time.time() - t0) / 60
    print("\n" + "=" * 72)
    print("=== SUMMARY ===")
    print("=" * 72)
    hdr = (
        f"{'br':<4} {'config':<14} {'ep':>3} {'lr':>8} "
        f"{'F-max':>7} {'AUC':>7} {'AUPR':>7}  checkpoint"
    )
    print(hdr)
    print("-" * len(hdr))
    for r in results:
        lr_s = f"{r['lr']:.0e}"
        fmax = f"{r.get('fmax', 0):.4f}" if "fmax" in r else "   -   "
        auc = f"{r.get('auc', 0):.4f}" if "auc" in r else "   -   "
        aupr = f"{r.get('aupr', 0):.4f}" if "aupr" in r else "   -   "
        print(
            f"{r['branch']:<4} {r['config']:<14} {r['epochs']:>3} {lr_s:>8} "
            f"{fmax:>7} {auc:>7} {aupr:>7}  {r['checkpoint']}"
        )

    for branch in args.branches:
        _compare_branch(results, branch)

    summary_path = data_dir / "log" / "fusion_ablation_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": args.profile,
        "elapsed_minutes": round(elapsed, 1),
        "results": results,
        "baseline_repro": BASELINE_REPRO,
        "best_prior_ppi_attn": BEST_PRIOR,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nWrote {summary_path}  (elapsed {elapsed:.1f} min)")

    if failed:
        print("Failed:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
