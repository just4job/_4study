#!/usr/bin/env python3
"""Train CAFA6 branches on Kaggle (cc → mf → bp), save outputs, build zip.

Usage (from notebook, after kaggle_link_data.py):
  %env DATA_DIR=/kaggle/working/CAFA6
  %env DGL_CUDA=1
  !python /kaggle/working/CAFA6/scripts/kaggle_run_branches.py

  # Train only, skip eval:
  !python .../kaggle_run_branches.py --no-eval

  # Custom order / branches:
  !python .../kaggle_run_branches.py --branches cc mf --skip mf
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.kaggle_save_results import copy_branch, copy_test_result, make_zip  # noqa: E402

# Thứ tự nhanh → chậm (BP validation F-max rất lâu)
DEFAULT_ORDER = ("cc", "mf", "bp")

BRANCH_TRAIN_ARGS: dict[str, list[str]] = {
    "cc": [],
    "mf": [],  # baseline-parity → dropout 0.1 via scripts/baseline_config.py
    "bp": ["-dropout", "0.1"],
}


def _run(cmd: list[str], cwd: Path, env: dict[str, str]) -> int:
    print("\n>>>", " ".join(cmd), flush=True)
    return subprocess.call(cmd, cwd=str(cwd), env=env)


def _pickle_patch_env() -> dict[str, str]:
    env = os.environ.copy()
    env.setdefault("DATA_DIR", "/kaggle/working/CAFA6")
    env.setdefault("DGL_CUDA", "1")
    return env


def _ensure_mydataset() -> None:
    import __main__

    from data_processing.divide_data import MyDataSet

    __main__.MyDataSet = MyDataSet


def train_branch(
    branch: str,
    kaggle: bool,
    baseline_parity: bool,
    extra: list[str],
    cwd: Path,
    env: dict[str, str],
) -> int:
    cmd = [sys.executable, "train_Struct2GO2.py", "-branch", branch]
    if baseline_parity:
        cmd.append("--baseline-parity")
    elif kaggle:
        cmd.append("--kaggle")
    cmd.extend(BRANCH_TRAIN_ARGS.get(branch, []))
    cmd.extend(extra)
    return _run(cmd, cwd, env)


def eval_branch(
    branch: str, cwd: Path, env: dict[str, str], baseline_parity: bool
) -> int:
    cmd = [sys.executable, "eval_Struct2GO2.py", "-branch", branch]
    if baseline_parity:
        cmd.extend(["--baseline-parity", "--split", "test"])
    else:
        cmd.extend(["--no-baseline-parity", "--split", "auto"])
    return _run(cmd, cwd, env)


def main() -> int:
    parser = argparse.ArgumentParser(description="Kaggle: train branches, save, zip")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="CAFA6 root (default: DATA_DIR or /kaggle/working/CAFA6)",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=list(DEFAULT_ORDER),
        choices=["mf", "cc", "bp"],
    )
    parser.add_argument(
        "--skip",
        nargs="*",
        default=[],
        choices=["mf", "cc", "bp"],
        help="Nhánh đã train xong — chỉ copy + zip, không train lại",
    )
    parser.add_argument("--no-kaggle", action="store_true", help="Dùng preset mặc định repo (20 epoch, chậm)")
    parser.add_argument(
        "--baseline-parity",
        action="store_true",
        help="Train/eval Table 1 (20 ep, hid 512); MF dropout 0.1 — không kết hợp --kaggle",
    )
    parser.add_argument("--no-eval", action="store_true", help="Không chạy eval sau mỗi nhánh")
    parser.add_argument("--no-zip", action="store_true", help="Không tạo/cập nhật zip sau mỗi nhánh")
    parser.add_argument(
        "--zip-name",
        default="cafa6_output.zip",
        help="File zip dưới /kaggle/working/",
    )
    parser.add_argument(
        "train_extra",
        nargs="*",
        help="Tham số thêm cho train_Struct2GO2.py (vd. -epochs 5)",
    )
    args = parser.parse_args()

    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "/kaggle/working/CAFA6"))
    cwd = data_dir if (data_dir / "train_Struct2GO2.py").is_file() else REPO
    out_log = Path("/kaggle/working/log")
    out_models = Path("/kaggle/working/save_models")
    out_test = Path("/kaggle/working/test_result")
    zip_path = Path("/kaggle/working") / args.zip_name

    os.chdir(cwd)
    _ensure_mydataset()
    env = _pickle_patch_env()
    os.environ.update(env)

    print("REPO:", cwd)
    print("DATA_DIR:", data_dir)
    print("Branches:", args.branches)
    print("Skip train:", args.skip or "(none)")

    failed: list[str] = []
    skip_set = set(args.skip or [])

    for branch in args.branches:
        print("\n" + "=" * 60)
        print(f"=== {branch.upper()} ===")
        print("=" * 60)

        if branch not in skip_set:
            rc = train_branch(
                branch,
                kaggle=not args.no_kaggle and not args.baseline_parity,
                baseline_parity=args.baseline_parity,
                extra=args.train_extra,
                cwd=cwd,
                env=env,
            )
            if rc != 0:
                print(f"[{branch}] TRAIN FAILED (exit {rc})", file=sys.stderr)
                failed.append(branch)
                copy_branch(branch, data_dir, out_log, out_models)
                copy_test_result(branch, data_dir, out_test, out_log)
                if not args.no_zip:
                    make_zip(out_log, out_models, out_test, zip_path, data_dir)
                continue

            if not args.no_eval:
                rc = eval_branch(branch, cwd, env, baseline_parity=args.baseline_parity)
                if rc != 0:
                    print(f"[{branch}] EVAL FAILED (exit {rc}) — vẫn lưu log/model", file=sys.stderr)
        else:
            print(f"[{branch}] skip train (--skip)")

        copy_branch(branch, data_dir, out_log, out_models)
        copy_test_result(branch, data_dir, out_test, out_log)

        if not args.no_zip:
            make_zip(out_log, out_models, out_test, zip_path, data_dir)

    print("\n=== DONE ===")
    if zip_path.is_file():
        print(f"Download: {zip_path}")
        print("Notebook: from IPython.display import FileLink; display(FileLink('cafa6_output.zip'))")
    if failed:
        print("Failed branches:", ", ".join(failed), file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
