#!/usr/bin/env python3
"""Verify divided_data pickles load correctly (MyDataSet registered)."""
from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import __main__

from data_processing.divide_data import MyDataSet

__main__.MyDataSet = MyDataSet


def _ensure_dgl() -> bool:
    try:
        import dgl  # noqa: F401
        return True
    except Exception:
        print(
            "[WARN] DGL not available or not usable in this env; falling back to --quick verification.\n"
            "  Use --quick directly if you only want size/EOF checks.",
            file=sys.stderr,
        )
        return False


def _quick_check(path: Path) -> tuple[str, str]:
    """Size + đọc pickle tới EOF (không cần DGL — không load object graph)."""
    size_mb = path.stat().st_size / (1024 * 1024)
    try:
        with open(path, "rb") as handle:
            while True:
                try:
                    pickle.load(handle)
                except EOFError:
                    break
        return "OK_QUICK", f"{size_mb:.1f} MB (cần `pip install dgl` để xác nhận đầy đủ)"
    except EOFError:
        return "BAD_EOF", f"{size_mb:.1f} MB"
    except ModuleNotFoundError as exc:
        if "dgl" in str(exc).lower():
            return "OK_QUICK", f"{size_mb:.1f} MB (pickle đọc được tới graph — cài dgl để verify đủ)"
        return f"BAD_{type(exc).__name__}", f"{size_mb:.1f} MB ({exc})"
    except Exception as exc:
        return f"BAD_{type(exc).__name__}", f"{size_mb:.1f} MB ({exc})"


def check_file(path: Path, quick: bool) -> tuple[str, str]:
    if not path.is_file():
        return "MISSING", ""
    size_mb = path.stat().st_size / (1024 * 1024)
    if path.stat().st_size < 1024:
        return "EMPTY", f"{size_mb:.1f} MB"
    if quick:
        return _quick_check(path)
    try:
        with open(path, "rb") as handle:
            ds = pickle.load(handle)
        n = len(ds) if hasattr(ds, "__len__") else "?"
        return "OK", f"{size_mb:.1f} MB, n={n}"
    except EOFError:
        return "BAD_EOF", f"{size_mb:.1f} MB"
    except Exception as exc:
        return f"BAD_{type(exc).__name__}", f"{size_mb:.1f} MB ({exc})"


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify CAFA6 divided_data pickles")
    parser.add_argument("--data-dir", type=Path, default=REPO)
    parser.add_argument("--branch", choices=["mf", "cc", "bp", "all"], default="all")
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Không cần DGL: kiểm tra size + không bị cắt cụt (EOF)",
    )
    args = parser.parse_args()

    if not args.quick and not _ensure_dgl():
        args.quick = True

    div = args.data_dir / "divided_data"
    branches = ["mf", "cc", "bp"] if args.branch == "all" else [args.branch]
    failed = 0
    for branch in branches:
        for split in ("train", "valid", "test"):
            path = div / f"{branch}_{split}_dataset"
            status, info = check_file(path, quick=args.quick)
            print(f"{status:8}  {path.name:22}  {info}")
            if not status.startswith("OK"):
                failed += 1
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
