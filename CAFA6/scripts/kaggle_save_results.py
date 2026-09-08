"""Copy CAFA6 logs/checkpoints to /kaggle/working for notebook Output.

Checks whether a branch log looks finished (saved checkpoint/final, or best_fscore
after the last training run). MF can be skipped for retrain if the log is OK.
"""
from __future__ import annotations

import argparse
import re
import shutil
import sys
import zipfile
from pathlib import Path


def _log_candidates(data_dir: Path, branch: str) -> list[Path]:
    names = [f"{branch}.log"]
    roots = [
        data_dir / "log",
        Path("/kaggle/working/log"),
        Path("/kaggle/working/CAFA6/log"),
    ]
    out: list[Path] = []
    for root in roots:
        for name in names:
            p = root / name
            if p.is_file():
                out.append(p)
    return out


def _model_dirs(data_dir: Path) -> list[Path]:
    return [
        data_dir / "save_models",
        Path("/kaggle/working/save_models"),
        Path("/kaggle/working/CAFA6/save_models"),
    ]


def analyze_log(text: str) -> dict:
    """Heuristic: last training block in log (after restart)."""
    blocks = text.split("########start training###########")
    block = blocks[-1] if blocks else text
    has_traceback = "Traceback" in block
    saved = bool(
        re.search(r"saved (final|checkpoint):", block, re.I)
        or re.search(r"\bSAVED\b", block)
    )
    best = re.findall(r"best_fscore:\s*([\d.]+)", block)
    epochs = [int(m.group(1)) for m in re.finditer(r"epoch:\s*(\d+)", block)]
    validating = "validating" in block
    valid_done = "valid forward done" in block or "########valid metric###########" in block
    last_epoch = max(epochs) if epochs else -1

    if saved:
        status = "ok"
        note = "Đã lưu model (saved checkpoint/final)."
    elif has_traceback:
        status = "error"
        note = "Log có Traceback — train lỗi, chỉ nên lưu log tham khảo."
    elif best and (valid_done or float(best[-1]) > 0):
        status = "ok"
        note = f"Có best_fscore={best[-1]} — coi như train xong (có thể thiếu .pkl cũ)."
    elif validating and not valid_done:
        status = "incomplete"
        note = "Dừng sau 'validating' — validation chưa xong (lỗi cacul_aupr cũ?)."
    elif last_epoch >= 0:
        status = "partial"
        note = f"Chỉ thấy train đến epoch {last_epoch}, chưa rõ validation."
    else:
        status = "empty"
        note = "Không thấy block train."

    return {
        "status": status,
        "note": note,
        "saved": saved,
        "best_fscore": best[-1] if best else None,
        "last_epoch": last_epoch,
    }


def copy_branch(
    branch: str,
    data_dir: Path,
    out_log: Path,
    out_models: Path,
) -> None:
    logs = _log_candidates(data_dir, branch)
    if not logs:
        print(f"[{branch}] Không tìm thấy {branch}.log", file=sys.stderr)
        return

    src_log = max(logs, key=lambda p: p.stat().st_mtime)
    text = src_log.read_text(encoding="utf-8", errors="replace")
    info = analyze_log(text)
    print(f"[{branch}] log: {src_log} ({src_log.stat().st_size} B)")
    print(f"         status={info['status']}: {info['note']}")

    out_log.mkdir(parents=True, exist_ok=True)
    dst_log = out_log / f"{branch}.log"
    shutil.copy2(src_log, dst_log)
    print(f"         -> {dst_log}")

    copied = 0
    out_models.mkdir(parents=True, exist_ok=True)
    for models_dir in _model_dirs(data_dir):
        if not models_dir.is_dir():
            continue
        patterns = (
            f"bestmodel_{branch}_*.pkl",
            f"final_{branch}_*.pkl",
            f"*_{branch}_*.pkl",
        )
        seen: set[Path] = set()
        for pat in patterns:
            for f in sorted(models_dir.glob(pat)):
                if f in seen or branch not in f.name:
                    continue
                seen.add(f)
                dst = out_models / f.name
                if not dst.exists() or f.stat().st_mtime > dst.stat().st_mtime:
                    shutil.copy2(f, dst)
                print(f"         model: {f} -> {dst}")
                copied += 1
    if copied == 0:
        print(f"         (không có .pkl cho nhánh {branch})")


def _test_log_candidates(data_dir: Path, branch: str) -> list[Path]:
    name = f"test_{branch}.log"
    roots = [
        data_dir / "log",
        Path("/kaggle/working/log"),
        Path("/kaggle/working/CAFA6/log"),
    ]
    out: list[Path] = []
    for root in roots:
        p = root / name
        if p.is_file():
            out.append(p)
    return out


def copy_test_result(
    branch: str, data_dir: Path, out_test: Path, out_log: Path | None = None
) -> None:
    src_dir = data_dir / "test_result"
    if not src_dir.is_dir():
        src_dir = Path("/kaggle/working/CAFA6/test_result")
    if src_dir.is_dir():
        out_test.mkdir(parents=True, exist_ok=True)
        for pattern in (
            f"{branch}_result.json",
            f"{branch}*_pred_actual.pkl",
            f"{branch}_roc_curve.png",
        ):
            for f in src_dir.glob(pattern):
                dst = out_test / f.name
                shutil.copy2(f, dst)
                print(f"         test_result: {f.name} -> {dst}")

    logs = _test_log_candidates(data_dir, branch)
    if not logs:
        return
    src_log = max(logs, key=lambda p: p.stat().st_mtime)
    out_test.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_log, out_test / src_log.name)
    print(f"         test log: {src_log} -> {out_test / src_log.name}")
    if out_log is not None:
        out_log.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_log, out_log / src_log.name)
        print(f"         test log: {src_log} -> {out_log / src_log.name}")


def _matches_branch(filename: str, branch: str) -> bool:
    name = filename.lower()
    br = branch.lower()
    return br in name or name.startswith(f"test_{br}")


def _collect_outputs(
    out_log: Path,
    out_models: Path,
    out_test: Path,
    branch: str | None = None,
    branches: list[str] | None = None,
) -> list[tuple[Path, str]]:
    """Return (file_path, arcname) pairs; optional filter by GO branch name in filename."""
    only = branches if branches is not None else ([branch] if branch else None)
    pairs: list[tuple[Path, str]] = []
    for folder, arc_prefix in (
        (out_log, "log"),
        (out_models, "save_models"),
        (out_test, "test_result"),
    ):
        if not folder.is_dir():
            continue
        for f in sorted(folder.rglob("*")):
            if not f.is_file():
                continue
            if only is not None:
                if f.name == "_kaggle_manifest.txt":
                    pass
                elif not any(_matches_branch(f.name, br) for br in only):
                    continue
            pairs.append((f, f"{arc_prefix}/{f.relative_to(folder).as_posix()}"))
    return pairs


def write_manifest(
    branches: list[str],
    out_log: Path,
    out_models: Path,
    out_test: Path,
    manifest_path: Path,
) -> None:
    lines = ["# CAFA6 Kaggle output manifest", f"branches_saved: {', '.join(branches)}", ""]
    for folder, label in (
        (out_log, "log"),
        (out_models, "save_models"),
        (out_test, "test_result"),
    ):
        if not folder.is_dir():
            continue
        lines.append(f"## {label}/")
        for f in sorted(folder.iterdir()):
            if f.is_file():
                lines.append(f"  {f.name}  ({f.stat().st_size} B)")
        lines.append("")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[manifest] {manifest_path}")


def make_zip(
    out_log: Path,
    out_models: Path,
    out_test: Path,
    zip_path: Path,
    data_dir: Path | None = None,
    branch: str | None = None,
) -> None:
    """Pack log/, save_models/, test_result/ into one zip for Kaggle download."""
    zip_path.parent.mkdir(parents=True, exist_ok=True)
    only = [branch] if branch else None
    pairs = _collect_outputs(out_log, out_models, out_test, branches=only)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for f, arcname in pairs:
            zf.write(f, arcname=arcname)
        if data_dir is not None and branch is None and only is None:
            manifest = out_log / "_kaggle_manifest.txt"
            if not manifest.is_file():
                manifest = data_dir / "log" / "_kaggle_manifest.txt"
            if manifest.is_file():
                zf.write(manifest, arcname="log/_kaggle_manifest.txt")
    mb = zip_path.stat().st_size / 1e6
    tag = f"[{branch}] " if branch else ""
    print(f"[zip] {tag}wrote {zip_path} ({mb:.1f} MB, {len(pairs)} files)")


def make_split_zips(
    out_log: Path,
    out_models: Path,
    out_test: Path,
    zip_base: Path,
    max_mb: float,
    branches: list[str] | None = None,
) -> list[Path]:
    """Split all outputs into zip_base_part1.zip, part2.zip, … each ≤ max_mb (approx)."""
    max_bytes = int(max_mb * 1e6)
    pairs = _collect_outputs(out_log, out_models, out_test, branches=branches)
    if not pairs:
        return []
    parts: list[Path] = []
    part_idx = 1
    current: list[tuple[Path, str]] = []
    current_size = 0

    def flush() -> None:
        nonlocal part_idx, current, current_size
        if not current:
            return
        part_path = zip_base.parent / f"{zip_base.stem}_part{part_idx}{zip_base.suffix}"
        with zipfile.ZipFile(part_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f, arcname in current:
                zf.write(f, arcname=arcname)
        mb = part_path.stat().st_size / 1e6
        print(f"[zip] split wrote {part_path} ({mb:.1f} MB, {len(current)} files)")
        parts.append(part_path)
        part_idx += 1
        current = []
        current_size = 0

    for f, arcname in pairs:
        sz = f.stat().st_size
        if current and current_size + sz > max_bytes:
            flush()
        current.append((f, arcname))
        current_size += sz
    flush()
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description="Lưu log/model CAFA6 ra Kaggle Output")
    parser.add_argument(
        "--data-dir",
        default=None,
        help="Thư mục CAFA6 (mặc định: DATA_DIR hoặc /kaggle/working/CAFA6)",
    )
    parser.add_argument(
        "--branches",
        nargs="+",
        default=["mf", "cc", "bp"],
        choices=["mf", "cc", "bp"],
    )
    parser.add_argument(
        "--skip-train-if-ok",
        action="store_true",
        help="In danh sách nhánh status=ok (chỉ copy, không gợi ý train lại)",
    )
    parser.add_argument(
        "--zip",
        action="store_true",
        help="Tạo /kaggle/working/cafa6_output.zip sau khi copy",
    )
    parser.add_argument(
        "--zip-name",
        default="cafa6_output.zip",
        help="Tên file zip (trong /kaggle/working/)",
    )
    parser.add_argument(
        "--per-branch-zip",
        action="store_true",
        help="Thêm cafa6_{cc,mf,bp}.zip — mỗi nhánh riêng, dễ tải từng phần",
    )
    parser.add_argument(
        "--split-mb",
        type=float,
        default=0,
        metavar="N",
        help="Chia zip đầy đủ thành cafa6_output_part1.zip, part2.zip, … mỗi phần ≤ N MB",
    )
    args = parser.parse_args()

    import os

    data_dir = Path(args.data_dir or os.environ.get("DATA_DIR", "/kaggle/working/CAFA6"))
    out_log = Path("/kaggle/working/log")
    out_models = Path("/kaggle/working/save_models")
    out_test = Path("/kaggle/working/test_result")

    print("DATA_DIR:", data_dir)
    ok_skip: list[str] = []
    for branch in args.branches:
        logs = _log_candidates(data_dir, branch)
        if logs:
            info = analyze_log(logs[0].read_text(encoding="utf-8", errors="replace"))
            if info["status"] == "ok":
                ok_skip.append(branch)
        copy_branch(branch, data_dir, out_log, out_models)
        copy_test_result(branch, data_dir, out_test, out_log)

    write_manifest(args.branches, out_log, out_models, out_test, out_log / "_kaggle_manifest.txt")

    if args.zip:
        zip_path = Path("/kaggle/working") / args.zip_name
        pairs = _collect_outputs(out_log, out_models, out_test, branches=args.branches)
        zip_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            for f, arcname in pairs:
                zf.write(f, arcname=arcname)
            manifest = out_log / "_kaggle_manifest.txt"
            if manifest.is_file():
                zf.write(manifest, arcname="log/_kaggle_manifest.txt")
        print(
            f"[zip] wrote {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB, "
            f"{len(pairs)} files, branches={','.join(args.branches)})"
        )

    if args.per_branch_zip:
        for branch in args.branches:
            branch_zip = Path("/kaggle/working") / f"cafa6_{branch}.zip"
            make_zip(out_log, out_models, out_test, branch_zip, data_dir, branch=branch)

    if args.split_mb and args.split_mb > 0:
        zip_base = Path("/kaggle/working") / Path(args.zip_name).stem
        make_split_zips(
            out_log,
            out_models,
            out_test,
            zip_base.with_suffix(".zip"),
            args.split_mb,
            branches=args.branches,
        )

    print("\n=== Output ===")
    if out_log.is_dir():
        for f in sorted(out_log.glob("*.log")):
            print(f"  log/{f.name}  ({f.stat().st_size} B)")
    if out_models.is_dir():
        for f in sorted(out_models.glob("*.pkl")):
            print(f"  save_models/{f.name}  ({f.stat().st_size / 1e6:.1f} MB)")
    else:
        print("  save_models/ (trống)")

    if args.skip_train_if_ok and ok_skip:
        print("\nCó thể bỏ qua train lại:", ", ".join(ok_skip))


if __name__ == "__main__":
    main()
