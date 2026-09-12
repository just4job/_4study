"""Giải quyết đường dẫn DATA_DIR / RAW_DIR cho toàn bộ pipeline.

Trước đây mỗi script hardcode `Path("D:/CAFA6")` và `Path("D:/raw_data")` — đúng
với 1 máy Windows duy nhất, chết ngay ở mọi chỗ khác (Linux, macOS, Kaggle). Module
này gom logic đó về 1 nơi, ưu tiên theo thứ tự:

    1. Biến môi trường DATA_DIR / RAW_DIR (rõ ràng nhất, dùng được trên Kaggle)
    2. Thư mục repo (data nằm cạnh code: <repo>/proceed_data, <repo>/../raw_data)
    3. Thư mục làm việc hiện tại
    4. D:/CAFA6 và D:/raw_data — chỉ khi đang chạy Windows

Dùng:
    from data_processing.paths import resolve_data_dir, resolve_raw_dir
    BASE_DIR = resolve_data_dir()
    RAW_DIR  = resolve_raw_dir()
"""
from __future__ import annotations

import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]

# Thư mục có proceed_data/ bên trong mới là DATA_DIR hợp lệ.
_DATA_MARKER = "proceed_data"
# raw_data hợp lệ nếu có ít nhất 1 trong các file/thư mục nguồn.
_RAW_MARKERS = ("ppi.txt", "seq.fasta", "goa_human.gaf.gz", "struct_feature", "go.obo")


def _windows_default(path: str) -> list[Path]:
    """D:/... chỉ có nghĩa trên Windows; ở nơi khác nó chỉ gây thông báo lỗi khó hiểu."""
    return [Path(path)] if os.name == "nt" else []


def resolve_data_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Thư mục chứa proceed_data/ và divided_data/."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("DATA_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    candidates += [REPO_DIR, Path.cwd(), *_windows_default("D:/CAFA6")]
    for c in candidates:
        if (c / _DATA_MARKER).is_dir():
            return c
    # Không tìm thấy: trả về lựa chọn đầu tiên để script tự báo lỗi ở đúng chỗ
    # thiếu file, thay vì báo một đường dẫn Windows mà người dùng chưa từng có.
    return Path(env) if env else REPO_DIR


def resolve_raw_dir(explicit: str | os.PathLike | None = None) -> Path:
    """Thư mục chứa dữ liệu thô: ppi.txt, seq.fasta, goa_human.gaf.gz, struct_feature/."""
    if explicit:
        return Path(explicit)
    env = os.environ.get("RAW_DIR")
    candidates: list[Path] = []
    if env:
        candidates.append(Path(env))
    data_dir = resolve_data_dir()
    candidates += [
        data_dir / "raw_data",
        data_dir.parent / "raw_data",
        REPO_DIR.parent / "raw_data",
        Path.cwd() / "raw_data",
        *_windows_default("D:/raw_data"),
    ]
    for c in candidates:
        if c.is_dir() and any((c / m).exists() for m in _RAW_MARKERS):
            return c
    for c in candidates:
        if c.is_dir():
            return c
    return Path(env) if env else (REPO_DIR.parent / "raw_data")
