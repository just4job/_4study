#!/usr/bin/env python3
"""Tải kết quả CAFA6 từ /kaggle/working về máy, ngay trong notebook.

Vì sao là module để `import` chứ không phải script chạy bằng `!python`:
trình duyệt chỉ tải được file khi notebook phát ra một thẻ <a download> — mà
muốn phát HTML thì code phải chạy TRONG tiến trình notebook. `!python` chạy ở
tiến trình con, output của nó là văn bản thuần, không render HTML nào cả.

Dùng:
    import sys; sys.path.insert(0, "/kaggle/working/_4study/CAFA6")
    from scripts.kaggle_download import download

    download("cc")                  # log + test_result (vài trăm KB) — mặc định
    download("cc", models=True)     # kèm checkpoint .pkl (~40-60 MB/nhánh)
    download()                      # mọi nhánh đang có trong /kaggle/working

Mặc định KHÔNG kèm checkpoint: log + test_result là thứ cần đọc ngay và chỉ
vài trăm KB, trong khi checkpoint chiếm hầu như toàn bộ 60+ MB của bản zip đầy
đủ. Cần checkpoint để train tiếp thì thêm models=True, hoặc lấy từ tab Output
của Version — cách đó không giới hạn dung lượng.
"""
from __future__ import annotations

import base64
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from scripts.kaggle_save_results import make_split_zips  # noqa: E402

WORK = Path("/kaggle/working")
BRANCHES = ("cc", "mf", "bp")

# Nhúng base64 làm phình notebook: 1 MB zip -> ~1,37 MB text trong DOM. Hai
# ngưỡng dưới giữ cho tab trình duyệt không treo; vượt ngưỡng thì đưa link thường.
MAX_AUTO_MB = 25.0      # mỗi file
MAX_TOTAL_MB = 120.0    # tổng một lần gọi


def _dirs() -> tuple[Path, Path, Path]:
    """log/, save_models/, test_result/ — ưu tiên bản đã copy ra /kaggle/working."""
    data_dir = Path(os.environ.get("DATA_DIR", str(REPO)))
    out = []
    for name in ("log", "save_models", "test_result"):
        p = WORK / name
        out.append(p if p.is_dir() else data_dir / name)
    return out[0], out[1], out[2]


def _found_branches(out_log: Path) -> list[str]:
    if not out_log.is_dir():
        return []
    names = {f.name.lower() for f in out_log.glob("*.log")}
    return [b for b in BRANCHES if f"{b}.log" in names]


def _emit(path: Path) -> bool:
    """Phát một NÚT tải nhìn thấy được. True nếu đã nhúng được file.

    Cố tình không tự bấm hộ: Kaggle render output trong iframe sandbox và trình
    duyệt chặn mọi download không đến từ thao tác của người dùng, nên `.click()`
    bằng script chỉ im lặng không làm gì. Một cú bấm thật thì luôn được phép.
    """
    from IPython.display import HTML, display

    mb = path.stat().st_size / 1e6
    b64 = base64.b64encode(path.read_bytes()).decode()
    display(HTML(
        f'<a download="{path.name}" href="data:application/zip;base64,{b64}" '
        f'style="display:inline-block;padding:10px 18px;margin:6px 0;'
        f'background:#20beff;color:#fff;font-weight:600;border-radius:6px;'
        f'text-decoration:none;font-family:sans-serif">'
        f'⬇️ Tải {path.name} ({mb:.1f} MB)</a>'
    ))
    return True


def _link(path: Path, why: str) -> None:
    from IPython.display import FileLink, display

    mb = path.stat().st_size / 1e6
    display(FileLink(str(path), result_html_suffix="?download=1"))
    print(f"   {path.name} ({mb:.0f} MB) — {why}. Bấm link trên, "
          f"hoặc Save Version rồi tải ở tab Output.")


def download(
    branch: str | None = None,
    models: bool = False,
    split_mb: float = MAX_AUTO_MB,
    auto: bool = True,
) -> list[Path]:
    """Đóng gói rồi tải kết quả về. Trả về danh sách file zip đã tạo."""
    out_log, out_models, out_test = _dirs()
    branches = [branch] if branch else _found_branches(out_log)
    if not branches:
        print(f"[!] Không thấy log nhánh nào trong {out_log} — đã train xong chưa?")
        return []

    tag = "_".join(branches) + ("_full" if models else "_results")
    parts = make_split_zips(
        out_log, out_models, out_test,
        zip_base=WORK / f"cafa6_{tag}.zip",
        max_mb=split_mb,
        branches=branches,
        include_models=models,
    )
    if not parts:
        print(f"[!] Không có file nào khớp nhánh {branches}")
        return []

    total_mb = sum(p.stat().st_size for p in parts) / 1e6
    print(f"\n{len(parts)} file, tổng {total_mb:.1f} MB  (nhánh: {', '.join(branches)}"
          f"{', kèm checkpoint' if models else ', không kèm checkpoint'})")

    if not auto:
        return parts

    try:
        import IPython  # noqa: F401
    except ImportError:
        print("Không chạy trong notebook — file nằm ở:")
        for p in parts:
            print(" ", p)
        return parts

    if total_mb > MAX_TOTAL_MB:
        print(f"\nTổng > {MAX_TOTAL_MB:.0f} MB nên không nhúng tự tải "
              f"(nhúng base64 sẽ làm treo tab):")
        for p in parts:
            _link(p, "quá lớn để tự tải")
        return parts

    print()
    for p in parts:
        if p.stat().st_size / 1e6 > MAX_AUTO_MB:
            _link(p, f"lớn hơn {MAX_AUTO_MB:.0f} MB")
        else:
            _emit(p)
    print("\nBấm nút xanh ở trên để tải. Không thấy nút (output bị Kaggle nuốt, "
          "hoặc đang chạy chế độ Commit)?\n"
          "  -> Panel bên phải > Output > /kaggle/working > bấm 🔄 refresh > "
          "chuột phải file > Download")
    return parts


if __name__ == "__main__":
    print(__doc__)
    print("Chạy bằng !python sẽ KHÔNG tải được gì — phải import trong notebook.")
