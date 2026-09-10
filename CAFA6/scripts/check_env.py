#!/usr/bin/env python3
"""Chẩn đoán môi trường TRƯỚC khi chạy pipeline — báo đúng cái gì thiếu, thiếu ở đâu.

Kịch bản mặc định của repo này: TIỀN XỬ LÝ ở máy local, TRAIN trên Kaggle. Hai
việc đó cần bộ thư viện KHÁC nhau, nên script chia rõ 2 nhóm và chỉ bắt lỗi ở
nhóm bạn đang chạy (`--target local` / `--target kaggle`, mặc định tự đoán).

Script này CHỈ dùng thư viện chuẩn — chạy được kể cả khi env đang hỏng, và
import mọi thứ trong try/except để không chết giữa chừng.

Cách chạy:
    python scripts/check_env.py
    python scripts/check_env.py --target local --raw-dir ~/raw_data --data-dir ~/CAFA6
    python scripts/check_env.py --target kaggle

Exit code: 0 = chạy được, 1 = có FAIL phải sửa.
"""
from __future__ import annotations

import argparse
import importlib
import os
import platform
import shutil
import sys
from pathlib import Path

# Python >= 3.10: 3_uniprot_mapping.py và 4_build_ppi_graph.py dùng cú pháp
# `str | None` trong annotation mà KHÔNG có `from __future__ import annotations`,
# nên trên 3.9 chúng chết ngay lúc import với TypeError.
MIN_PY = (3, 10)

# Dependency mà `import dgl` cần nhưng metadata của wheel dgl KHÔNG khai báo đủ
# -> pip install dgl xong vẫn thiếu. Mỗi cái chỉ lộ ra SAU khi cái trước đã có,
# nên kiểm tra trọn nhóm ở đây để không phải sửa từng vòng một.
DGL_DEPS = [
    ("packaging", "packaging", "dgl import lúc khởi động: dgl/utils/__init__.py"),
    ("yaml", "PyYAML", "dgl import lúc khởi động: dgl/graphbolt/impl/ondisk_dataset.py "
                       "(gói pip tên PyYAML, KHÔNG phải 'yaml')"),
    ("pydantic", "pydantic", "dgl import lúc khởi động: schema của dgl.graphbolt"),
    ("psutil", "psutil", "dgl import lúc khởi động: dgl.graphbolt"),
    ("torchdata", "torchdata", "dgl 2.x import torchdata.datapipes — torchdata >= 0.10 "
                               "đã bỏ module này, phải dùng torchdata==0.9.0"),
]

# (module import, tên gói pip, dùng để làm gì)
LOCAL_PKGS = [
    ("Bio", "biopython", "đọc PDB.gz + FASTA (2_extract_struct_map, get_sequence, 5_build_seq_feature)"),
    ("numpy", "numpy", "toàn bộ pipeline"),
    ("scipy", "scipy", "contact map (2_extract_struct_map)"),
    ("pandas", "pandas", "valid_protein_ids.csv (1_get_valid_ids, get_sequence)"),
    ("tqdm", "tqdm", "progress bar"),
    ("requests", "requests", "UniProt REST API (3_uniprot_mapping)"),
    ("torch", "torch", "5_build_seq_feature, 4_build_ppi_graph, 3_build_graph_dataset"),
    *DGL_DEPS,
    ("dgl", "dgl", "4_build_ppi_graph, 3_build_graph_dataset"),
    ("esm", "fair-esm", "ESM-2 encoder (5_build_seq_feature)"),
]
# Chỉ cần khi build lại protein_node2vec từ ppi.txt mới.
LOCAL_OPTIONAL_PKGS = [
    ("networkx", "networkx", "4_build_ppi_node2vec (chỉ khi rebuild protein_node2vec)"),
    ("node2vec", "node2vec", "4_build_ppi_node2vec (chỉ khi rebuild protein_node2vec)"),
]
KAGGLE_PKGS = [
    ("torch", "torch", "train/eval"),
    *DGL_DEPS,
    ("dgl", "dgl", "train/eval — Kaggle KHÔNG cài sẵn, xem hướng dẫn cuối"),
    ("numpy", "numpy", "train/eval"),
    ("sklearn", "scikit-learn", "model/evaluation.py (f1, precision, recall)"),
    ("transformers", "transformers", "get_cosine_schedule_with_warmup (train_Struct2GO2)"),
    ("tqdm", "tqdm", "progress bar"),
    ("pandas", "pandas", "eval_Struct2GO2 xuất kết quả"),
    ("matplotlib", "matplotlib", "eval_Struct2GO2 vẽ biểu đồ"),
]

# Input thô — (đường dẫn tương đối raw_dir, mô tả, bước dùng nó)
RAW_ITEMS = [
    ("struct_feature", "thư mục PDB AlphaFold (*.pdb.gz)", "bước 1, 2, get_sequence"),
    ("ppi.txt", "STRING PPI", "bước 5, 6"),
    ("goa_human.gaf.gz", "GO annotation", "go_anno.py"),
    ("seq.fasta", "FASTA protein người", "5_build_seq_feature"),
]
# Sản phẩm trung gian — (tên trong proceed_data, script sinh ra nó, bytes tối thiểu
# hợp lý). Ngưỡng để bắt file RỖNG/HỎNG: `pickle.dump({})` chỉ ra 5 byte và vẫn
# load được, nên "file tồn tại" không chứng minh được gì — phải xem cả kích thước.
PROC_ITEMS = [
    ("valid_protein_ids.csv", "1_get_valid_ids.py", 10_000),
    ("proteins_edges", "2_extract_struct_map.py", 0),
    ("protein_node2onehot", "get_sequence.py", 1_000_000),
    ("protein_node2vec", "4_build_ppi_node2vec.py", 100_000),
    ("HUMAN_protein_info.json", "go_anno.py", 100_000),
    ("human_BP_ACS.json", "2_build_go_namespace.py", 100_000),
    ("human_MF_ACS.json", "2_build_go_namespace.py", 100_000),
    ("human_CC_ACS.json", "2_build_go_namespace.py", 100_000),
    ("dict_sequence_feature", "5_build_seq_feature.py", 10_000_000),
    ("uniprot_ensembl_mapping.csv", "3_uniprot_mapping.py", 10_000),
    ("ppi_graph_global", "4_build_ppi_graph.py", 1_000_000),
    ("ppi_protein_index", "4_build_ppi_graph.py", 100_000),
]

# Pickle của container RỖNG — dấu hiệu script sinh ra file nhưng không ghi được
# dữ liệu nào (vd. chạy sai đường dẫn input rồi vẫn dump ở cuối).
EMPTY_PICKLES = {
    b"\x80\x04}\x94.": "dict rỗng {}",
    b"\x80\x05}\x94.": "dict rỗng {}",
    b"\x80\x02}q\x00.": "dict rỗng {}",
    b"\x80\x03}q\x00.": "dict rỗng {}",
    b"\x80\x04]\x94.": "list rỗng []",
    b"\x80\x05]\x94.": "list rỗng []",
}

OK, WARN, FAIL, INFO = "  OK  ", " WARN ", " FAIL ", " INFO "


class Report:
    def __init__(self) -> None:
        self.fails: list[str] = []
        self.warns: list[str] = []

    def add(self, level: str, msg: str, detail: str = "") -> None:
        print(f"[{level}] {msg}")
        if detail:
            for line in detail.splitlines():
                print(f"         {line}")
        if level == FAIL:
            self.fails.append(msg)
        elif level == WARN:
            self.warns.append(msg)


def human(n: float) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if abs(n) < 1024:
            return f"{n:,.1f} {unit}"
        n /= 1024
    return f"{n:,.1f} PB"


def section(title: str) -> None:
    print(f"\n{'=' * 70}\n  {title}\n{'=' * 70}")


def check_python(rep: Report) -> None:
    section("1. Python & hệ thống")
    v = sys.version_info
    rep.add(
        OK if v[:2] >= MIN_PY else FAIL,
        f"Python {v.major}.{v.minor}.{v.micro} (cần >= {MIN_PY[0]}.{MIN_PY[1]})",
        ""
        if v[:2] >= MIN_PY
        else "3_uniprot_mapping.py và 4_build_ppi_graph.py dùng cú pháp `str | None`\n"
        "trong annotation mà không có `from __future__ import annotations`,\n"
        "nên trên Python < 3.10 chúng chết ngay lúc import.",
    )
    print(f"[{INFO}] {platform.platform()}")
    print(f"[{INFO}] executable: {sys.executable}")
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    conda = os.environ.get("CONDA_DEFAULT_ENV")
    if conda:
        print(f"[{INFO}] conda env: {conda}")
    elif in_venv:
        print(f"[{INFO}] venv: {sys.prefix}")
    else:
        rep.add(WARN, "Đang dùng Python hệ thống (không venv/conda)",
                "Nên tạo env riêng để không xung đột phiên bản torch/dgl.")
    try:
        cpu = os.cpu_count() or 0
        print(f"[{INFO}] CPU cores: {cpu}")
    except Exception:
        pass


def _version_of(mod) -> str:
    for attr in ("__version__", "version", "VERSION"):
        val = getattr(mod, attr, None)
        if isinstance(val, str):
            return val
    return "?"


def _installed_version(pip_name: str) -> str | None:
    """Bản đã cài theo pip metadata — KHÁC với 'import được'."""
    try:
        from importlib.metadata import PackageNotFoundError, version
    except Exception:
        return None
    try:
        return version(pip_name)
    except PackageNotFoundError:
        return None
    except Exception:
        return None


def check_packages(rep: Report, pkgs: list[tuple[str, str, str]], required: bool) -> dict[str, object]:
    loaded: dict[str, object] = {}
    to_install: list[str] = []
    for mod_name, pip_name, why in pkgs:
        try:
            mod = importlib.import_module(mod_name)
        except Exception as exc:
            dist = _installed_version(pip_name)
            # Phân biệt 2 tình huống rất khác nhau mà cùng ném ModuleNotFoundError:
            #   (a) chưa cài  -> pip install là xong
            #   (b) CÓ cài nhưng import hỏng vì thiếu dependency của chính nó,
            #       hoặc build không khớp torch/numpy -> pip install lại vô ích.
            # Thông báo lỗi gốc mới là thứ chỉ đúng chỗ, nên luôn in nguyên văn.
            if dist is None:
                to_install.append(pip_name)
                rep.add(
                    FAIL if required else WARN,
                    f"{pip_name}: CHƯA CÀI ({type(exc).__name__}: {exc})",
                    f"dùng cho: {why}",
                )
            else:
                rep.add(
                    FAIL if required else WARN,
                    f"{pip_name} {dist}: ĐÃ CÀI nhưng import HỎNG "
                    f"({type(exc).__name__}: {exc})",
                    f"dùng cho: {why}\n"
                    f"pip install lại KHÔNG sửa được. Xem traceback đầy đủ:\n"
                    f"    python -c \"import {mod_name}\"\n"
                    f"Thiếu module khác -> cài module đó. Lỗi symbol/ABI -> bản\n"
                    f"{pip_name} không khớp torch/numpy đang cài, phải hạ hoặc đổi bản."
                    + (
                        "\nVới dgl, 3 thủ phạm hay gặp (đều không phải lỗi bản dgl):"
                        "\n  packaging/PyYAML/pydantic/psutil thiếu"
                        "\n      -> pip install packaging PyYAML pydantic psutil"
                        "\n  torchdata.datapipes thiếu"
                        "\n      -> pip install --no-deps 'torchdata==0.9.0'"
                        "\n         (torchdata >= 0.10 đã bỏ datapipes)"
                        "\n  libgraphbolt_pytorch_<torch>.so không tìm thấy"
                        "\n      -> python scripts/fix_dgl_windows.py"
                        "\n         dgl ship .so build sẵn CHO TỪNG BẢN TORCH; torch mới hơn"
                        "\n         mọi bản dgl thì file đó không tồn tại. Repo không dùng"
                        "\n         graphbolt nên script bỏ qua nó là an toàn."
                        if mod_name == "dgl"
                        else ""
                    ),
                )
            continue
        loaded[mod_name] = mod
        print(f"[{OK}] {pip_name} {_version_of(mod)}")
    if to_install:
        print(f"\n  -> pip install {' '.join(sorted(set(to_install)))}")
    return loaded


def check_torch_dgl(rep: Report, loaded: dict[str, object], want_gpu: bool) -> None:
    section("3. torch / dgl / GPU")
    torch = loaded.get("torch")
    if torch is None:
        rep.add(FAIL, "Không import được torch — bỏ qua phần còn lại của mục này")
        return

    cuda_build = getattr(getattr(torch, "version", None), "cuda", None)
    print(f"[{INFO}] torch {_version_of(torch)} (build CUDA: {cuda_build or 'CPU-only'})")
    try:
        has_cuda = bool(torch.cuda.is_available())
    except Exception as exc:
        has_cuda = False
        rep.add(WARN, f"torch.cuda.is_available() lỗi: {type(exc).__name__}: {exc}")
    if has_cuda:
        try:
            name = torch.cuda.get_device_name(0)
            total = torch.cuda.get_device_properties(0).total_memory
            rep.add(OK, f"GPU: {name} ({human(total)} VRAM)")
        except Exception:
            rep.add(OK, "GPU: có CUDA (không đọc được tên thiết bị)")
    elif want_gpu:
        rep.add(
            WARN,
            "Không có GPU — 5_build_seq_feature.py (ESM-2, ~23k protein) sẽ RẤT chậm trên CPU",
            "Cách né: chạy riêng bước đó trên Kaggle/Colab rồi tải\n"
            "dict_sequence_feature về. Các bước còn lại chạy CPU bình thường.",
        )
    else:
        rep.add(WARN, "Không có GPU — train sẽ rất chậm")

    dgl = loaded.get("dgl")
    if dgl is None:
        rep.add(FAIL, "Không import được dgl")
        return
    print(f"[{INFO}] dgl {_version_of(dgl)}")
    # Import được không đủ: dgl build sai torch/CUDA thường chỉ chết khi TẠO graph.
    try:
        g = dgl.graph((torch.tensor([0, 1]), torch.tensor([1, 2])), num_nodes=3)
        g.ndata["feat"] = torch.zeros(3, 4)
        sub = dgl.edge_subgraph(g, torch.tensor([0]), relabel_nodes=False)
        assert sub.num_nodes() == 3 and sub.num_edges() == 1
        rep.add(OK, "dgl tạo graph + edge_subgraph OK (khớp với torch đang cài)")
    except Exception as exc:
        rep.add(
            FAIL,
            f"dgl cài KHÔNG khớp torch: {type(exc).__name__}: {exc}",
            "Đây là lỗi phổ biến nhất. Gỡ rồi cài lại dgl đúng bản torch/CUDA\n"
            "đang dùng (xem mục hướng dẫn cài ở cuối).",
        )
    if has_cuda:
        try:
            dgl.graph((torch.tensor([0]), torch.tensor([1])), num_nodes=2).to("cuda")
            rep.add(OK, "dgl chuyển graph sang GPU OK")
        except Exception as exc:
            rep.add(
                WARN,
                f"dgl không dùng được GPU: {type(exc).__name__}: {exc}",
                "Bản dgl đang cài là CPU-only. Train vẫn chạy nhưng chậm hơn nhiều.",
            )


def _empty_pickle_kind(path: Path) -> str | None:
    """Trả về mô tả nếu file là pickle của container rỗng, ngược lại None."""
    try:
        if path.stat().st_size > 16:
            return None
        with open(path, "rb") as f:
            head = f.read(16)
    except Exception:
        return None
    return EMPTY_PICKLES.get(head)


def _dir_stats(path: Path, pattern: str = "*") -> tuple[int, int]:
    """(số file, tổng bytes) — dừng sớm nếu thư mục quá lớn để không treo."""
    count = 0
    size = 0
    for i, f in enumerate(path.rglob(pattern)):
        if i > 200_000:
            break
        if f.is_file():
            count += 1
            size += f.stat().st_size
    return count, size


def check_data(rep: Report, raw_dir: Path, data_dir: Path) -> None:
    section("4. Dữ liệu")
    proc = data_dir / "proceed_data"
    div = data_dir / "divided_data"
    print(f"[{INFO}] raw_dir : {raw_dir}")
    print(f"[{INFO}] data_dir: {data_dir}")

    print("\n-- Input thô (raw_data) --")
    for name, desc, used_by in RAW_ITEMS:
        p = raw_dir / name
        if p.is_dir():
            n, sz = _dir_stats(p, "*.pdb.gz")
            if n:
                rep.add(OK, f"{name}/: {n:,} file .pdb.gz ({human(sz)}) — {desc}")
            else:
                rep.add(WARN, f"{name}/: có thư mục nhưng KHÔNG có .pdb.gz — {desc}",
                        f"cần cho: {used_by}")
        elif p.is_file():
            rep.add(OK, f"{name}: {human(p.stat().st_size)} — {desc}")
        else:
            rep.add(WARN, f"{name}: THIẾU — {desc}", f"cần cho: {used_by}")

    print("\n-- Sản phẩm trung gian (proceed_data) --")
    if not proc.is_dir():
        rep.add(WARN, f"Chưa có {proc} — pipeline chưa chạy bước nào")
    else:
        for name, made_by, min_bytes in PROC_ITEMS:
            p = proc / name
            if p.is_dir():
                n, sz = _dir_stats(p)
                rep.add(OK if n else WARN, f"{name}/: {n:,} file ({human(sz)})")
            elif p.is_file():
                size = p.stat().st_size
                empty = _empty_pickle_kind(p)
                if empty:
                    rep.add(
                        FAIL,
                        f"{name}: {human(size)} — file RỖNG ({empty})",
                        f"Load được nhưng không có dữ liệu nào bên trong, nên mọi bước\n"
                        f"dùng nó sẽ âm thầm rơi về zero vector thay vì báo lỗi.\n"
                        f"Chạy lại {made_by} để sinh lại.",
                    )
                elif min_bytes and size < min_bytes:
                    rep.add(
                        FAIL,
                        f"{name}: {human(size)} — nhỏ bất thường (chờ >= {human(min_bytes)})",
                        f"Nhiều khả năng bị cắt cụt hoặc sinh ra từ input rỗng.\n"
                        f"Chạy lại {made_by}.",
                    )
                else:
                    rep.add(OK, f"{name}: {human(size)}")
            else:
                rep.add(WARN, f"{name}: chưa có — sinh ra bởi {made_by}")

    print("\n-- Dataset đã chia (divided_data) --")
    if not div.is_dir():
        rep.add(WARN, f"Chưa có {div} — chạy divide_data.py, hoặc giải nén bản pack")
    else:
        n, sz = _dir_stats(div)
        rep.add(OK if n else WARN, f"divided_data: {n} file ({human(sz)})")


def check_disk(rep: Report, raw_dir: Path, data_dir: Path) -> None:
    section("5. Dung lượng đĩa")
    seen: set[str] = set()
    for label, path in (("raw_dir", raw_dir), ("data_dir", data_dir)):
        probe = path
        while not probe.exists() and probe != probe.parent:
            probe = probe.parent
        try:
            usage = shutil.disk_usage(probe)
        except Exception as exc:
            rep.add(WARN, f"{label}: không đọc được dung lượng ({exc})")
            continue
        key = f"{usage.total}-{usage.free}"
        if key in seen:
            print(f"[{INFO}] {label}: cùng ổ đĩa với mục ở trên")
            continue
        seen.add(key)
        free_gb = usage.free / 1024**3
        rep.add(
            OK if free_gb >= 80 else WARN,
            f"{label} ({probe}): trống {human(usage.free)} / {human(usage.total)}",
            "" if free_gb >= 80 else
            "Ước lượng cần: PDB giải nén ~25-40 GB, proteins_edges vài GB,\n"
            "divided_data ~30 GB. Nên có >= 80 GB trống.",
        )


def print_install_help(target: str) -> None:
    section("Hướng dẫn cài")
    if target == "kaggle":
        print(
            "Kaggle đã cài sẵn: torch (CUDA), numpy, pandas, scikit-learn,\n"
            "transformers, tqdm, matplotlib. Chỉ THIẾU dgl:\n\n"
            "  !pip install dgl==2.1.0 -f https://data.dgl.ai/wheels/torch-2.1/cu121/repo.html\n\n"
            "Nếu bản torch của Kaggle đã đổi, thay 'torch-2.1/cu121' cho khớp\n"
            "(xem torch.__version__ và torch.version.cuda ở mục 3), rồi CHẠY LẠI\n"
            "script này để xác nhận dgl khớp torch.\n\n"
            "KHÔNG cần trên Kaggle: fair-esm, biopython, scipy, node2vec, networkx\n"
            "— đó là thư viện cho tiền xử lý ở máy local."
        )
        return
    activate = ".venv\\Scripts\\activate" if os.name == "nt" else "source .venv/bin/activate"
    print(
        "Máy local CHỈ chạy tiền xử lý (không train), nên không cần bản torch CUDA nặng.\n\n"
        f"  python -m venv .venv && {activate}\n"
        "  pip install --upgrade pip\n\n"
        "  # torch CPU là đủ cho tiền xử lý; muốn ESM-2 chạy nhanh thì cài bản CUDA\n"
        "  pip install torch --index-url https://download.pytorch.org/whl/cpu\n"
        "  pip install biopython scipy numpy pandas tqdm requests fair-esm\n\n"
        "  # chỉ khi rebuild protein_node2vec từ ppi.txt mới:\n"
        "  pip install networkx node2vec\n"
    )
    print(
        "  pip install dgl -f https://data.dgl.ai/wheels/repo.html\n\n"
        "  # Wheel dgl KHÔNG khai báo đủ dependency -> cài dgl xong vẫn thiếu, và\n"
        "  # mỗi cái chỉ lộ ra sau khi cái trước đã có. Cài trọn nhóm 1 lần:\n"
        "  pip install packaging PyYAML pydantic psutil\n"
        "  pip install --no-deps 'torchdata==0.9.0'   # >=0.10 đã bỏ torchdata.datapipes\n\n"
        "  # Còn lỗi 'Cannot find DGL C++ graphbolt library ... .so' -> vá thẳng dgl:\n"
        "  python scripts/fix_dgl_windows.py   # chạy được cả Linux/macOS, không riêng Windows\n"
        "  # (bỏ qua graphbolt + distributed — repo chỉ dùng core graph API)\n\n"
        "  # Pip không tìm được wheel nào -> tạo env Python 3.11, đừng build từ nguồn.\n"
    )
    print(
        "Cài xong chạy lại script này — mục 3 sẽ kiểm tra dgl có khớp torch không\n"
        "(import được KHÔNG có nghĩa là dùng được)."
    )


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra môi trường chạy pipeline CAFA6",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target", choices=["local", "kaggle", "auto"], default="auto",
        help="local = tiền xử lý dữ liệu; kaggle = train/eval",
    )
    # D:/raw_data là đường dẫn trong docstring của các script (viết trên Windows).
    # Trên Linux/macOS nó không tồn tại và mọi kiểm tra raw_data sẽ báo THIẾU sai,
    # nên mặc định đổi sang <data_dir>/../raw_data.
    default_raw = os.environ.get("RAW_DIR")
    if not default_raw:
        default_raw = "D:/raw_data" if os.name == "nt" else str(_repo_root().parent / "raw_data")
    parser.add_argument("--raw-dir", type=Path, default=Path(default_raw))
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path(os.environ.get("DATA_DIR", Path(__file__).resolve().parents[1])),
    )
    args = parser.parse_args()

    target = args.target
    if target == "auto":
        target = "kaggle" if Path("/kaggle").is_dir() else "local"

    rep = Report()
    print(f"MỤC TIÊU: {target}  (đổi bằng --target)")
    check_python(rep)

    section("2. Thư viện")
    if target == "kaggle":
        loaded = check_packages(rep, KAGGLE_PKGS, required=True)
    else:
        loaded = check_packages(rep, LOCAL_PKGS, required=True)
        print("\n-- Tuỳ chọn --")
        loaded.update(check_packages(rep, LOCAL_OPTIONAL_PKGS, required=False))

    check_torch_dgl(rep, loaded, want_gpu=(target == "local"))
    check_data(rep, args.raw_dir, args.data_dir)
    check_disk(rep, args.raw_dir, args.data_dir)
    print_install_help(target)

    section("TỔNG KẾT")
    if rep.fails:
        print(f"{len(rep.fails)} lỗi PHẢI sửa trước khi chạy:")
        for m in rep.fails:
            print(f"  - {m}")
    if rep.warns:
        print(f"\n{len(rep.warns)} cảnh báo (thường là file dữ liệu chưa có — bình thường "
              f"nếu bạn chưa chạy bước đó):")
        for m in rep.warns:
            print(f"  - {m}")
    if not rep.fails:
        print("\nMôi trường OK — chạy được.")
    print("=" * 70)
    return 1 if rep.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
