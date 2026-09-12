#!/usr/bin/env python3
"""Kiểm tra NỘI DUNG dữ liệu đầu vào trước khi chạy pipeline.

Khác với 2 script kia:
    check_env.py    — thư viện + file có tồn tại / đủ lớn không
    check_inputs.py — (script này) MỞ file ra đọc, và ĐỐI CHIẾU CHÉO giữa chúng
    audit_data.py   — artifact CUỐI pipeline (split/vocab/PPI guard/chiều nhãn)

Vì sao cần: file tải bằng wget/curl rất hay là trang lỗi HTML hoặc bị cắt giữa
chừng mà kích thước vẫn "trông hợp lý". Và quan trọng hơn — số protein sống sót
qua pipeline KHÔNG phải số protein trong từng file, mà là GIAO của chúng. Biết
con số đó TRƯỚC khi chạy 3_build_graph_dataset.py (hàng giờ) rẻ hơn nhiều so với
phát hiện sau.

Chạy:
    python scripts/check_inputs.py
    python scripts/check_inputs.py --raw-dir ~/raw_data --data-dir ~/CAFA6
    python scripts/check_inputs.py --skip-ppi        # bỏ qua quét ppi.txt (~13M dòng)

Exit code: 0 = dùng được, 1 = có FAIL phải sửa.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import pickle
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_env import FAIL, INFO, OK, WARN, Report, section  # noqa: E402

BRANCHES = ("BP", "MF", "CC")


def _read_max_seq_len(default: int = 5000) -> int:
    """Đọc MAX_SEQ_LEN THẲNG TỪ 5_build_seq_feature.py thay vì chép lại.

    Chép cứng thì sửa 1 nơi quên nơi kia, và cảnh báo ở đây sẽ nói sai số protein
    bị bỏ. Dùng regex vì file đó import torch/esm — không import được ở môi trường
    chỉ chạy kiểm tra.
    """
    src = Path(__file__).resolve().parents[1] / "data_processing" / "5_build_seq_feature.py"
    try:
        m = re.search(r"^MAX_SEQ_LEN\s*=\s*(\d+)", src.read_text(encoding="utf-8"), re.M)
        return int(m.group(1)) if m else default
    except OSError:
        return default


MAX_SEQ_LEN = _read_max_seq_len()


def _pct(part: int, whole: int) -> str:
    return f"{100.0 * part / whole:.1f}%" if whole else "n/a"


def _head_bytes(path: Path, n: int = 512) -> bytes:
    opener = gzip.open if path.suffix == ".gz" else open
    try:
        with opener(path, "rb") as f:
            return f.read(n)
    except Exception:
        return b""


def _looks_like_html(head: bytes) -> bool:
    """wget một URL hỏng thường lưu về trang lỗi HTML chứ không báo lỗi."""
    s = head.lstrip()[:200].lower()
    return s.startswith(b"<!doctype html") or s.startswith(b"<html") or b"<title>" in s


# ── raw_data ─────────────────────────────────────────────────────────────────
def check_gaf(rep: Report, path: Path) -> set[str]:
    """goa_human.gaf.gz -> tập UniProt ID có annotation."""
    if not path.is_file():
        rep.add(WARN, f"{path.name}: chưa tải")
        return set()
    head = _head_bytes(path)
    if _looks_like_html(head):
        rep.add(FAIL, f"{path.name}: là trang HTML, không phải GAF — tải lại")
        return set()
    if not head.startswith(b"!gaf-version"):
        rep.add(WARN, f"{path.name}: không bắt đầu bằng '!gaf-version' — định dạng lạ")

    ids: set[str] = set()
    terms: set[str] = set()
    rows = 0
    try:
        with gzip.open(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith("!"):
                    continue
                cols = line.rstrip("\n").split("\t")
                if len(cols) < 5:
                    continue
                rows += 1
                ids.add(cols[1])
                terms.add(cols[4])
    except Exception as exc:
        rep.add(FAIL, f"{path.name}: đọc lỗi ({type(exc).__name__}: {exc}) — file hỏng, tải lại")
        return set()
    rep.add(
        OK if rows > 100_000 else WARN,
        f"{path.name}: {rows:,} annotation, {len(ids):,} protein, {len(terms):,} GO term",
    )
    return ids


def check_fasta(rep: Report, path: Path) -> dict[str, int]:
    """seq.fasta -> {uniprot_id: độ dài chuỗi}. Parse y hệt 5_build_seq_feature.py."""
    if not path.is_file():
        rep.add(WARN, f"{path.name}: chưa tải")
        return {}
    head = _head_bytes(path)
    if _looks_like_html(head):
        rep.add(
            FAIL,
            f"{path.name}: là trang HTML, không phải FASTA — tải lại",
            "REST API của UniProt trả HTML khi query sai; wget vẫn lưu file bình thường.",
        )
        return {}
    if not head.lstrip().startswith(b">"):
        rep.add(FAIL, f"{path.name}: không bắt đầu bằng '>' — không phải FASTA")
        return {}

    lengths: dict[str, int] = {}
    bad_header = 0
    cur_id: str | None = None
    cur_len = 0
    try:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                if line.startswith(">"):
                    if cur_id is not None:
                        lengths[cur_id] = cur_len
                    # header: sp|P12345|NAME_HUMAN ...  -> lấy trường giữa
                    parts = line[1:].split()[0].split("|")
                    if len(parts) >= 2:
                        cur_id = parts[1]
                    else:
                        cur_id = parts[0]
                        bad_header += 1
                    cur_len = 0
                else:
                    cur_len += len(line.strip())
        if cur_id is not None:
            lengths[cur_id] = cur_len
    except Exception as exc:
        rep.add(FAIL, f"{path.name}: đọc lỗi ({type(exc).__name__}: {exc})")
        return {}

    if not lengths:
        rep.add(FAIL, f"{path.name}: không đọc được record nào")
        return {}
    avg = sum(lengths.values()) / len(lengths)
    rep.add(OK, f"{path.name}: {len(lengths):,} chuỗi, dài trung bình {avg:,.0f} residue")
    if bad_header:
        rep.add(
            WARN,
            f"{path.name}: {bad_header:,} header không có dạng sp|ID|NAME",
            "5_build_seq_feature.py lấy ID ở trường giữa của header — sai dạng thì ID "
            "sẽ không khớp valid_protein_ids.csv.",
        )
    return lengths


def check_ppi(rep: Report, path: Path, skip: bool) -> set[str]:
    """ppi.txt (STRING) -> tập ENSP xuất hiện. Trả về set rỗng nếu bỏ qua."""
    if not path.is_file():
        rep.add(WARN, f"{path.name}: chưa tải")
        return set()
    head = _head_bytes(path)
    if _looks_like_html(head):
        rep.add(FAIL, f"{path.name}: là trang HTML, không phải STRING — tải lại")
        return set()
    first = head.split(b"\n", 1)[0].decode("utf-8", "replace").strip()
    if "protein1" not in first:
        rep.add(
            WARN,
            f"{path.name}: dòng đầu không phải header STRING",
            f"đọc được: {first[:80]!r}\n"
            "4_build_ppi_graph.py bỏ dòng đầu vô điều kiện — header sai thì mất 1 cạnh, "
            "không nghiêm trọng, nhưng nên kiểm tra đúng file chưa.",
        )
    if skip:
        rep.add(INFO, f"{path.name}: bỏ qua quét nội dung (--skip-ppi)")
        return set()

    t0 = time.perf_counter()
    ensps: set[str] = set()
    rows = 0
    kept = 0
    bad = 0
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            next(f, None)
            for line in f:
                parts = line.split()
                if len(parts) < 3:
                    bad += 1
                    continue
                rows += 1
                try:
                    score = int(parts[2])
                except ValueError:
                    bad += 1
                    continue
                # 4_build_ppi_graph.py lọc combined_score >= 700
                if score >= 700:
                    kept += 1
                    for tok in (parts[0], parts[1]):
                        if "." in tok:
                            ensps.add(tok.split(".", 1)[1])
    except Exception as exc:
        rep.add(FAIL, f"{path.name}: đọc lỗi ({type(exc).__name__}: {exc})")
        return set()

    rep.add(
        OK if kept else FAIL,
        f"{path.name}: {rows:,} cạnh, {kept:,} cạnh có score >= 700 ({_pct(kept, rows)}), "
        f"{len(ensps):,} ENSP  [{time.perf_counter() - t0:.0f}s]",
        "" if kept else "Không cạnh nào qua ngưỡng 700 — sai cột score hoặc sai file.",
    )
    if bad:
        rep.add(WARN, f"{path.name}: {bad:,} dòng không parse được (bỏ qua)")
    return ensps


# ── proceed_data ─────────────────────────────────────────────────────────────
def read_valid_ids(rep: Report, path: Path) -> set[str]:
    if not path.is_file():
        rep.add(FAIL, f"{path.name}: không có — chạy 1_get_valid_ids.py")
        return set()
    ids: set[str] = set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            col = "Protein_ID" if reader.fieldnames and "Protein_ID" in reader.fieldnames else None
            if col is None:
                f.seek(0)
                ids = {ln.strip() for ln in f if ln.strip()}
            else:
                ids = {r[col].strip() for r in reader if r.get(col)}
    except Exception as exc:
        rep.add(FAIL, f"{path.name}: đọc lỗi ({type(exc).__name__}: {exc})")
        return set()
    rep.add(OK if ids else FAIL, f"{path.name}: {len(ids):,} protein ID")
    return ids


def read_edge_ids(rep: Report, edges_dir: Path) -> set[str]:
    if not edges_dir.is_dir():
        rep.add(FAIL, f"{edges_dir.name}/: không có — chạy 2_extract_struct_map.py")
        return set()
    ids = {p.stem for p in edges_dir.glob("*.txt")}
    if not ids:
        rep.add(FAIL, f"{edges_dir.name}/: không có file .txt nào")
        return set()
    # Đọc thử vài file: rỗng hoặc sai định dạng thì protein đó bị bỏ lúc build graph.
    empty = 0
    sample = sorted(ids)[:: max(len(ids) // 200, 1)][:200]
    for pid in sample:
        p = edges_dir / f"{pid}.txt"
        try:
            if p.stat().st_size == 0:
                empty += 1
        except OSError:
            empty += 1
    rep.add(OK, f"{edges_dir.name}/: {len(ids):,} contact map")
    if empty:
        rep.add(
            WARN,
            f"{edges_dir.name}/: {empty}/{len(sample)} file mẫu rỗng "
            f"(~{_pct(empty, len(sample))} tổng thể)",
            "Protein có contact map rỗng bị bỏ qua lúc build dataset.",
        )
    return ids


def read_acs(rep: Report, proc: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    for ns in BRANCHES:
        path = proc / f"human_{ns}_ACS.json"
        if not path.is_file():
            rep.add(WARN, f"human_{ns}_ACS.json: chưa có — chạy 2_build_go_namespace.py")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as exc:
            rep.add(FAIL, f"human_{ns}_ACS.json: JSON hỏng ({type(exc).__name__}: {exc})")
            continue
        terms = {t for v in data.values() for t in v}
        avg = sum(len(v) for v in data.values()) / max(len(data), 1)
        rep.add(
            OK,
            f"human_{ns}_ACS.json: {len(data):,} protein, {len(terms):,} GO term, "
            f"{avg:.1f} nhãn/protein",
        )
        out[ns] = data
    return out


def read_mapping(rep: Report, path: Path) -> dict[str, str]:
    if not path.is_file():
        rep.add(WARN, f"{path.name}: chưa có — chạy 3_uniprot_mapping.py")
        return {}
    m: dict[str, str] = {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                ac = (row.get("UniProtKB_AC") or "").strip()
                ensp = (row.get("Ensembl_Protein") or "").strip()
                if ac and ensp:
                    m[ensp] = ac
    except Exception as exc:
        rep.add(FAIL, f"{path.name}: đọc lỗi ({type(exc).__name__}: {exc})")
        return {}
    rep.add(OK if m else FAIL, f"{path.name}: {len(m):,} ánh xạ ENSP -> UniProt")
    return m


def read_pickle_keys(rep: Report, path: Path, made_by: str) -> set[str] | None:
    """Chỉ lấy tập key — đủ để đối chiếu coverage, không cần giữ cả giá trị."""
    if not path.is_file():
        rep.add(WARN, f"{path.name}: chưa có — sinh ra bởi {made_by}")
        return None
    try:
        with open(path, "rb") as f:
            obj = pickle.load(f)
    except Exception as exc:
        rep.add(
            FAIL,
            f"{path.name}: pickle hỏng ({type(exc).__name__}: {exc})",
            f"Chạy lại {made_by}.",
        )
        return None
    if not isinstance(obj, dict):
        rep.add(FAIL, f"{path.name}: không phải dict mà là {type(obj).__name__}")
        return None
    if not obj:
        rep.add(FAIL, f"{path.name}: dict RỖNG — chạy lại {made_by}")
        return set()
    first = next(iter(obj.values()))
    shape = getattr(first, "shape", None)
    desc = f"shape {tuple(shape)}" if shape is not None else f"{type(first).__name__}"
    try:
        desc += f", len={len(first)}"
    except TypeError:
        pass
    rep.add(OK, f"{path.name}: {len(obj):,} protein (giá trị mẫu: {desc})")
    return set(obj.keys())


# ── đối chiếu chéo ───────────────────────────────────────────────────────────
def cross_check(
    rep: Report,
    struct_ids: set[str],
    valid_ids: set[str],
    fasta_len: dict[str, int],
    acs: dict[str, dict],
    mapping: dict[str, str],
    ppi_ensps: set[str],
    node2vec_ids: set[str] | None,
    onehot_ids: set[str] | None,
    seqfeat_ids: set[str] | None,
) -> None:
    section("3. Đối chiếu chéo — bao nhiêu protein thật sự sống sót")
    if not struct_ids:
        rep.add(FAIL, "Không có contact map nào — không đối chiếu được")
        return

    only_valid = valid_ids - struct_ids
    only_struct = struct_ids - valid_ids
    if only_valid or only_struct:
        rep.add(
            WARN,
            f"valid_protein_ids.csv vs proteins_edges/: lệch "
            f"({len(only_valid):,} chỉ ở csv, {len(only_struct):,} chỉ có contact map)",
            "Bình thường nếu 2_extract_struct_map.py bỏ qua vài PDB lỗi.",
        )
    else:
        rep.add(OK, "valid_protein_ids.csv khớp đúng proteins_edges/")

    fasta_ids = set(fasta_len)
    if fasta_ids:
        hit = len(struct_ids & fasta_ids)
        rep.add(
            OK if hit >= 0.9 * len(struct_ids) else FAIL,
            f"seq.fasta phủ {hit:,}/{len(struct_ids):,} protein có cấu trúc "
            f"({_pct(hit, len(struct_ids))})",
            "" if hit >= 0.9 * len(struct_ids) else
            "Phủ thấp -> dict_sequence_feature sẽ thiếu, và 3_build_graph_dataset.py\n"
            "lặng lẽ dùng ZERO VECTOR cho nhánh sequence. Kiểm tra FASTA có đúng\n"
            "proteome UP000005640 và header dạng sp|ID|NAME không.",
        )

    # 5_build_seq_feature.py có MAX_SEQ_LEN = 2000 và BỎ HẲN protein dài hơn —
    # chúng sẽ không có trong dict_sequence_feature, và 3_build_graph_dataset.py
    # lặng lẽ thay bằng zero vector cho TOÀN BỘ nhánh sequence của protein đó.
    # Biết con số này trước khi chạy ESM-2 (hàng chục phút trên GPU) thì hơn.
    if fasta_len:
        too_long = [p for p in struct_ids if fasta_len.get(p, 0) > MAX_SEQ_LEN]
        rep.add(
            OK if len(too_long) < 0.02 * len(struct_ids) else WARN,
            f"Dài hơn MAX_SEQ_LEN={MAX_SEQ_LEN}: {len(too_long):,}/{len(struct_ids):,} protein "
            f"({_pct(len(too_long), len(struct_ids))})",
            f"5_build_seq_feature.py bỏ qua chúng -> zero vector cho cả nhánh sequence.\n"
            f"Muốn giữ thì sửa MAX_SEQ_LEN trong script đó (tốn VRAM hơn)."
            if too_long else "",
        )

    for ns, data in acs.items():
        hit = len(struct_ids & set(data.keys()))
        rep.add(
            OK if hit else FAIL,
            f"Nhánh {ns}: {hit:,} protein vừa có cấu trúc vừa có nhãn "
            f"({_pct(hit, len(struct_ids))} số protein có cấu trúc)",
            "" if hit else "Không giao nhau -> sai định dạng ID giữa 2 nguồn.",
        )
    if acs:
        print(f"[{INFO}] Con số trên chính là kích thước dataset cuối của từng nhánh.")

    if mapping:
        acs_in_struct = len(set(mapping.values()) & struct_ids)
        rep.add(
            OK if acs_in_struct else FAIL,
            f"uniprot_ensembl_mapping.csv: {acs_in_struct:,} UniProt AC có trong "
            f"proteins_edges/ ({_pct(acs_in_struct, len(struct_ids))})",
        )
        if ppi_ensps:
            covered = len(ppi_ensps & set(mapping.keys()))
            rep.add(
                OK if covered >= 0.5 * len(ppi_ensps) else WARN,
                f"mapping phủ {covered:,}/{len(ppi_ensps):,} ENSP trong ppi.txt "
                f"({_pct(covered, len(ppi_ensps))})",
                "" if covered >= 0.5 * len(ppi_ensps) else
                "Phủ thấp -> mất nhiều cạnh PPI. Nếu bạn vừa tải ppi.txt phiên bản\n"
                "MỚI hơn lúc tạo mapping, chạy lại 3_uniprot_mapping.py.",
            )

    # protein_node2vec KHÁC 2 cái kia: nó chỉ dựng được cho protein CÓ MẶT trong
    # đồ thị PPI (score >= 700). Protein không có tương tác tin cậy nào thì không
    # có embedding — đó là bản chất dữ liệu, không phải lỗi build. Nên ngưỡng thấp
    # hơn và chỉ WARN. onehot/seqfeat thì dựng từ chuỗi nên PHẢI phủ gần hết.
    for name, ids, need, floor, sev in (
        ("protein_node2vec", node2vec_ids, "node feature 30 chiều", 0.5, WARN),
        ("protein_node2onehot", onehot_ids, "node feature 26 chiều", 0.9, FAIL),
        ("dict_sequence_feature", seqfeat_ids, "toàn bộ nhánh sequence", 0.9, FAIL),
    ):
        if ids is None:
            continue
        hit = len(struct_ids & ids)
        good = hit >= floor * len(struct_ids)
        if name == "protein_node2vec":
            detail = (
                f"{len(struct_ids) - hit:,} protein còn lại không có cạnh PPI nào đạt "
                "score >= 700\n"
                f"nên nhận zero vector cho {need}. Bình thường với dữ liệu STRING."
            )
            if not good:
                detail += "\nDưới 50% thì nên xem lại mapping hoặc ngưỡng --min-score."
        else:
            detail = "" if good else (
                f"Protein không được phủ sẽ nhận ZERO VECTOR cho {need} — "
                "không có lỗi nào được in ra lúc build."
            )
        rep.add(
            OK if good else sev,
            f"{name} phủ {hit:,}/{len(struct_ids):,} protein ({_pct(hit, len(struct_ids))})",
            detail,
        )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Kiểm tra nội dung dữ liệu đầu vào của pipeline CAFA6",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    repo = Path(__file__).resolve().parents[1]
    default_raw = os.environ.get("RAW_DIR") or (
        "D:/raw_data" if os.name == "nt" else str(repo.parent / "raw_data")
    )
    parser.add_argument("--raw-dir", type=Path, default=Path(default_raw))
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", repo)))
    parser.add_argument(
        "--skip-ppi", action="store_true",
        help="Bỏ qua quét ppi.txt (~13M dòng, mất vài chục giây)",
    )
    args = parser.parse_args()

    raw = args.raw_dir
    proc = args.data_dir / "proceed_data"
    print(f"raw_dir : {raw}\ndata_dir: {args.data_dir}")
    rep = Report()

    section("1. raw_data — đọc thật, không chỉ xem kích thước")
    gaf_ids = check_gaf(rep, raw / "goa_human.gaf.gz")
    fasta_len = check_fasta(rep, raw / "seq.fasta")
    ppi_ensps = check_ppi(rep, raw / "ppi.txt", args.skip_ppi)

    section("2. proceed_data")
    valid_ids = read_valid_ids(rep, proc / "valid_protein_ids.csv")
    struct_ids = read_edge_ids(rep, proc / "proteins_edges")
    acs = read_acs(rep, proc)
    mapping = read_mapping(rep, proc / "uniprot_ensembl_mapping.csv")
    node2vec_ids = read_pickle_keys(rep, proc / "protein_node2vec", "4_build_ppi_node2vec.py")
    onehot_ids = read_pickle_keys(rep, proc / "protein_node2onehot", "get_sequence.py")
    seqfeat_ids = read_pickle_keys(rep, proc / "dict_sequence_feature", "5_build_seq_feature.py")

    cross_check(
        rep, struct_ids, valid_ids, fasta_len, acs, mapping, ppi_ensps,
        node2vec_ids, onehot_ids, seqfeat_ids,
    )
    if gaf_ids and acs:
        section("4. GAF vs ACS.json")
        total_acs = set()
        for data in acs.values():
            total_acs |= set(data.keys())
        stale = total_acs - gaf_ids
        rep.add(
            OK if not stale else WARN,
            f"ACS.json có {len(total_acs):,} protein; {len(stale):,} trong số đó "
            f"KHÔNG có trong goa_human.gaf.gz vừa tải",
            "" if not stale else
            "ACS.json đang build từ bản GAF CŨ. Muốn data sạch thì chạy lại\n"
            "go_anno.py + 2_build_go_namespace.py với GAF mới tải.",
        )

    section("TỔNG KẾT")
    if rep.fails:
        print(f"{len(rep.fails)} lỗi PHẢI sửa:")
        for m in rep.fails:
            print(f"  - {m}")
    if rep.warns:
        print(f"\n{len(rep.warns)} cảnh báo:")
        for m in rep.warns:
            print(f"  - {m}")
    if not rep.fails:
        print("\nDữ liệu đầu vào dùng được.")
    print("=" * 70)
    return 1 if rep.fails else 0


if __name__ == "__main__":
    raise SystemExit(main())
