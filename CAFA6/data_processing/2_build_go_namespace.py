"""Build GO namespace files needed by the Kaggle pipeline.

Inputs:
  - proceed_data/HUMAN_protein_info.json  (protein -> raw GO terms)
  - raw_data/go.obo                       (GO term namespace and parents)
  - proceed_data/valid_protein_ids.csv     (optional filter to structure proteins)

Outputs:
  - proceed_data/human_BP_ACS.json
  - proceed_data/human_MF_ACS.json
  - proceed_data/human_CC_ACS.json
  - proceed_data/label_vocab_bp.json
  - proceed_data/label_vocab_mf.json
  - proceed_data/label_vocab_cc.json

The label vocabularies are sorted GO term lists, matching the index order used
by data_processing/3_build_graph_dataset.py and eval_Struct2GO2.py.
"""

from __future__ import annotations

import argparse
import json
import shutil
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path


BASE_DIR = Path("D:/CAFA6")
RAW_DIR = Path("D:/raw_data")
PROC_DIR = BASE_DIR / "proceed_data"
GO_OBO_URLS = [
    "https://current.geneontology.org/ontology/go-basic.obo",
    "https://snapshot.geneontology.org/ontology/go-basic.obo",
    "https://purl.obolibrary.org/obo/go/go-basic.obo",
]


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def _resolve_go_obo(path: Path) -> Path:
    candidates = [
        path,
        BASE_DIR / "go.obo",
        BASE_DIR / "raw_data" / "go.obo",
        RAW_DIR / "go.obo",
        RAW_DIR / "go-basic.obo",
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate

    download_to = RAW_DIR / "go-basic.obo"
    download_to.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "Mozilla/5.0"}
    last_error: Exception | None = None
    for url in GO_OBO_URLS:
        print(f"[INFO] Downloading GO OBO from {url} -> {download_to}")
        request = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(request, timeout=120) as response, download_to.open("wb") as handle:
                shutil.copyfileobj(response, handle)
            return download_to
        except Exception as exc:
            last_error = exc
            print(f"[WARN] Failed to download from {url}: {exc}")
    raise FileNotFoundError(
        "Missing GO OBO file and all download mirrors failed. "
        "Please place go.obo in D:/raw_data or D:/CAFA6/raw_data."
    ) from last_error


def _load_valid_ids(path: Path | None) -> set[str]:
    if path is None or not path.is_file():
        return set()
    with path.open("r", encoding="utf-8") as handle:
        lines = [line.strip() for line in handle if line.strip()]
    if not lines:
        return set()
    if lines[0].lower().startswith("protein_id"):
        lines = lines[1:]
    return set(lines)


def _parse_go_obo(path: Path) -> tuple[dict[str, str], dict[str, set[str]]]:
    namespace: dict[str, str] = {}
    parents: dict[str, set[str]] = defaultdict(set)
    current_term: str | None = None

    with path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            if line == "[Term]":
                current_term = None
                continue
            if line == "[Typedef]":
                break
            if line.startswith("id: GO:"):
                current_term = line.split()[1]
                continue
            if current_term is None:
                continue
            if line.startswith("namespace:"):
                namespace[current_term] = line.split()[1]
                continue
            if line.startswith("is_a:"):
                parents[current_term].add(line.split()[1])
                continue
            if line.startswith("relationship: part_of"):
                parts = line.split()
                if len(parts) >= 3:
                    parents[current_term].add(parts[2])

    return namespace, parents


def _propagate_terms(terms: set[str], parents: dict[str, set[str]]) -> set[str]:
    expanded = set(terms)
    while True:
        before = len(expanded)
        next_terms = set()
        for term in expanded:
            next_terms.update(parents.get(term, set()))
        expanded.update(next_terms)
        if len(expanded) == before:
            return expanded


def main() -> None:
    parser = argparse.ArgumentParser(description="Build GO namespace ACS/vocab files")
    parser.add_argument(
        "--protein-info",
        type=Path,
        default=PROC_DIR / "HUMAN_protein_info.json",
        help="Path to HUMAN_protein_info.json",
    )
    parser.add_argument(
        "--go-obo",
        type=Path,
        default=RAW_DIR / "go.obo",
        help="Path to go.obo",
    )
    parser.add_argument(
        "--valid-ids",
        type=Path,
        default=PROC_DIR / "valid_protein_ids.csv",
        help="Optional CSV of valid protein IDs to keep",
    )
    parser.add_argument(
        "--min-bp",
        type=int,
        default=250,
        help="Minimum protein count for BP GO terms",
    )
    parser.add_argument(
        "--min-other",
        type=int,
        default=100,
        help="Minimum protein count for MF/CC GO terms",
    )
    args = parser.parse_args()

    if not args.protein_info.is_file():
        raise FileNotFoundError(f"Missing protein info file: {args.protein_info}")
    args.go_obo = _resolve_go_obo(args.go_obo)
    print(f"[INFO] Using GO OBO: {args.go_obo}")

    protein_info = _load_json(args.protein_info)
    valid_ids = _load_valid_ids(args.valid_ids)
    namespace, parents = _parse_go_obo(args.go_obo)

    by_ns: dict[str, dict[str, list[str]]] = {"bp": {}, "mf": {}, "cc": {}}
    all_terms: dict[str, Counter[str]] = {"bp": Counter(), "mf": Counter(), "cc": Counter()}

    ns_map = {
        "biological_process": "bp",
        "molecular_function": "mf",
        "cellular_component": "cc",
    }

    for protein_id, raw_terms in protein_info.items():
        if valid_ids and protein_id not in valid_ids:
            continue
        propagated = _propagate_terms(set(raw_terms), parents)
        per_ns: dict[str, list[str]] = {"bp": [], "mf": [], "cc": []}
        for term in propagated:
            ns = ns_map.get(namespace.get(term, ""))
            if ns is None:
                continue
            per_ns[ns].append(term)
            all_terms[ns][term] += 1
        for ns in per_ns:
            if per_ns[ns]:
                by_ns[ns][protein_id] = sorted(set(per_ns[ns]))

    min_counts = {"bp": args.min_bp, "mf": args.min_other, "cc": args.min_other}
    for ns in ("bp", "mf", "cc"):
        acs_path = PROC_DIR / f"human_{ns.upper()}_ACS.json"
        vocab_path = PROC_DIR / f"label_vocab_{ns}.json"

        acs = by_ns[ns]
        _write_json(acs_path, acs)

        selected_terms = sorted(
            term for term, count in all_terms[ns].items() if count >= min_counts[ns]
        )
        _write_json(vocab_path, selected_terms)

        print(
            f"[{ns}] proteins={len(acs):,} terms={len(selected_terms):,} "
            f"-> {acs_path.name}, {vocab_path.name}"
        )


if __name__ == "__main__":
    main()