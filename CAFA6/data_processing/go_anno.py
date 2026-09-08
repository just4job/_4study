"""Parse GOA GAF into HUMAN_protein_info.json.

Default paths match the CAFA6 workspace layout:
  - raw_data/goa_human.gaf.gz
  - proceed_data/HUMAN_protein_info.json
"""

from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path


BASE_DIR = Path("D:/CAFA6")
RAW_FILE = Path("D:/raw_data/goa_human.gaf.gz")
OUT_FILE = BASE_DIR / "proceed_data" / "HUMAN_protein_info.json"


def main() -> None:
    parser = argparse.ArgumentParser(description="Build HUMAN_protein_info.json from GOA GAF")
    parser.add_argument("--input", type=Path, default=RAW_FILE, help="Path to goa_human.gaf.gz")
    parser.add_argument("--output", type=Path, default=OUT_FILE, help="Output HUMAN_protein_info.json")
    args = parser.parse_args()

    if not args.input.is_file():
        raise FileNotFoundError(f"Missing GOA file: {args.input}")

    go_annotations: dict[str, list[str]] = {}
    with gzip.open(args.input, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            cols = line.rstrip().split("\t")
            if len(cols) < 5:
                continue
            protein_id = cols[1]
            go_id = cols[4]
            if not protein_id or not go_id:
                continue
            go_annotations.setdefault(protein_id, []).append(go_id)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        json.dump(go_annotations, handle, indent=4)

    print(f"Saved {len(go_annotations):,} proteins -> {args.output}")


if __name__ == "__main__":
    main()
