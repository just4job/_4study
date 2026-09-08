"""
UniProt ID Mapping Script
--------------------------
Input : 9606.protein.links.v12.0.txt  (STRING database)
Output: uniprot_ensembl_mapping.csv
Columns: UniProtKB_AC, Ensembl_Protein

Workflow:
  1. Extract unique ENSP IDs from the STRING file.
  2. Submit a job to the UniProt ID Mapping REST API
     (From: Ensembl_Protein  →  To: UniProtKB)
  3. Poll until the job finishes, then download + save as CSV.
"""

import argparse
import csv
import time
from pathlib import Path

import requests

# ── Configuration ──────────────────────────────────────────────────────────────
DEFAULT_INPUT_FILE = Path(r"D:\raw_data\ppi.txt")
DEFAULT_OUTPUT_FILE = Path(r"D:\CAFA6\proceed_data\uniprot_ensembl_mapping.csv")

UNIPROT_API = "https://rest.uniprot.org"
BATCH_SIZE = 500       # UniProt recommends <= 500 IDs per batch
POLL_SECONDS = 5       # seconds between status checks
MAX_RETRIES = 60       # give up after MAX_RETRIES * POLL_SECONDS seconds
REQUEST_TIMEOUT = 60
MAX_HTTP_RETRIES = 3

def request_with_retry(method: str, url: str, **kwargs) -> requests.Response:
    """Send an HTTP request with simple retry for transient network/server errors."""
    last_err: Exception | None = None
    for attempt in range(1, MAX_HTTP_RETRIES + 1):
        try:
            resp = requests.request(method, url, timeout=REQUEST_TIMEOUT, **kwargs)
            if resp.status_code in {429, 500, 502, 503, 504} and attempt < MAX_HTTP_RETRIES:
                time.sleep(attempt * 2)
                continue
            resp.raise_for_status()
            return resp
        except requests.RequestException as exc:
            last_err = exc
            if attempt == MAX_HTTP_RETRIES:
                break
            time.sleep(attempt * 2)
    raise RuntimeError(f"HTTP request failed after retries: {method} {url}") from last_err


def parse_ensp(token: str) -> str | None:
    """Parse STRING token like '9606.ENSP...' into ENSP id; return None on invalid format."""
    if "." not in token:
        return None
    _, ensp = token.split(".", 1)
    if not ensp.startswith("ENSP"):
        return None
    return ensp


def extract_unique_ensp_ids(input_file: Path) -> list[str]:
    print("Step 1 - Reading STRING file and extracting unique ENSP IDs ...")
    ensp_ids: set[str] = set()
    if not input_file.exists():
        raise FileNotFoundError(f"Input file not found: {input_file}")

    with input_file.open("rt", encoding="utf-8") as fh:
        for line in fh:
            parts = line.rstrip().split()
            if len(parts) < 2:
                continue
            ensp1 = parse_ensp(parts[0])
            ensp2 = parse_ensp(parts[1])
            if ensp1:
                ensp_ids.add(ensp1)
            if ensp2:
                ensp_ids.add(ensp2)
    ensp_list = sorted(ensp_ids)
    print(f"  Found {len(ensp_list):,} unique Ensembl Protein IDs.")
    return ensp_list


def submit_batch(ids: list[str]) -> str:
    """Submit a mapping job and return the job ID."""
    response = request_with_retry(
        "POST",
        f"{UNIPROT_API}/idmapping/run",
        data={
            "ids": ",".join(ids),
            "from": "Ensembl_Protein",
            "to": "UniProtKB",
        },
    )
    payload = response.json()
    job_id = payload.get("jobId")
    if not job_id:
        raise RuntimeError(f"No jobId returned by UniProt API: {payload}")
    return job_id


def wait_for_job(job_id: str) -> None:
    """Poll until the job is FINISHED (or raise on failure/timeout)."""
    url = f"{UNIPROT_API}/idmapping/status/{job_id}"
    for _ in range(MAX_RETRIES):
        response = request_with_retry("GET", url)
        data = response.json()
        status = data.get("jobStatus")

        if status == "FAILED":
            raise RuntimeError(f"Job {job_id} FAILED: {data}")

        # UniProt may omit 'jobStatus' when job is complete and include results counters.
        if status == "FINISHED" or (status is None and (data.get("results") or data.get("failedIds") is not None)):
            return

        time.sleep(POLL_SECONDS)

    raise TimeoutError(f"Job {job_id} did not finish within timeout.")


def parse_next_link(link_header: str) -> str | None:
    if not link_header:
        return None
    for part in link_header.split(","):
        part = part.strip()
        if 'rel="next"' in part and "<" in part and ">" in part:
            return part[part.index("<") + 1: part.index(">")]
    return None


def fetch_results(job_id: str) -> list[dict[str, str]]:
    """Download all result pages for a finished job."""
    results: list[dict[str, str]] = []
    url = (
        f"{UNIPROT_API}/idmapping/uniprotkb/results/{job_id}"
        "?format=tsv&fields=accession&size=500"
    )

    while url:
        response = request_with_retry("GET", url)
        lines = response.text.strip().splitlines()

        if lines:
            for row in lines[1:]:
                parts = row.split("\t")
                if len(parts) >= 2 and parts[0] and parts[1]:
                    results.append(
                        {
                            "UniProtKB_AC": parts[1],
                            "Ensembl_Protein": parts[0],
                        }
                    )

        url = parse_next_link(response.headers.get("Link", ""))

    return results


def deduplicate_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Deduplicate by (UniProtKB_AC, Ensembl_Protein) pairs while preserving order."""
    seen: set[tuple[str, str]] = set()
    unique_rows: list[dict[str, str]] = []
    for row in rows:
        key = (row["UniProtKB_AC"], row["Ensembl_Protein"])
        if key in seen:
            continue
        seen.add(key)
        unique_rows.append(row)
    return unique_rows


def write_csv(output_file: Path, rows: list[dict[str, str]]) -> None:
    print(f"\nStep 3 - Writing CSV to {output_file} ...")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["UniProtKB_AC", "Ensembl_Protein"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"  Done! {len(rows):,} rows written to:\n  {output_file}")


def main() -> None:
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--input-file", type=Path, default=DEFAULT_INPUT_FILE, help="STRING/PPI link file containing ENSP ids")
    parser.add_argument("--output-file", type=Path, default=DEFAULT_OUTPUT_FILE, help="CSV output file for UniProt mappings")
    args = parser.parse_args()

    ensp_list = extract_unique_ensp_ids(args.input_file)

    print("\nStep 2 - Submitting ID mapping jobs to UniProt ...")
    all_results: list[dict[str, str]] = []
    total_batches = (len(ensp_list) + BATCH_SIZE - 1) // BATCH_SIZE

    for batch_num in range(total_batches):
        start = batch_num * BATCH_SIZE
        end = (batch_num + 1) * BATCH_SIZE
        batch = ensp_list[start:end]
        print(f"  Batch {batch_num + 1}/{total_batches} ({len(batch)} IDs) ... ", end="", flush=True)

        job_id = submit_batch(batch)
        wait_for_job(job_id)
        batch_results = fetch_results(job_id)
        all_results.extend(batch_results)
        print(f"-> {len(batch_results)} mappings found.")

    print(f"\n  Total mappings collected (before dedupe): {len(all_results):,}")
    unique_results = deduplicate_rows(all_results)
    print(f"  Total mappings after dedupe: {len(unique_results):,}")

    write_csv(args.output_file, unique_results)


if __name__ == "__main__":
    main()
