#!/usr/bin/env python3
"""Extract every packet once and cache the Records as JSONL.

Extraction dominates runtime (~1.2s/PDF); policy and calibration work do not
touch the PDFs at all. Caching the extracted Records turns a 20-minute
full-corpus experiment into a sub-second one, which is what makes out-of-fold
fitting and policy iteration practical.

Usage:
  python tools/build_cache.py --pdf-dir ../mib-doc-challenge/data/train \
      --output cache/train.jsonl
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from dataclasses import asdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.extract import extract_record  # noqa: E402
from mib.pdfio import load_packet  # noqa: E402
from mib.record import Record  # noqa: E402


def _extract(path: str) -> dict:
    record = extract_record(load_packet(path))
    payload = asdict(record)
    payload["risk_flags"] = sorted(record.risk_flags)
    return payload


def record_from_dict(payload: dict) -> Record:
    data = dict(payload)
    data["risk_flags"] = frozenset(data.get("risk_flags") or ())
    data["missing_fields"] = tuple(data.get("missing_fields") or ())
    return Record(**data)


def load_cache(path: str | Path) -> list[Record]:
    records = []
    with open(path) as handle:
        for line in handle:
            line = line.strip()
            if line:
                records.append(record_from_dict(json.loads(line)))
    return records


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pdfs = sorted(str(p) for p in Path(args.pdf_dir).glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    start = time.time()
    done = 0
    with open(output, "w") as handle, mp.get_context("spawn").Pool(args.workers) as pool:
        for payload in pool.imap(_extract, pdfs, chunksize=4):
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
            done += 1
            if done % 100 == 0:
                rate = (time.time() - start) / done
                remaining = rate * (len(pdfs) - done)
                print(f"  {done}/{len(pdfs)}  {rate:.2f}s/PDF  eta {remaining / 60:.1f}m",
                      flush=True)

    elapsed = time.time() - start
    print(f"cached {done} records to {output} in {elapsed / 60:.1f}m "
          f"({elapsed / max(1, done):.2f}s/PDF)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
