#!/usr/bin/env python3
"""Run the pipeline on labeled packets and report per-field extraction quality.

This is the inner development loop. It reports the same section scores the
official evaluator computes, plus the per-field breakdown the evaluator omits,
so a change can be attributed to the field it actually moved.

Usage:
  python tools/score_train.py --pdf-dir ../mib-doc-challenge/data/train \
      --labels ../mib-doc-challenge/data/train_labels.csv --limit 200
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
import time
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.constants import PAYOFF  # noqa: E402
from mib.extract import extract_record  # noqa: E402
from mib.pdfio import load_packet  # noqa: E402
from mib.policy import (  # noqa: E402
    Calibration,
    apply_corpus_context,
    corpus_reference_date,
    corpus_revoked_sponsors,
    emitted_guardrail,
)

FIELDS = ("applicant_name", "species_code", "home_world", "visa_class",
          "sponsor_id", "arrival_date", "declared_purpose", "risk_flags",
          "fee_status")
WEIGHTS = {"applicant_name": 5, "species_code": 6, "home_world": 5,
           "visa_class": 5, "sponsor_id": 5, "arrival_date": 4,
           "declared_purpose": 3, "risk_flags": 8, "fee_status": 4}


def _extract(path: str):
    return extract_record(load_packet(path))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        labels = {row["case_id"]: row for row in csv.DictReader(f)}

    pdfs = sorted(str(p) for p in Path(args.pdf_dir).glob("*.pdf"))
    if args.limit:
        pdfs = pdfs[: args.limit]

    start = time.time()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        records = pool.map(_extract, pdfs, chunksize=4)
    elapsed = time.time() - start

    reference = corpus_reference_date(
        [r.arrival_date for r in records if r.arrival() is not None]
    )
    revoked = corpus_revoked_sponsors([r.sponsor_id for r in records])
    calibration = Calibration()

    hits = Counter()
    classification_raw = 0.0
    correct = catastrophic = 0
    brier = 0.0
    paths = Counter()
    confusion = Counter()

    for record in records:
        truth = labels.get(record.case_id)
        if truth is None:
            continue
        apply_corpus_context(record, reference, revoked)
        adjudication, confidence, path = calibration.adjudicate(record)

        row = {f: ("|".join(sorted(record.flag_set())) or "none")
               if f == "risk_flags" else str(getattr(record, f)) for f in FIELDS}
        row["adjudication"] = adjudication
        demotion = emitted_guardrail(row, record)
        if demotion:
            adjudication, confidence = demotion

        for field in FIELDS:
            expected = truth[field].strip().lower() or "none"
            if row[field].lower() == expected:
                hits[field] += 1

        gold = truth["adjudication"].strip()
        classification_raw += PAYOFF[adjudication][gold]
        correct += adjudication == gold
        catastrophic += gold == "DENIED" and adjudication == "APPROVED"
        brier += (confidence - (1.0 if adjudication == gold else 0.0)) ** 2
        paths[path] += 1
        confusion[(gold, adjudication)] += 1

    n = len(records)
    extraction_raw = sum(WEIGHTS[f] * hits[f] for f in FIELDS)
    extraction_max = sum(WEIGHTS.values()) * n
    extraction = 50.0 * extraction_raw / extraction_max
    classification = 80.0 * classification_raw / (8.0 * n)
    mean_brier = brier / n
    calibration_score = 20.0 * max(0.0, 1.0 - 2.0 * mean_brier)

    print(f"{n} packets | {elapsed:.0f}s wall on {args.workers} workers "
          f"= {elapsed / n:.2f}s/PDF\n")
    for field in FIELDS:
        print(f"  {field:20s} {100 * hits[field] / n:5.1f}%")
    print(f"\n  extraction     {extraction:6.2f} / 50")
    print(f"  classification {classification:6.2f} / 80   (acc {correct / n:.3f})")
    print(f"  calibration    {calibration_score:6.2f} / 20   (Brier {mean_brier:.4f})")
    print(f"  TOTAL          {extraction + classification + calibration_score:6.2f} / 150")
    print(f"  catastrophic false approvals: {catastrophic}")
    print("\n  confusion (truth -> predicted):")
    for (gold, predicted), count in sorted(confusion.items()):
        if gold != predicted:
            print(f"    {gold:12s} -> {predicted:12s} {count:4d}")
    print("\n  top decision paths:")
    for path, count in paths.most_common(10):
        print(f"    {path:28s} {count:4d}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
