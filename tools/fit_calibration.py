#!/usr/bin/env python3
"""Fit the path -> outcome-distribution calibration table from public labels.

Builds gold-field Records from data/train_labels.csv, names each case's
decision path, and records the empirical outcome distribution per path with
Laplace smoothing. Also reports what the policy alone would score, which is
the ceiling the extraction layer is chasing.

Usage:
  python tools/fit_calibration.py --labels ../mib-doc-challenge/data/train_labels.csv \
      --output policy/calibration.json
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.constants import PAYOFF, UNKNOWN  # noqa: E402
from mib.policy import (  # noqa: E402
    OUTCOMES,
    apply_corpus_context,
    corpus_reference_date,
    corpus_revoked_sponsors,
    decide,
    decision_path,
)
from mib.record import Record  # noqa: E402

LAPLACE = 1.0


def record_from_label_row(row: dict) -> Record:
    flags = row["risk_flags"].strip()
    return Record(
        case_id=row["case_id"].strip(),
        applicant_name=row["applicant_name"].strip() or UNKNOWN,
        species_code=row["species_code"].strip() or UNKNOWN,
        home_world=row["home_world"].strip() or UNKNOWN,
        visa_class=row["visa_class"].strip() or UNKNOWN,
        sponsor_id=row["sponsor_id"].strip() or UNKNOWN,
        arrival_date=row["arrival_date"].strip() or UNKNOWN,
        declared_purpose=row["declared_purpose"].strip() or UNKNOWN,
        fee_status=row["fee_status"].strip() or UNKNOWN,
        risk_flags=frozenset()
        if flags in ("", "none")
        else frozenset(flags.split("|")),
        # Gold labels state fee_status outright; "unknown" there is the printed
        # value, not an extraction failure.
        fee_explicit_unknown=row["fee_status"].strip() == "unknown",
        risk_flags_known=True,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--report", default=None, help="Optional JSON report path.")
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))

    records = [record_from_label_row(row) for row in rows]

    reference = corpus_reference_date([r.arrival_date for r in records])
    revoked = corpus_revoked_sponsors([r.sponsor_id for r in records])
    for record in records:
        apply_corpus_context(record, reference, revoked)

    path_outcomes: dict[str, Counter] = defaultdict(Counter)
    for record, row in zip(records, rows):
        path_outcomes[decision_path(record)][row["adjudication"].strip()] += 1

    paths = {}
    for path, counts in sorted(path_outcomes.items()):
        total = sum(counts.values()) + LAPLACE * len(OUTCOMES)
        paths[path] = {
            outcome: (counts.get(outcome, 0) + LAPLACE) / total for outcome in OUTCOMES
        }

    # Paths gold-field fitting cannot produce (they encode packet-level
    # evidence states) get provisional priors, replaced by an out-of-fold
    # refit once extraction runs on real PDFs. All lean conservative: an
    # unnamed or evidence-poor state must hedge to review, never deny or
    # approve on a global class prior.
    priors = {
        "manual_approved": {"APPROVED": 0.92, "DENIED": 0.03, "NEEDS_REVIEW": 0.05},
        "manual_denied": {"APPROVED": 0.02, "DENIED": 0.93, "NEEDS_REVIEW": 0.05},
        "manual_needs_review": {"APPROVED": 0.05, "DENIED": 0.05, "NEEDS_REVIEW": 0.90},
        "sponsor_revoked_override": {"APPROVED": 0.70, "DENIED": 0.15, "NEEDS_REVIEW": 0.15},
        "stale_dip_exempt": {"APPROVED": 0.75, "DENIED": 0.10, "NEEDS_REVIEW": 0.15},
        "clean_injection_seen": {"APPROVED": 0.85, "DENIED": 0.03, "NEEDS_REVIEW": 0.12},
        "risk_page_unreadable": {"APPROVED": 0.25, "DENIED": 0.20, "NEEDS_REVIEW": 0.55},
        "risk_page_unreadable_med": {"APPROVED": 0.10, "DENIED": 0.55, "NEEDS_REVIEW": 0.35},
        "risk_page_absent": {"APPROVED": 0.50, "DENIED": 0.10, "NEEDS_REVIEW": 0.40},
        "arrival_date_untrusted": {"APPROVED": 0.10, "DENIED": 0.15, "NEEDS_REVIEW": 0.75},
        "arrival_date_unknown": {"APPROVED": 0.10, "DENIED": 0.15, "NEEDS_REVIEW": 0.75},
        "visa_unknown": {"APPROVED": 0.15, "DENIED": 0.25, "NEEDS_REVIEW": 0.60},
        "sponsor_unknown": {"APPROVED": 0.15, "DENIED": 0.25, "NEEDS_REVIEW": 0.60},
        "fee_unknown": {"APPROVED": 0.15, "DENIED": 0.20, "NEEDS_REVIEW": 0.65},
        "staleness_indeterminate": {"APPROVED": 0.20, "DENIED": 0.20, "NEEDS_REVIEW": 0.60},
    }
    for path, distribution in priors.items():
        paths.setdefault(path, distribution)

    fallback = {"APPROVED": 0.20, "DENIED": 0.20, "NEEDS_REVIEW": 0.60}

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(
            {
                "fitted_from": Path(args.labels).name,
                "corpus_reference_date": reference,
                "corpus_revoked_outliers": sorted(revoked),
                "laplace": LAPLACE,
                "paths": paths,
                "fallback": fallback,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    # --- Oracle report: what does policy alone score on gold fields? --------
    raw = correct = cfa = 0.0
    confusion = Counter()
    brier_total = 0.0
    per_path: dict[str, Counter] = defaultdict(Counter)
    for record, row in zip(records, rows):
        truth = row["adjudication"].strip()
        path = decision_path(record)
        pred, confidence = decide(paths[path])
        raw += PAYOFF[pred][truth]
        confusion[(truth, pred)] += 1
        per_path[path][("hit" if pred == truth else "miss")] += 1
        if pred == truth:
            correct += 1
        if truth == "DENIED" and pred == "APPROVED":
            cfa += 1
        brier_total += (confidence - (1.0 if pred == truth else 0.0)) ** 2

    n = len(rows)
    mean_brier = brier_total / n
    classification = 80.0 * raw / (8.0 * n)
    calibration = 20.0 * max(0.0, 1.0 - 2.0 * mean_brier)
    print(f"Oracle-field policy on {n} train cases:")
    print(f"  accuracy:            {correct / n:.3f}")
    print(f"  classification:      {classification:.2f} / 80")
    print(f"  calibration:         {calibration:.2f} / 20  (mean Brier {mean_brier:.4f})")
    print(f"  catastrophic FA:     {cfa:.0f}")
    print(f"  corpus reference:    {reference}")
    print(f"  corpus revoked ids:  {sorted(revoked)}")
    print("  per-path outcomes:")
    for path, counts in sorted(path_outcomes.items(), key=lambda kv: -sum(kv[1].values())):
        pred, confidence = decide(paths[path])
        hits = per_path[path]["hit"]
        size = sum(counts.values())
        print(
            f"    {path:28s} n={size:4d} -> {pred:12s} conf={confidence:.2f} "
            f"acc={hits / size:.2f}  truth={dict(counts)}"
        )

    if args.report:
        with open(args.report, "w") as f:
            json.dump(
                {
                    "accuracy": correct / n,
                    "classification_score": classification,
                    "calibration_score": calibration,
                    "mean_brier": mean_brier,
                    "catastrophic_false_approvals": cfa,
                    "confusion": {f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
                },
                f,
                indent=2,
                sort_keys=True,
            )
            f.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
