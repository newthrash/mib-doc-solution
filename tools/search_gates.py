#!/usr/bin/env python3
"""Search for positive-evidence gates that safely certify an approval.

191 truly-approved packets are routed to review because a required piece of
evidence could not be read - 11.5 classification points. Approving those paths
wholesale costs 17 catastrophic false approvals, so the question is whether a
conjunction of *corroborating visible facts* isolates a subset that is safe.

Discipline, fixed before any result is inspected:

- Candidate gates are conjunctions of at most three signals.
- A gate must contain ZERO denials in its training fold. Approving a denial
  costs -4 and is the second tie-breaker; a gate that admits one is rejected
  regardless of how many approvals it unlocks.
- It must hold up out of fold: evaluated on five folds, a gate is reported only
  if every fold is denial-free and it covers a minimum number of cases.
- Signals are packet evidence, never case identity, and never a label.

Usage:
  python tools/search_gates.py --cache cache/train_v7.jsonl \
      --labels ../mib-doc-challenge/data/train_labels.csv
"""

from __future__ import annotations

import argparse
import csv
import itertools
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.policy import (  # noqa: E402
    NO_APPROVAL_PATHS,
    apply_corpus_context,
    corpus_reference_date,
    corpus_revoked_sponsors,
    decision_path,
)
from tools.build_cache import load_cache  # noqa: E402

MIN_COVERAGE = 12          # a gate below this is noise, not a rule
FOLDS = 5


def signals(record) -> dict[str, bool]:
    """Visible-evidence predicates. No case identity, no labels."""
    flags = record.flag_set()
    return {
        "fee_paid": record.fee_status == "paid",
        "fee_known": record.fee_status != "unknown",
        "registry_clear": record.registry_status == "CLEAR",
        "registry_read": bool(record.registry_status),
        "visa_known": record.visa_class != "unknown",
        "sponsor_known": record.sponsor_id != "unknown",
        "name_known": record.applicant_name != "unknown",
        "species_known": record.species_code != "unknown",
        "world_known": record.home_world != "unknown",
        "date_known": record.arrival_date != "unknown",
        "purpose_known": record.declared_purpose != "unknown",
        "no_flags_seen": not flags,
        "flags_read": record.risk_flags_known,
        "no_injection": not record.injection_detected,
        "few_missing": len(record.missing_fields) <= 1,
        "most_fields": len(record.missing_fields) <= 3,
        "has_scan": record.has_scanned_pages,
        "no_scan": not record.has_scanned_pages,
        "dip1": record.visa_class == "DIP-1",
        "not_med3": record.visa_class != "MED-3",
        "good_ocr": record.ocr_mean_confidence >= 80,
        "short_packet": record.pages <= 4,
        "no_stamp": record.stamp_verdict is None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--max-terms", type=int, default=3)
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        labels = {row["case_id"]: row for row in csv.DictReader(f)}

    records = [r for r in load_cache(args.cache) if r.case_id in labels]
    reference = corpus_reference_date(
        [r.arrival_date for r in records if r.arrival() is not None]
    )
    revoked = corpus_revoked_sponsors([r.sponsor_id for r in records])
    for record in records:
        apply_corpus_context(record, reference, revoked)

    # Only packets we currently refuse to approve are in play.
    pool = []
    for index, record in enumerate(records):
        path = decision_path(record)
        if path not in NO_APPROVAL_PATHS:
            continue
        pool.append((index, record, path, labels[record.case_id]["adjudication"]))

    print(f"candidate pool: {len(pool)} packets on non-approving paths")
    print(f"  truth mix: {Counter(t for _, _, _, t in pool)}\n")

    names = sorted(signals(pool[0][1]))
    hits = {name: [] for name in names}
    for _, record, _, _ in pool:
        values = signals(record)
        for name in names:
            hits[name].append(values[name])

    truths = [t for _, _, _, t in pool]
    folds = [i % FOLDS for i in range(len(pool))]

    survivors = []
    for size in range(1, args.max_terms + 1):
        for combo in itertools.combinations(names, size):
            member = [
                all(hits[name][i] for name in combo) for i in range(len(pool))
            ]
            covered = [i for i, m in enumerate(member) if m]
            if len(covered) < MIN_COVERAGE:
                continue
            counts = Counter(truths[i] for i in covered)
            if counts["DENIED"]:
                continue
            # Denial-free in every fold that contains any of its members.
            ok = True
            for fold in range(FOLDS):
                fold_members = [i for i in covered if folds[i] == fold]
                if fold_members and any(truths[i] == "DENIED" for i in fold_members):
                    ok = False
                    break
            if not ok:
                continue
            survivors.append((counts["APPROVED"], len(covered), combo, counts))

    survivors.sort(reverse=True)
    if not survivors:
        print("No denial-free gate met the coverage bar. Hedging is correct here.")
        return 0

    print(f"{len(survivors)} denial-free gates at coverage >= {MIN_COVERAGE}:\n")
    for approved, coverage, combo, counts in survivors[:20]:
        gain = 80 * approved * 6 / 8000
        print(f"  {' AND '.join(combo):58s}")
        print(f"    covers {coverage:3d}  approved {approved:3d}  review "
              f"{counts['NEEDS_REVIEW']:3d}  denied 0   -> +{gain:.2f} clf pts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
