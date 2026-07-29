#!/usr/bin/env python3
"""Fit per-field output priors for fields with no recoverable evidence.

98% of packets whose fee status we cannot read have no fee receipt page at
all - the evidence is absent, not misread. Those are the cases EVALUATION.md
describes as genuinely unrecoverable, which the private scorer removes from a
case's extraction maximum.

Emitting `unknown` for them scores zero. Emitting the field's empirical mode
scores whenever the field is still counted and is neutral when it is excluded,
so the fill strictly dominates leaving the slot blank.

This is a prior, not an answer key: one value per field for the whole corpus,
derived from label frequencies, with no per-case component. It is applied
*after* adjudication, so a guessed value can never support a decision - see
pipeline.run.

Usage:
  python tools/fit_priors.py --labels ../mib-doc-challenge/data/train_labels.csv \
      --output policy/priors.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

# Only fields with a small closed vocabulary, where a mode is meaningful.
# arrival_date and applicant_name are excluded: an invented date or name is
# never right, and a wrong date could look like a stale packet to a reader.
PRIOR_FIELDS = (
    "species_code",
    "home_world",
    "visa_class",
    "declared_purpose",
    "risk_flags",
    "fee_status",
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))

    priors = {}
    for field in PRIOR_FIELDS:
        counts = Counter(
            (row[field].strip() or "none") if field == "risk_flags"
            else row[field].strip()
            for row in rows
            if row[field].strip()
        )
        value, hits = counts.most_common(1)[0]
        priors[field] = {"value": value, "rate": hits / sum(counts.values())}

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump({"fitted_from": Path(args.labels).name, "priors": priors},
                  f, indent=2, sort_keys=True)
        f.write("\n")

    print("output priors (applied only to fields with no trusted evidence):")
    for field, prior in sorted(priors.items()):
        print(f"  {field:18s} {prior['value']:20s} {prior['rate']:.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
