#!/usr/bin/env python3
"""Derive the applicant-name grammar from public labels.

Names are not free text: the corpus uses 144 distinct tokens, shared between
first and last position, and they factor cleanly as 12 stems x 12 suffixes.
That makes applicant_name a closed vocabulary like every other field, so a
damaged reading can be snapped instead of emitted wrong.

Storing the factors rather than the 144 literals is the point: a private set
built by the same generator may pair stems and suffixes we never saw, and the
grammar covers those combinations while a token list would not.

Usage:
  python tools/fit_names.py --labels ../mib-doc-challenge/data/train_labels.csv \
      --output policy/names.json
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path


def factor(tokens: set[str]) -> tuple[list[str], list[str]]:
    """Split tokens into the stem and suffix sets that regenerate them."""
    best = (None, None, 0)
    for cut in range(2, 6):
        stems = Counter(t[:cut].lower() for t in tokens if len(t) > cut)
        suffixes = Counter(t[cut:].lower() for t in tokens if len(t) > cut)
        covered = sum(
            1 for t in tokens
            if len(t) > cut and stems[t[:cut].lower()] > 1 and suffixes[t[cut:].lower()] > 1
        )
        if covered > best[2]:
            best = (stems, suffixes, covered)
    stems, suffixes, _ = best
    return (
        sorted(s for s, n in stems.items() if n > 1),
        sorted(s for s, n in suffixes.items() if n > 1),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        rows = list(csv.DictReader(f))

    tokens: set[str] = set()
    for row in rows:
        tokens.update(row["applicant_name"].split())

    stems, suffixes = factor(tokens)
    generated = sorted({s + x for s in stems for x in suffixes})

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(
            {
                "fitted_from": Path(args.labels).name,
                "observed_tokens": sorted(tokens),
                "stems": stems,
                "suffixes": suffixes,
                "generated": generated,
            },
            f, indent=2, sort_keys=True,
        )
        f.write("\n")

    covered = sum(1 for t in tokens if t.lower() in set(generated))
    print(f"observed tokens : {len(tokens)}")
    print(f"stems x suffixes: {len(stems)} x {len(suffixes)} = {len(generated)}")
    print(f"grammar covers  : {covered}/{len(tokens)} observed tokens "
          f"({covered / len(tokens):.0%})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
