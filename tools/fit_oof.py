#!/usr/bin/env python3
"""Fit the path -> outcome table on EXTRACTED records, out of fold.

The previous fitter used gold label fields. That is a train/serve mismatch:
with perfect fields the `clean` path is 93% APPROVED, but the pipeline reaches
that path from noisy OCR, where it is not. Calibrating on the distribution the
policy actually sees is what makes the expected-value argmax correct and the
reported confidence honest.

Folds matter for the same reason. A path's outcome distribution fitted on the
packets it is then scored against is optimistic, and the reported out-of-fold
total is the number that should be believed - the in-sample one is what the
leaderboard's overfitters were quoting.

Usage:
  python tools/fit_oof.py --cache cache/train.jsonl \
      --labels ../mib-doc-challenge/data/train_labels.csv \
      --output policy/calibration.json --folds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.constants import PAYOFF  # noqa: E402
from mib.policy import (  # noqa: E402
    NO_APPROVAL_PATHS,
    OUTCOMES,
    apply_corpus_context,
    corpus_reference_date,
    corpus_revoked_sponsors,
    decide,
    decision_path,
    emitted_guardrail,
)
from mib.output import apply_priors, load_priors  # noqa: E402
from tools.build_cache import load_cache  # noqa: E402

FIELDS = ("applicant_name", "species_code", "home_world", "visa_class",
          "sponsor_id", "arrival_date", "declared_purpose", "risk_flags",
          "fee_status")
WEIGHTS = {"applicant_name": 5, "species_code": 6, "home_world": 5,
           "visa_class": 5, "sponsor_id": 5, "arrival_date": 4,
           "declared_purpose": 3, "risk_flags": 8, "fee_status": 4}
LAPLACE = 1.0

# A path seen this rarely cannot support its own distribution; it falls back to
# the global prior rather than overfitting a handful of packets.
MIN_PATH_SUPPORT = 8


def fit_table(paths_and_truth) -> tuple[dict, dict]:
    counts: dict[str, Counter] = defaultdict(Counter)
    overall = Counter()
    for path, truth in paths_and_truth:
        counts[path][truth] += 1
        overall[truth] += 1

    total = sum(overall.values()) + LAPLACE * len(OUTCOMES)
    fallback = {o: (overall.get(o, 0) + LAPLACE) / total for o in OUTCOMES}

    table = {}
    for path, outcome_counts in counts.items():
        support = sum(outcome_counts.values())
        if support < MIN_PATH_SUPPORT:
            continue
        denominator = support + LAPLACE * len(OUTCOMES)
        table[path] = {
            o: (outcome_counts.get(o, 0) + LAPLACE) / denominator for o in OUTCOMES
        }
    return table, fallback


def evaluate(records, labels, table, fallback, priors=None):
    priors = priors or {}
    hits = Counter()
    classification_raw = 0.0
    correct = catastrophic = 0
    brier = 0.0
    confusion = Counter()
    scored = 0

    for record in records:
        truth = labels.get(record.case_id)
        if truth is None:
            continue
        scored += 1
        path = decision_path(record)
        adjudication, confidence = decide(
            table.get(path, fallback), allow_approval=path not in NO_APPROVAL_PATHS
        )

        row = {f: ("|".join(sorted(record.flag_set())) or "none")
               if f == "risk_flags" else str(getattr(record, f)) for f in FIELDS}
        row["adjudication"] = adjudication
        demotion = emitted_guardrail(row, record)
        if demotion:
            adjudication, confidence = demotion

        emitted = apply_priors(row, priors)
        for field in FIELDS:
            if emitted[field].lower() == (truth[field].strip().lower() or "none"):
                hits[field] += 1

        gold = truth["adjudication"].strip()
        classification_raw += PAYOFF[adjudication][gold]
        correct += adjudication == gold
        catastrophic += gold == "DENIED" and adjudication == "APPROVED"
        brier += (confidence - (1.0 if adjudication == gold else 0.0)) ** 2
        confusion[(gold, adjudication)] += 1

    extraction = 50.0 * sum(WEIGHTS[f] * hits[f] for f in FIELDS) / (
        sum(WEIGHTS.values()) * scored
    )
    classification = 80.0 * classification_raw / (8.0 * scored)
    mean_brier = brier / scored
    calibration = 20.0 * max(0.0, 1.0 - 2.0 * mean_brier)
    return {
        "n": scored,
        "extraction": extraction,
        "classification": classification,
        "calibration": calibration,
        "total": extraction + classification + calibration,
        "accuracy": correct / scored,
        "brier": mean_brier,
        "catastrophic": catastrophic,
        "hits": hits,
        "confusion": confusion,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        labels = {row["case_id"]: row for row in csv.DictReader(f)}

    priors = load_priors()
    records = [r for r in load_cache(args.cache) if r.case_id in labels]
    reference = corpus_reference_date(
        [r.arrival_date for r in records if r.arrival() is not None]
    )
    revoked = corpus_revoked_sponsors([r.sponsor_id for r in records])
    for record in records:
        apply_corpus_context(record, reference, revoked)

    # Out-of-fold: each record is scored by a table fitted without it.
    folds = args.folds
    oof_paths = []
    for fold in range(folds):
        train = [r for i, r in enumerate(records) if i % folds != fold]
        held = [r for i, r in enumerate(records) if i % folds == fold]
        table, fallback = fit_table(
            (decision_path(r), labels[r.case_id]["adjudication"].strip()) for r in train
        )
        oof_paths.append((held, table, fallback))

    aggregate = Counter()
    oof_total = 0.0
    for held, table, fallback in oof_paths:
        result = evaluate(held, labels, table, fallback, priors)
        oof_total += result["total"] * result["n"]
        aggregate["n"] += result["n"]
        aggregate["catastrophic"] += result["catastrophic"]
    oof_score = oof_total / aggregate["n"]

    # Ship a table fitted on everything; report the out-of-fold estimate.
    table, fallback = fit_table(
        (decision_path(r), labels[r.case_id]["adjudication"].strip()) for r in records
    )
    in_sample = evaluate(records, labels, table, fallback, priors)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(
            {
                "fitted_from": Path(args.cache).name,
                "fitted_on": "extracted_records",
                "corpus_reference_date": reference,
                "corpus_revoked_outliers": sorted(revoked),
                "folds": folds,
                "min_path_support": MIN_PATH_SUPPORT,
                "oof_total_estimate": oof_score,
                "paths": table,
                "fallback": fallback,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    print(f"in-sample : {in_sample['total']:6.2f}/150  "
          f"(ext {in_sample['extraction']:.2f} | cls {in_sample['classification']:.2f} | "
          f"cal {in_sample['calibration']:.2f} | acc {in_sample['accuracy']:.3f} | "
          f"CFA {in_sample['catastrophic']})")
    print(f"out-of-fold: {oof_score:6.2f}/150   <- the number to believe "
          f"(CFA {aggregate['catastrophic']})")
    print("\nconfusion (in-sample, errors only):")
    for (gold, predicted), count in sorted(in_sample["confusion"].items()):
        if gold != predicted:
            print(f"  {gold:12s} -> {predicted:12s} {count:4d}")
    print("\npaths fitted:")
    counts = Counter(decision_path(r) for r in records)
    for path, count in counts.most_common(24):
        probs = table.get(path)
        if probs:
            choice, confidence = decide(
                probs, allow_approval=path not in NO_APPROVAL_PATHS
            )
            print(f"  {path:26s} n={count:4d} -> {choice:12s} conf={confidence:.2f}")
        else:
            print(f"  {path:26s} n={count:4d} -> (fallback, below support)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
