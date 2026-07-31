#!/usr/bin/env python3
"""Fit a learned resolver for the hedge pool, exported as plain coefficients.

The pipeline routes ~380 packets to review because a required field could not
be read. Path-level outcome distributions cannot separate them further, and a
boolean gate search found nothing that survives its own multiple-testing. But
both top public entries resolve exactly this pool with a small learned model
scoped so it can never touch a decided case - and the features that might
separate a recoverable approval from a hidden denial (which documents are
present, what was readable, how the packet is damaged) are already on the
Record.

This fits an L2 multinomial logistic on evidence-quality features, out of
fold, and exports raw coefficients to JSON. The runtime evaluates it with
arithmetic - no sklearn, no new dependencies, nothing pickled.

Safety is structural, mirroring the rest of the policy:
- consulted ONLY on hedge-pool paths; guarded and decided cases are untouched
- its probabilities pass through the same fail-closed decide(): approval
  needs the 1.5x margin over denial mass
- the emitted-fields guardrail still runs afterwards and can only demote
- identity features are excluded by construction (no ids, no names)

Usage:
  python tools/fit_resolver.py --cache cache/train_v13.jsonl \
      --labels ../mib-doc-challenge/data/train_labels.csv \
      --output policy/resolver.json --folds 5
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.constants import PAYOFF, UNKNOWN  # noqa: E402
from mib.policy import (  # noqa: E402
    NO_APPROVAL_PATHS,
    OUTCOMES,
    apply_corpus_context,
    corpus_reference_date,
    corpus_revoked_sponsors,
    decide,
    decision_path,
)
from mib.resolver import FEATURES, feature_vector  # noqa: E402
from tools.build_cache import load_cache  # noqa: E402


def fit_logistic(X, y, l2=1.0, iters=400, lr=0.5):
    """Multinomial logistic via gradient descent - no sklearn dependency."""
    n, d = X.shape
    k = 3
    W = np.zeros((d, k))
    b = np.zeros(k)
    Y = np.zeros((n, k))
    for i, label in enumerate(y):
        Y[i, label] = 1.0
    for _ in range(iters):
        logits = X @ W + b
        logits -= logits.max(axis=1, keepdims=True)
        P = np.exp(logits)
        P /= P.sum(axis=1, keepdims=True)
        G = (P - Y) / n
        W -= lr * (X.T @ G + l2 * W / n)
        b -= lr * G.sum(axis=0)
    return W, b


def predict_probs(W, b, X):
    logits = X @ W + b
    logits -= logits.max(axis=1, keepdims=True)
    P = np.exp(logits)
    return P / P.sum(axis=1, keepdims=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--folds", type=int, default=5)
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

    pool = [(r, labels[r.case_id]["adjudication"].strip())
            for r in records if decision_path(r) in NO_APPROVAL_PATHS]
    print(f"hedge pool: {len(pool)}  truth mix: {Counter(t for _, t in pool)}")

    outcome_index = {o: i for i, o in enumerate(OUTCOMES)}
    X = np.array([feature_vector(r) for r, _ in pool])
    y = np.array([outcome_index[t] for _, t in pool])

    # Standardize; store moments for the runtime.
    mu, sigma = X.mean(axis=0), X.std(axis=0)
    sigma[sigma < 1e-9] = 1.0
    Xs = (X - mu) / sigma

    # Out-of-fold evaluation with the SAME fail-closed decide() as runtime.
    folds = np.arange(len(pool)) % args.folds
    oof_raw = path_raw = 0.0
    oof_cfa = 0
    changed = Counter()
    for fold in range(args.folds):
        train, held = folds != fold, folds == fold
        W, b = fit_logistic(Xs[train], y[train])
        P = predict_probs(W, b, Xs[held])
        for probs, (record, truth) in zip(P, [p for p, m in zip(pool, held) if m]):
            dist = {o: float(probs[outcome_index[o]]) for o in OUTCOMES}
            # Deny-or-review only: the model cannot tell an approvable packet
            # from one where review is the intended answer (52 of its 167
            # approvals were truth-review, 27 truth-denied). It CAN smell a
            # hidden denial. A deny-only resolver is structurally incapable
            # of a catastrophic false approval.
            pred, _ = decide(dist, allow_approval=False)
            oof_raw += PAYOFF[pred][truth]
            oof_cfa += truth == "DENIED" and pred == "APPROVED"
            base = "NEEDS_REVIEW"  # what the pool paths emit today, near enough
            path_raw += PAYOFF[base][truth]
            if pred != base:
                changed[(pred, truth)] += 1

    n = len(pool)
    print(f"\npool classification, out of fold:")
    print(f"  hedging (today):   {path_raw / n:.3f} raw/case")
    print(f"  learned resolver:  {oof_raw / n:.3f} raw/case")
    print(f"  delta on 80-pt scale (whole corpus): "
          f"{80 * (oof_raw - path_raw) / (8 * len(records)):+.2f}")
    print(f"  out-of-fold CFA: {oof_cfa}")
    print(f"  decisions changed vs hedge: {dict(changed)}")

    W, b = fit_logistic(Xs, y)
    payload = {
        "features": list(FEATURES),
        "mu": mu.tolist(),
        "sigma": sigma.tolist(),
        "weights": W.tolist(),
        "bias": b.tolist(),
        "outcomes": list(OUTCOMES),
        "fitted_from": Path(args.cache).name,
        "oof_cfa": int(oof_cfa),
        "oof_delta_80pt": float(80 * (oof_raw - path_raw) / (8 * len(records))),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"\nwrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
