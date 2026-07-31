"""Learned resolver for the hedge pool: plain-arithmetic logistic evaluation.

Scope is the safety property. The model is consulted only for packets on
NO_APPROVAL_PATHS - cases whose required evidence could not be read and which
today all hedge to review. It never touches a case decided by rules, stamps,
or signed findings, its probabilities pass through the same fail-closed
``decide()`` (approval requires the 1.5x margin over denial mass), and the
emitted-fields guardrail still runs afterwards.

Features are evidence quality and document structure only - what was
readable, what is present, how damaged - never identities. Coefficients are
fitted out of fold by tools/fit_resolver.py and shipped as JSON; evaluation
here is a dot product and a softmax, adding no runtime dependencies.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from .constants import UNKNOWN
from .record import Record

_RESOLVER_PATH = Path(__file__).resolve().parent.parent / "policy" / "resolver.json"

FEATURES = (
    "fee_paid",
    "fee_waived",
    "fee_known",
    "registry_clear",
    "registry_read",
    "visa_known",
    "visa_dip1",
    "visa_med3",
    "visa_transit",
    "sponsor_known",
    "name_known",
    "species_known",
    "world_known",
    "date_known",
    "purpose_known",
    "flags_read",
    "has_scan",
    "injection",
    "missing_count",
    "page_count",
    "ocr_confidence",
    "waiver_seen",
)


def feature_vector(record: Record) -> list[float]:
    return [
        float(record.fee_status == "paid"),
        float(record.fee_status == "waived"),
        float(record.fee_status != UNKNOWN),
        float(record.registry_status == "CLEAR"),
        float(bool(record.registry_status)),
        float(record.visa_class != UNKNOWN),
        float(record.visa_class == "DIP-1"),
        float(record.visa_class == "MED-3"),
        float(record.visa_class == "TRANSIT-7"),
        float(record.sponsor_id != UNKNOWN),
        float(record.applicant_name != UNKNOWN),
        float(record.species_code != UNKNOWN),
        float(record.home_world != UNKNOWN),
        float(record.arrival_date != UNKNOWN),
        float(record.declared_purpose != UNKNOWN),
        float(record.risk_flags_known),
        float(record.has_scanned_pages),
        float(record.injection_detected),
        float(len(record.missing_fields)),
        float(record.pages),
        float(record.ocr_mean_confidence) / 100.0,
        float(record.has_hardship_waiver),
    ]


class Resolver:
    def __init__(self, path: str | Path = _RESOLVER_PATH):
        self.available = False
        try:
            with open(path) as handle:
                payload = json.load(handle)
        except (OSError, ValueError):
            return
        if payload.get("features") != list(FEATURES):
            # A stale coefficient file must fail closed, not misalign.
            return
        self.mu = payload["mu"]
        self.sigma = payload["sigma"]
        self.weights = payload["weights"]
        self.bias = payload["bias"]
        self.outcomes = payload["outcomes"]
        self.available = True

    def probs(self, record: Record) -> dict[str, float] | None:
        if not self.available:
            return None
        x = feature_vector(record)
        z = [(v - m) / s for v, m, s in zip(x, self.mu, self.sigma)]
        logits = [
            sum(zi * self.weights[i][k] for i, zi in enumerate(z)) + self.bias[k]
            for k in range(len(self.outcomes))
        ]
        peak = max(logits)
        exps = [math.exp(l - peak) for l in logits]
        total = sum(exps)
        return {o: e / total for o, e in zip(self.outcomes, exps)}
