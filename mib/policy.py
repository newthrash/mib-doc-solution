"""Adjudication policy: evidence -> named decision path -> EV-optimal decision.

Three deliberately separated stages:

1. ``decision_path(record)`` maps trusted evidence to a named policy state.
   Pure function of the Record; no probabilities, no labels.
2. A calibration table maps each path to an empirical outcome distribution
   fitted on the public training labels (tools/fit_calibration.py).
3. ``decide(probs)`` picks the expected-value argmax under the evaluator's
   payoff matrix and reports P(chosen is correct) as confidence, which is
   exactly the quantity the Brier calibration term scores.

Corpus-level context (staleness reference, revoked-sponsor outliers) is
computed from the corpus being scored, not hardcoded from training data, so
the policy transfers to a private test set whose constants differ.

Design influences, credited in ATTRIBUTION.md: the path->distribution->EV
structure and corpus-relative constants follow zubalr's public MIT solution;
the one-way emitted-fields guardrail follows tylergibbs1's.
"""

from __future__ import annotations

import datetime as dt
import json
from collections import Counter
from pathlib import Path

from .constants import (
    DISQUALIFYING_FLAGS,
    EMBARGOED_HOME_WORLDS,
    PAYOFF,
    REVIEW_FLAGS,
    REVOKED_SPONSORS,
    STALE_DAYS,
    UNKNOWN,
)
from .record import Record

DEFAULT_CALIBRATION = Path(__file__).resolve().parent.parent / "policy" / "calibration.json"

APPROVED, DENIED, NEEDS_REVIEW = "APPROVED", "DENIED", "NEEDS_REVIEW"
OUTCOMES = (APPROVED, DENIED, NEEDS_REVIEW)


# --- Corpus-level context (transfer-safe mined constants) -------------------

# A revoked sponsor recurs across many packets while genuine sponsors are
# near-unique. Flag ids above this multiple of the 99th-percentile sponsor
# frequency. On train the flagged ids sit at 6-40x p99 while the honest tail
# tops out near 1x, so the exact multiple is not load-bearing.
REVOKED_FREQUENCY_MULTIPLE = 4
MIN_CORPUS_FOR_FREQUENCY = 400

# Staleness needs a receipt date. Prefer one read from the packet; else fall
# back to a corpus-level reference: the 90th-percentile arrival date is a
# robust "current batch epoch" estimator that survives OCR outliers.
MIN_CORPUS_FOR_REFERENCE = 50


def corpus_revoked_sponsors(sponsor_ids: list[str]) -> frozenset[str]:
    counts = Counter(s for s in sponsor_ids if s and s != UNKNOWN)
    if len(counts) < MIN_CORPUS_FOR_FREQUENCY:
        return frozenset()
    frequencies = sorted(counts.values())
    p99 = frequencies[int(len(frequencies) * 0.99)]
    threshold = REVOKED_FREQUENCY_MULTIPLE * p99
    return frozenset(s for s, n in counts.items() if n > threshold)


def corpus_reference_date(arrival_dates: list[str]) -> str | None:
    parsed = []
    for value in arrival_dates:
        try:
            parsed.append(dt.date.fromisoformat(value))
        except (ValueError, TypeError):
            continue
    if len(parsed) < MIN_CORPUS_FOR_REFERENCE:
        return None
    parsed.sort()
    return parsed[int(len(parsed) * 0.90)].isoformat()


def apply_corpus_context(
    record: Record,
    reference_date: str | None,
    revoked_outliers: frozenset[str] = frozenset(),
) -> Record:
    if record.receipt_date is None and reference_date:
        record.receipt_date = reference_date
    if record.sponsor_id in revoked_outliers:
        record.sponsor_revoked_in_corpus = True
    return record


# --- Decision paths ---------------------------------------------------------

def _is_stale(record: Record) -> bool | None:
    """True/False, or None when staleness cannot be determined."""
    arrival, receipt = record.arrival(), record.receipt()
    if arrival is None or receipt is None:
        return None
    return (receipt - arrival).days > STALE_DAYS


def decision_path(record: Record) -> str:
    """Name the policy rule governing this case.

    Present (visible) evidence is evaluated before missing evidence, so gaps
    can never suppress a stronger visible signal.
    """
    flags = record.flag_set()
    visa_known = record.visa_class != UNKNOWN

    # Tier 0: the manual puts "visible MIB adjudicator stamp or signed manual
    # note" at the top of the precedence list. Both are read here; when the two
    # disagree the packet is contested and belongs in review rather than being
    # resolved by preferring one channel over the other.
    finding = record.manual_finding if record.manual_finding in OUTCOMES else None
    stamp = record.stamp_verdict if record.stamp_verdict in OUTCOMES else None
    if finding and stamp and finding != stamp:
        return "adjudicator_conflict"
    if finding:
        return f"manual_{finding.lower()}"
    if stamp:
        return f"stamp_{stamp.lower()}"
    if record.stamp_contested:
        # Stamps present but all cancelled, or disagreeing with each other.
        return "stamp_contested"

    # Tier 1: present disqualifying evidence.
    if record.visa_class == "TRANSIT-7":
        return "transit_7"
    if flags & DISQUALIFYING_FLAGS:
        return "disqualifying_flag"
    if record.fee_status == "unpaid" and not record.has_hardship_waiver:
        return "fee_unpaid"
    if (
        record.home_world in EMBARGOED_HOME_WORLDS
        and visa_known
        and record.visa_class != "DIP-1"
    ):
        return "embargoed_home_world"

    # Tier 2: present adverse evidence.
    if visa_known and record.visa_class != "DIP-1" and record.sponsor_id != UNKNOWN:
        if record.sponsor_id in REVOKED_SPONSORS or record.sponsor_revoked_in_corpus:
            return (
                "sponsor_revoked_override"
                if record.has_approval_override
                else "sponsor_revoked"
            )

    if flags & REVIEW_FLAGS:
        return "review_flags_multi" if len(flags & REVIEW_FLAGS) > 1 else "review_flags"

    if (
        record.fee_status == "waived"
        and visa_known
        and record.visa_class != "DIP-1"
        and not record.has_hardship_waiver
    ):
        return "fee_waived_unjustified"

    if record.fee_explicit_unknown:
        # The receipt *states* "unknown": present evidence with a documented
        # rule (44/44 NEEDS_REVIEW on train), unlike the unread sentinel below.
        return "fee_stated_unknown"

    stale = _is_stale(record)
    if stale is True:
        if record.visa_class == "DIP-1" and record.has_diplomatic_note:
            return "stale_dip_exempt"
        return "stale_arrival"

    # Tier 3: missing evidence.
    if record.arrival_date_untrusted:
        return "arrival_date_untrusted"
    if record.arrival_date == UNKNOWN:
        return "arrival_date_unknown"
    if not visa_known:
        return "visa_unknown"
    if record.visa_class != "DIP-1" and record.sponsor_id == UNKNOWN:
        return "sponsor_unknown"
    if not record.risk_flags_known:
        if record.has_scanned_pages:
            # MED-3 is where an unobserved biohazard_red hides; empirically the
            # one sub-split of the unreadable bucket that transfers.
            return (
                "risk_page_unreadable_med"
                if record.visa_class == "MED-3"
                else "risk_page_unreadable"
            )
        return "risk_page_absent"
    if record.fee_status == UNKNOWN:
        return "fee_unknown"
    if stale is None:
        return "staleness_indeterminate"

    # Tier 4: no adverse evidence.
    if record.injection_detected:
        return "clean_injection_seen"
    return "clean"


# --- Decisions --------------------------------------------------------------

# An approval must clear the denial mass by this factor, not merely win on
# expected value. Raw EV is optimal against a *known* distribution; ours is a
# sample estimate, and its errors are not symmetric. Approving a denial costs
# -4 and is both an explicit tie-breaker and a minimum-bar criterion in
# EVALUATION.md, while the review hedge still pays 2. So approvals are held to
# a margin and everything else follows the EV argmax.
APPROVAL_MARGIN = 1.5


def decide(probs: dict[str, float], *, allow_approval: bool = True) -> tuple[str, float]:
    """Expected-value argmax, fail-closed on approvals.

    Returns ``(adjudication, P(this decision is correct))`` - the quantity the
    Brier calibration term scores, not the winning class probability.
    """
    approved = probs.get(APPROVED, 0.0)
    denied = probs.get(DENIED, 0.0)
    if not allow_approval or approved < APPROVAL_MARGIN * denied:
        candidates = (DENIED, NEEDS_REVIEW)
    else:
        candidates = OUTCOMES

    best, best_ev = NEEDS_REVIEW, float("-inf")
    for candidate in candidates:
        ev = sum(PAYOFF[candidate][truth] * probs.get(truth, 0.0) for truth in OUTCOMES)
        if ev > best_ev:
            best, best_ev = candidate, ev
    return best, probs.get(best, 0.0)


# Paths that exist precisely because required evidence was missing or unread.
# The field manual routes incomplete packets to review, and an approval here
# would rest on absence rather than on evidence, so these can never approve
# however the sample happens to fall.
NO_APPROVAL_PATHS = frozenset({
    "risk_page_absent",
    "risk_page_unreadable",
    "risk_page_unreadable_med",
    "arrival_date_unknown",
    "arrival_date_untrusted",
    "visa_unknown",
    "sponsor_unknown",
    "fee_unknown",
    "fee_stated_unknown",
    "staleness_indeterminate",
    "adjudicator_conflict",
    "stamp_contested",
    "unreadable_packet",
})


class Calibration:
    """path -> empirical outcome distribution, fitted on public train labels."""

    def __init__(self, path: str | Path = DEFAULT_CALIBRATION):
        with open(path) as f:
            payload = json.load(f)
        self.paths: dict[str, dict[str, float]] = payload["paths"]
        self.fallback: dict[str, float] = payload["fallback"]

    def probs(self, path: str) -> dict[str, float]:
        return self.paths.get(path, self.fallback)

    def adjudicate(self, record: Record) -> tuple[str, float, str]:
        path = decision_path(record)
        adjudication, confidence = decide(
            self.probs(path), allow_approval=path not in NO_APPROVAL_PATHS
        )
        return adjudication, confidence, path


# --- Output-boundary guardrail ---------------------------------------------

def emitted_guardrail(output: dict, record: Record) -> tuple[str, float] | None:
    """One-way check that serialized fields cannot contradict the decision.

    Output-fill heuristics may complete fields after adjudication. This pass
    can only demote an APPROVED to DENIED/NEEDS_REVIEW - it can never create
    an approval - so output-only estimates cannot leak into a grant of work
    authorization. A visible signed manual finding outranks it.
    """
    if record.manual_finding in OUTCOMES:
        return None
    if output.get("adjudication") != APPROVED:
        return None

    flags = set()
    raw_flags = str(output.get("risk_flags", "none"))
    if raw_flags and raw_flags != "none":
        flags = set(raw_flags.split("|"))

    visa = str(output.get("visa_class", ""))
    if flags & DISQUALIFYING_FLAGS:
        return DENIED, 0.95
    if visa == "TRANSIT-7":
        return DENIED, 0.94
    if str(output.get("sponsor_id", "")) in REVOKED_SPONSORS and visa != "DIP-1":
        return DENIED, 0.94
    if str(output.get("home_world", "")) in EMBARGOED_HOME_WORLDS and visa != "DIP-1":
        return DENIED, 0.94
    if str(output.get("fee_status", "")) == "unpaid" and not record.has_hardship_waiver:
        return DENIED, 0.94
    if flags & REVIEW_FLAGS:
        return NEEDS_REVIEW, 0.85
    if str(output.get("fee_status", "")) == UNKNOWN:
        return NEEDS_REVIEW, 0.85
    return None
