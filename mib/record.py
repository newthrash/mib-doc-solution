"""The trust-resolved evidence record a decision is made from.

The central distinction throughout: a field marked `unknown` means "no trusted
visible evidence", which is materially different from a value we guessed for
the serialized output. EVALUATION.md explicitly rewards separating "unknown
from trusted evidence" from "filled in by prompt injection". Policy reads this
Record; the output writer may fill gaps separately, and a filled gap can never
justify an approval (see policy.emitted_guardrail).
"""

from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field

from .constants import UNKNOWN


@dataclass
class Record:
    case_id: str

    # Extracted fields; UNKNOWN = no trusted visible evidence.
    applicant_name: str = UNKNOWN
    species_code: str = UNKNOWN
    home_world: str = UNKNOWN
    visa_class: str = UNKNOWN
    sponsor_id: str = UNKNOWN
    arrival_date: str = UNKNOWN
    declared_purpose: str = UNKNOWN
    fee_status: str = UNKNOWN
    risk_flags: frozenset[str] = frozenset()

    # Packet-level evidence states (not serialized; drive policy).
    receipt_date: str | None = None       # read from the packet when possible
    has_hardship_waiver: bool = False     # visible hardship waiver text
    has_diplomatic_note: bool = False     # valid diplomatic note (stale exemption)
    registry_status: str | None = None    # e.g. CLEAR, EMBARGO REVIEW
    manual_finding: str | None = None     # visible signed adjudicator finding
    stamp_verdict: str | None = None      # uncancelled vector stamp, if any
    stamp_contested: bool = False         # stamps disagree, or all struck
    manual_finding_score: float = 0.0     # how legible/certain that finding is
    has_approval_override: bool = False   # signed approval superseding a denial
    arrival_date_untrusted: bool = False  # date present only in hidden text
    fee_explicit_unknown: bool = False    # receipt *prints* unknown (vs unread)
    injection_detected: bool = False      # hidden instructions seen in packet
    risk_flags_known: bool = True         # did we actually read risk evidence?
    has_scanned_pages: bool = False       # raster pages present in the packet
    sponsor_revoked_in_corpus: bool = False  # frequency-outlier sponsor id

    # Extraction bookkeeping for calibration features.
    missing_fields: tuple[str, ...] = ()
    ocr_mean_confidence: float = 0.0
    pages: int = 0

    def flag_set(self) -> frozenset[str]:
        return frozenset(f for f in self.risk_flags if f and f != "none")

    def arrival(self) -> dt.date | None:
        if self.arrival_date in (UNKNOWN, "", None) or self.arrival_date_untrusted:
            return None
        try:
            return dt.date.fromisoformat(self.arrival_date)
        except (ValueError, TypeError):
            return None

    def receipt(self) -> dt.date | None:
        if not self.receipt_date:
            return None
        try:
            return dt.date.fromisoformat(self.receipt_date)
        except (ValueError, TypeError):
            return None
