"""Closed vocabularies and policy constants for the MIB Doc Challenge.

Sources are marked per constant:
- PUBLIC: stated in FIELD_MANUAL.md or the challenge repo.
- MINED: inferred from the public training labels, which the PRD explicitly
  invites ("Candidates are expected to infer missing details from labeled
  examples"). Mined constants are also re-derivable at score time from the
  scored corpus itself (see policy.corpus_context) so a private test set with
  different values degrades gracefully instead of breaking.
"""

from __future__ import annotations

# --- Output schema ---------------------------------------------------------

OUTPUT_FIELDS = (
    "case_id",
    "applicant_name",
    "species_code",
    "home_world",
    "visa_class",
    "sponsor_id",
    "arrival_date",
    "declared_purpose",
    "risk_flags",
    "fee_status",
    "adjudication",
    "confidence",
)

ADJUDICATIONS = ("APPROVED", "DENIED", "NEEDS_REVIEW")
FEE_VALUES = ("paid", "waived", "unpaid", "unknown")

# --- Closed vocabularies (PUBLIC: observed in public labels/schemas) -------

SPECIES_CODES = (
    "ALPHA_DRACONIAN",
    "ANDROMEDAN",
    "AQUARIAN_MANTIS",
    "ARCTURIAN",
    "CENTAURI_SYNTH",
    "JOVIAN_GASFORM",
    "KAIJU_MICRO",
    "LUNA_SECURID",
    "ORION_GRAYS",
    "SIRIUS_AVIAN",
    "TRIANGULAN",
    "VENUSIAN_MYCELIAL",
)

HOME_WORLDS = (
    "Barnard-c",
    "Eris Relay",  # absent from train labels; appears in public solution vocab
    "Europa Station",
    "Gliese-581g",
    "Kepler-186f",
    "Luyten-b",
    "Mars Dome-7",
    "Proxima-b",
    "Sirius Outpost",
    "Titan Freeport",
    "TRAPPIST-1e",
    "Wolf-1061c",
    "Zeta Reticuli",
)

VISA_CLASSES = ("XW-1", "XW-2", "DIP-1", "MED-3", "TRANSIT-7")

PURPOSES = (
    "archive audit",
    "cultural exchange",
    "diplomatic",
    "field repair",
    "medical consult",
    "reactor maintenance",
    "research",
    "transit",
    "translation",
    "xenobotany",
)

RISK_FLAGS = (
    "active_warrant",
    "biohazard_red",
    "identity_conflict",
    "illegible_biometrics",
    "memory_tampering",
    "planetary_embargo",
    "rescinded_denial",
    "sponsor_mismatch",
)

# --- Policy constants ------------------------------------------------------

# PUBLIC: FIELD_MANUAL.md "Risk Flags".
DISQUALIFYING_FLAGS = frozenset(
    {"memory_tampering", "planetary_embargo", "active_warrant", "biohazard_red"}
)
REVIEW_FLAGS = frozenset(
    {"identity_conflict", "sponsor_mismatch", "illegible_biometrics", "rescinded_denial"}
)

# PUBLIC: FIELD_MANUAL.md "Sponsor Rules".
REVOKED_SPONSORS_PUBLIC = frozenset({"SPN-0007", "SPN-0139", "SPN-4040"})

# MINED: these three sponsor ids recur across dozens of otherwise-clean train
# packets with ~75-80% denial rates, matching the manual's note that "other
# revoked sponsors may appear in examples". Also detected at score time by
# frequency-outlier analysis (policy.corpus_context) as a transfer-safe backstop.
REVOKED_SPONSORS_MINED = frozenset({"SPN-2718", "SPN-7331", "SPN-9090"})

REVOKED_SPONSORS = REVOKED_SPONSORS_PUBLIC | REVOKED_SPONSORS_MINED

# MINED: non-diplomatic packets from this world are consistently denied in the
# training labels (planetary embargo policy the manual leaves implicit).
EMBARGOED_HOME_WORLDS = frozenset({"Wolf-1061c"})

# PUBLIC: FIELD_MANUAL.md "Date Rules".
STALE_DAYS = 180

# The evaluator's payoff matrix, transcribed from
# scripts/evaluate.py::classification_points. Rows are predictions,
# columns are ground truth.
PAYOFF = {
    "APPROVED": {"APPROVED": 8.0, "DENIED": -4.0, "NEEDS_REVIEW": 1.0},
    "DENIED": {"APPROVED": 0.0, "DENIED": 8.0, "NEEDS_REVIEW": 1.0},
    "NEEDS_REVIEW": {"APPROVED": 2.0, "DENIED": 2.0, "NEEDS_REVIEW": 8.0},
}

UNKNOWN = "unknown"
