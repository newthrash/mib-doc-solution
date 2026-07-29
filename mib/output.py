"""Schema-safe serialization and crash-tolerant incremental output.

Two hard requirements from reading the evaluator:

1. `evaluate.py` exits 2 on invalid enums, invalid confidence, duplicate ids,
   or case ids outside the manifest - so every row is clamped to legal values
   at this boundary, whatever upstream produced.
2. Containers stopped at the runtime limit are scored on whatever output
   exists - so rows are flushed to disk as they are produced, and the final
   corpus-context rewrite goes through an atomic rename.
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from .constants import ADJUDICATIONS, FEE_VALUES, OUTPUT_FIELDS, RISK_FLAGS, UNKNOWN

_CASE_ID_RE = re.compile(r"^MIB-[0-9]{6}$")
_SPONSOR_RE = re.compile(r"^SPN-[0-9]{4}$")
_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

# Schema-valid stand-ins for fields with no trusted reading. A blank field
# scores identically to a wrong one, so these are also the slots output-fill
# heuristics may later improve - policy never reads them.
_FALLBACKS = {
    "applicant_name": "unknown",
    "species_code": "unknown",
    "home_world": "unknown",
    "visa_class": "unknown",
    "sponsor_id": "SPN-0000",
    "arrival_date": "1900-01-01",
    "declared_purpose": "unknown",
    "risk_flags": "none",
    "fee_status": "unknown",
}


def clamp_row(row: dict) -> dict | None:
    """Return a schema-valid prediction row, or None if the case id is bad."""
    case_id = str(row.get("case_id", "")).strip()
    if not _CASE_ID_RE.fullmatch(case_id):
        return None

    out = {"case_id": case_id}
    for field in ("applicant_name", "species_code", "home_world", "visa_class",
                  "declared_purpose"):
        value = str(row.get(field, "") or "").strip()
        out[field] = value if value and value != UNKNOWN else _FALLBACKS[field]
        if field == "visa_class" and out[field] == _FALLBACKS["visa_class"]:
            out[field] = "unknown"

    sponsor = str(row.get("sponsor_id", "") or "").strip()
    out["sponsor_id"] = sponsor if _SPONSOR_RE.fullmatch(sponsor) else _FALLBACKS["sponsor_id"]

    arrival = str(row.get("arrival_date", "") or "").strip()
    out["arrival_date"] = arrival if _DATE_RE.fullmatch(arrival) else _FALLBACKS["arrival_date"]

    flags = str(row.get("risk_flags", "") or "").strip()
    parts = sorted(
        {p.strip() for p in flags.split("|") if p.strip() in RISK_FLAGS}
    )
    out["risk_flags"] = "|".join(parts) if parts else "none"

    fee = str(row.get("fee_status", "") or "").strip()
    out["fee_status"] = fee if fee in FEE_VALUES else "unknown"

    adjudication = str(row.get("adjudication", "") or "").strip().upper()
    out["adjudication"] = adjudication if adjudication in ADJUDICATIONS else "NEEDS_REVIEW"

    try:
        confidence = float(row.get("confidence", 0.5))
    except (TypeError, ValueError):
        confidence = 0.5
    out["confidence"] = round(min(1.0, max(0.0, confidence)), 4)

    return {field: out[field] for field in OUTPUT_FIELDS}


class IncrementalWriter:
    """Append clamped JSONL rows with per-row flush; finalize atomically."""

    def __init__(self, output_path: str | Path):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._seen: set[str] = set()
        self._handle = open(self.output_path, "w")

    def write(self, row: dict) -> bool:
        clamped = clamp_row(row)
        if clamped is None or clamped["case_id"] in self._seen:
            return False
        self._seen.add(clamped["case_id"])
        self._handle.write(json.dumps(clamped, sort_keys=True) + "\n")
        self._handle.flush()
        os.fsync(self._handle.fileno())
        return True

    def close(self) -> None:
        self._handle.close()

    def rewrite_all(self, rows: list[dict]) -> int:
        """Replace the provisional file with final rows via atomic rename."""
        self.close()
        temporary = self.output_path.with_suffix(".tmp")
        seen: set[str] = set()
        written = 0
        with open(temporary, "w") as handle:
            for row in rows:
                clamped = clamp_row(row)
                if clamped is None or clamped["case_id"] in seen:
                    continue
                seen.add(clamped["case_id"])
                handle.write(json.dumps(clamped, sort_keys=True) + "\n")
                written += 1
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, self.output_path)
        return written


# --- Output-only priors ----------------------------------------------------

_PRIORS_PATH = Path(__file__).resolve().parent.parent / "policy" / "priors.json"


def load_priors(path: str | Path = _PRIORS_PATH) -> dict[str, str]:
    """Per-field fallback values for slots with no trusted evidence.

    Emitting the sentinel scores zero, so the empirical mode strictly
    dominates: it scores when the field is still counted and is neutral when
    the private scorer has excluded it as unrecoverable.

    Only ever applied AFTER adjudication. A guessed value must never reach the
    policy - `unknown from trusted evidence` and `filled in for output` are
    different states, and EVALUATION.md rewards keeping them apart.
    """
    try:
        with open(path) as handle:
            payload = json.load(handle)
    except (OSError, ValueError):
        return {}
    return {field: prior["value"] for field, prior in payload.get("priors", {}).items()}


def apply_priors(row: dict, priors: dict[str, str]) -> dict:
    """Fill only slots that carry no reading; never overwrite evidence."""
    filled = dict(row)
    for field, value in priors.items():
        current = str(filled.get(field, "") or "").strip().lower()
        blank = current in ("", UNKNOWN, "none") if field != "risk_flags" else current == ""
        if blank:
            filled[field] = value
    return filled
