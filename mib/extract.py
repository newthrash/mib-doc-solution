"""First-cut field extraction: labeled-line parsing over trusted page text.

This is deliberately conservative scaffolding: values come only from visible
text, matched against closed vocabularies, with per-field label contexts.
It will be reshaped by corpus reconnaissance once the training PDFs are
inspected; the interface (`Packet -> Record`) is the stable part.
"""

from __future__ import annotations

import re

from .constants import RISK_FLAGS, UNKNOWN
from .lexicon import (
    parse_date,
    repair_sponsor_id,
    snap_home_world,
    snap_purpose,
    snap_species,
    snap_visa,
)
from .pdfio import Packet
from .record import Record

_LABELS = {
    "applicant_name": ("APPLICANT NAME", "APPLICANT", "NAME"),
    "species_code": ("SPECIES CODE", "SPECIES"),
    "home_world": ("HOME WORLD", "HOMEWORLD", "WORLD"),
    "visa_class": ("VISA CLASS", "VISA"),
    "sponsor_id": ("SPONSOR ID", "SPONSOR"),
    "arrival_date": ("ARRIVAL DATE", "ARRIVAL"),
    "declared_purpose": ("DECLARED PURPOSE", "PURPOSE"),
    "fee_status": ("FEE STATUS",),
    "risk_flags": ("OBSERVED FLAGS", "RISK FLAGS", "FLAGS"),
}

_FINDING_RE = re.compile(
    r"FINDING\s*[:\-]?\s*(APPROVED|DENIED|NEEDS[\s_]*REVIEW)", re.IGNORECASE
)
_RECEIPT_RE = re.compile(
    r"(?:RECEIVED|RECEIPT\s+DATE|PACKET\s+RECEIVED)\s*[:\-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
    re.IGNORECASE,
)


def _labeled_values(lines: list[str], labels: tuple[str, ...]) -> list[str]:
    values = []
    for line in lines:
        upper = line.upper()
        for label in labels:
            position = upper.find(label)
            if position < 0:
                continue
            remainder = line[position + len(label):]
            remainder = remainder.lstrip(" :.|-\t")
            if remainder.strip():
                values.append(remainder.strip())
            break
    return values


def _first(values: list[str]) -> str | None:
    return values[0] if values else None


def _extract_flags(text: str) -> tuple[frozenset[str], bool]:
    """Return (flags, evidence_seen). `none` counts as evidence."""
    normalized = re.sub(r"[^a-z_|]+", " ", text.lower())
    found = {flag for flag in RISK_FLAGS if flag in normalized}
    explicit_none = bool(
        re.search(r"(?:OBSERVED|RISK)\s+FLAGS?\s*[:\-]?\s*none", text, re.IGNORECASE)
    )
    return frozenset(found), bool(found) or explicit_none


def extract_record(packet: Packet) -> Record:
    record = Record(case_id=packet.case_id)
    if packet.error or not packet.pages:
        record.risk_flags_known = False
        record.missing_fields = tuple(
            f for f in _LABELS
        )
        return record

    all_lines: list[str] = []
    for page in packet.pages:
        all_lines.extend(page.text.splitlines())
    full_text = packet.full_text()

    # Manual adjudicator finding: highest-precedence visible evidence.
    findings = _FINDING_RE.findall(full_text)
    if findings:
        final = findings[-1].upper().replace(" ", "_").replace("__", "_")
        record.manual_finding = "NEEDS_REVIEW" if "REVIEW" in final else final
        if record.manual_finding == "APPROVED" and len(findings) > 1:
            record.has_approval_override = True

    raw: dict[str, str | None] = {
        field: _first(_labeled_values(all_lines, labels))
        for field, labels in _LABELS.items()
    }

    if raw["applicant_name"]:
        name = re.split(r"\s{2,}|[|]", raw["applicant_name"])[0].strip()
        parts = [p for p in name.split() if re.fullmatch(r"[A-Za-z'’-]+", p)]
        if len(parts) >= 2:
            record.applicant_name = " ".join(parts[:2])
    if raw["species_code"]:
        record.species_code = snap_species(raw["species_code"]) or UNKNOWN
    if raw["home_world"]:
        record.home_world = snap_home_world(raw["home_world"]) or UNKNOWN
    if raw["visa_class"]:
        record.visa_class = snap_visa(raw["visa_class"]) or UNKNOWN
    if raw["sponsor_id"]:
        record.sponsor_id = repair_sponsor_id(raw["sponsor_id"]) or UNKNOWN
    elif match := re.search(r"SPN[-\s]?\d{4}", full_text):
        record.sponsor_id = repair_sponsor_id(match.group(0)) or UNKNOWN
    if raw["arrival_date"]:
        record.arrival_date = parse_date(raw["arrival_date"]) or UNKNOWN
    if raw["declared_purpose"]:
        record.declared_purpose = snap_purpose(raw["declared_purpose"]) or UNKNOWN

    if raw["fee_status"]:
        fee = raw["fee_status"].split()[0].lower().strip(".,;")
        if fee in ("paid", "waived", "unpaid", "unknown"):
            record.fee_status = fee
            record.fee_explicit_unknown = fee == "unknown"
    if record.fee_status == UNKNOWN and not record.fee_explicit_unknown:
        # Receipt geometry fallback: the standard fee amount implies payment.
        if re.search(r"\$\s*809(?:[.,]00)?\b", full_text):
            record.fee_status = "paid"

    flags, flags_seen = _extract_flags(full_text)
    record.risk_flags = flags
    record.risk_flags_known = flags_seen

    record.has_hardship_waiver = bool(
        re.search(r"HARDSHIP\s+WAIVER|WAIVER\s+APPROVED|DIP-?WAIVER", full_text, re.IGNORECASE)
    )
    record.has_diplomatic_note = bool(
        re.search(r"DIPLOMATIC\s+NOTE", full_text, re.IGNORECASE)
    )
    if match := _RECEIPT_RE.search(full_text):
        record.receipt_date = match.group(1)

    record.injection_detected = packet.injection_detected
    record.has_scanned_pages = packet.has_scanned_pages
    record.pages = len(packet.pages)
    confidences = [p.ocr_confidence for p in packet.pages if p.ocr_confidence > 0]
    record.ocr_mean_confidence = sum(confidences) / len(confidences) if confidences else 0.0

    record.missing_fields = tuple(
        f
        for f in _LABELS
        if f != "risk_flags"
        and getattr(record, f) in (UNKNOWN, "", None)
    )
    return record
