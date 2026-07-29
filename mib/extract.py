"""First-cut field extraction: labeled-line parsing over trusted page text.

This is deliberately conservative scaffolding: values come only from visible
text, matched against closed vocabularies, with per-field label contexts.
It will be reshaped by corpus reconnaissance once the training PDFs are
inspected; the interface (`Packet -> Record`) is the stable part.
"""

from __future__ import annotations

import re
from collections import Counter

from .constants import RISK_FLAGS, UNKNOWN
from .lexicon import (
    parse_date,
    snap_fee,
    repair_sponsor_id,
    snap_home_world,
    snap_purpose,
    snap_species,
    snap_visa,
)
from .pdfio import Packet
from .record import Record

# Packet forms lay out a field as the label on one line and the value on the
# next. OCR of the same page more often yields `Label: value` on one line, so
# both shapes are parsed. Labels are ordered most- to least-specific because
# the first match wins ("Species Code" must beat a bare "Species").
_LABELS = {
    "applicant_name": ("APPLICANT NAME", "REGISTRY NAME", "APPLICANT"),
    "species_code": ("SPECIES CODE", "SPECIES MATCH", "SPECIES"),
    "home_world": ("HOME WORLD", "HOMEWORLD"),
    "visa_class": ("VISA CLASS", "VISA"),
    "sponsor_id": ("SPONSOR ID", "SPONSOR"),
    "arrival_date": ("ARRIVAL DATE", "ARRIVAL"),
    "declared_purpose": ("DECLARED PURPOSE", "PURPOSE"),
    "fee_status": ("FEE STATUS",),
    "risk_flags": ("OBSERVED FLAGS", "RISK FLAGS", "FLAGS"),
}

# Footer boilerplate and image placeholders are never field values.
_NOISE_RE = re.compile(
    r"^(?:Packet\s+MIB-\d+|Synthetic hiring|MIB-\d+\s*\|\s*MIB Eyes Only"
    r"|(?:REGISTRY|PASSPORT|SCAN|BIOMETRIC)\s+IMAGE|N/?A|-+)\s*$",
    re.IGNORECASE,
)

# A value line that is itself another field label means the real value was
# blank or unreadable; adopting it would emit a label as a field value.
_ALL_LABELS = {name for names in _LABELS.values() for name in names} | {
    "CASE ID", "AMOUNT", "WAIVER CODE", "REGISTRY STATUS", "FINDING",
    "BIOMETRIC CONFIDENCE", "PRIMARY INTAKE RECORD",
}

_FINDING_RE = re.compile(
    r"FINDING\s*[:\-]?\s*(APPROVED|DENIED|NEEDS[\s_]*REVIEW)", re.IGNORECASE
)
_RECEIPT_RE = re.compile(
    r"(?:RECEIVED|RECEIPT\s+DATE|PACKET\s+RECEIVED)\s*[:\-]?\s*([0-9]{4}-[0-9]{2}-[0-9]{2})",
    re.IGNORECASE,
)

# A signed manual correction supersedes the printed form field.
_SPONSOR_CORRECTION_RE = re.compile(
    r"(?:MANUAL\s+)?CORRECTION[^.\n]*?SPONSOR\s+(?:IS|=|:)?\s*([A-Z0-9-]{6,10})",
    re.IGNORECASE,
)
# Sponsor letters state the id in prose rather than as a labeled field.
_SPONSOR_PROSE_RE = re.compile(
    r"\bSPONSOR\s+(S[PFR][NRM][-\s]?[0-9OQDILZSBGT]{4})\b", re.IGNORECASE
)
_WAIVER_CODE_RE = re.compile(
    r"WAIVER\s+CODE\s*[:\-]?\s*(?!N/?A\b)([A-Z][A-Z0-9-]{2,})", re.IGNORECASE
)


def _parse_name(value: str) -> str | None:
    name = re.split(r"\s{2,}|[|]", value)[0].strip()
    parts = [p for p in name.split() if re.fullmatch(r"[A-Za-z'’-]+", p)]
    if len(parts) < 2:
        return None
    return " ".join(parts[:2])


def _parse_fee(value: str) -> str | None:
    token = value.strip().split()[0] if value.strip() else ""
    if token.lower().strip(".,;:") in ("paid", "waived", "unpaid", "unknown"):
        return token.lower().strip(".,;:")
    return snap_fee(token)


def _is_noise(line: str) -> bool:
    stripped = line.strip()
    if not stripped or _NOISE_RE.match(stripped):
        return True
    return stripped.upper().rstrip(":") in _ALL_LABELS


def _labeled_values(lines: list[str], labels: tuple[str, ...]) -> list[str]:
    """Collect values for `labels` in both packet layouts.

    Layout A (native form text): the label occupies its own line and the value
    is the next non-noise line. Layout B (OCR of the same page): the label and
    value share a line, separated by a colon.
    """
    values: list[str] = []
    for index, line in enumerate(lines):
        stripped = line.strip()
        upper = stripped.upper().rstrip(":")
        for label in labels:
            if upper == label:
                for candidate in lines[index + 1: index + 3]:
                    if not _is_noise(candidate):
                        values.append(candidate.strip())
                        break
                break
            if upper.startswith(label):
                remainder = stripped[len(label):].lstrip(" :.|-\t")
                if remainder and not _is_noise(remainder):
                    values.append(remainder)
                break
    return values


def _first_parsed(values: list[str], parser) -> str | None:
    """Return the first candidate that actually parses.

    A label match does not imply a usable value: "Sponsor Attestation Letter"
    matches the SPONSOR label and yields prose. Trying only the first candidate
    strands the real value later in the packet, so every candidate is offered
    to the field's parser and the first success wins.
    """
    for value in values:
        parsed = parser(value)
        if parsed:
            return parsed
    return None


def _consensus(values: list[str], parser) -> str | None:
    """Prefer the parsed value seen on the most pages, then the first seen.

    Fields are repeated across intake form, registry extract, and biometric
    slip. When OCR garbles one copy, agreement across pages recovers the
    truth without trusting any single reading.
    """
    parsed = [p for p in (parser(v) for v in values) if p]
    if not parsed:
        return None
    counts = Counter(parsed)
    best = max(counts.values())
    for value in parsed:  # stable: earliest value among the tied winners
        if counts[value] == best:
            return value
    return None


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

    raw: dict[str, list[str]] = {
        field: _labeled_values(all_lines, labels)
        for field, labels in _LABELS.items()
    }

    record.applicant_name = _consensus(raw["applicant_name"], _parse_name) or UNKNOWN
    record.species_code = _consensus(raw["species_code"], snap_species) or UNKNOWN
    record.home_world = _consensus(raw["home_world"], snap_home_world) or UNKNOWN
    record.visa_class = _consensus(raw["visa_class"], snap_visa) or UNKNOWN
    record.arrival_date = _consensus(raw["arrival_date"], parse_date) or UNKNOWN
    record.declared_purpose = _consensus(raw["declared_purpose"], snap_purpose) or UNKNOWN

    # Sponsor id, in precedence order: a signed manual correction outranks the
    # form field, which outranks a sponsor letter's prose.
    sponsor = None
    if match := _SPONSOR_CORRECTION_RE.search(full_text):
        sponsor = repair_sponsor_id(match.group(1))
    if not sponsor:
        sponsor = _consensus(raw["sponsor_id"], repair_sponsor_id)
    if not sponsor:
        prose = _SPONSOR_PROSE_RE.findall(full_text)
        sponsor = _consensus(prose, repair_sponsor_id)
    record.sponsor_id = sponsor or UNKNOWN

    fee = _first_parsed(raw["fee_status"], _parse_fee)
    if fee:
        record.fee_status = fee
        record.fee_explicit_unknown = fee == "unknown"
    else:
        # Receipt geometry: the standard fee amount implies payment, and a
        # printed waiver code implies a waiver, when the status word is damaged.
        if re.search(r"\$\s*809(?:[.,]0{2})?\b", full_text):
            record.fee_status = "paid"
        elif _WAIVER_CODE_RE.search(full_text):
            record.fee_status = "waived"

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
