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
    snap_name,
    snap_name_token,
    snap_flag,
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
# The registry extract states an embargo outright. Reading it beats inferring
# one from a mined world list: on the public corpus EMBARGO REVIEW appears for
# three different home worlds, and a private set may use others entirely.
_REGISTRY_STATUS_RE = re.compile(
    r"REGISTRY\s+STATUS\s*[:\n]?\s*([A-Z][A-Z ]{2,25})", re.IGNORECASE
)
_WAIVER_CODE_RE = re.compile(
    r"WAIVER\s+CODE\s*[:\-]?\s*(?!N/?A\b)([A-Z][A-Z0-9-]{2,})", re.IGNORECASE
)


def _parse_name(value: str) -> str | None:
    """Parse a two-token name and repair OCR damage against the vocabulary.

    A candidate that resolves to another field's vocabulary is field bleed -
    the label's own value was blank and the scan ran into a neighbouring cell.
    'Home Europa' is a home world, never an applicant.
    """
    if snap_home_world(value) or snap_species(value) or snap_purpose(value):
        return None
    name = re.split(r"\s{2,}|[|]", value)[0].strip()
    parts = [p for p in name.split() if re.fullmatch(r"[A-Za-z'’-]+", p)]
    if len(parts) < 2:
        return None
    candidate = " ".join(parts[:2])
    # Applicant names are drawn from a known token set. Requiring at least one
    # token to belong to it rejects merged-column debris like "Home Europa",
    # which no amount of trimming would otherwise distinguish from a name.
    if not any(snap_name_token(part) for part in parts[:2]):
        return None
    # A closed vocabulary turns 'Mirequell Qcrul' back into a real applicant;
    # tokens it cannot place are kept verbatim rather than discarded.
    return snap_name(candidate) or candidate


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


def _fuzzy_label_prefix(stripped: str, label: str) -> str | None:
    """Match a possibly OCR-damaged label at the start of a line.

    'Sponser ID: SPN-8802' carries a perfect value behind a misspelled label;
    exact prefix matching threw the value away. Each label word may differ by
    up to a third of its characters - safe on printed form labels in a way it
    would not be on values, and any junk this admits is still filtered by the
    field's own parser and vocabulary snap.
    """
    from .lexicon import weighted_distance

    label_words = label.split()
    line_words = stripped.split()
    if len(line_words) < len(label_words):
        return None
    for expected, seen in zip(label_words, line_words):
        cleaned = re.sub(r"[^A-Za-z]", "", seen).upper()
        if not cleaned:
            return None
        if weighted_distance(cleaned, expected) / len(expected) > 0.34:
            return None
    return " ".join(line_words[len(label_words):]).lstrip(" :.|-\t")


def _labeled_values(lines: list[str], labels: tuple[str, ...]) -> list[str]:
    """Collect values for `labels` in both packet layouts.

    Layout A (native form text): the label occupies its own line and the value
    is the next non-noise line. Layout B (OCR of the same page): the label and
    value share a line, separated by a colon. Labels are matched exactly first
    and fuzzily second, so damaged labels ('Sponser ID', 'Visa Cisse') no
    longer strand intact values.
    """
    values: list[str] = []
    for index, line in enumerate(lines):
        # OCR frequently prefixes a line with a stray mark ("'Applicant:"),
        # which silently defeats a prefix match and strands the real value.
        stripped = re.sub(r"^[^A-Za-z0-9]+", "", line.strip())
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
            remainder = _fuzzy_label_prefix(stripped, label)
            if remainder is not None:
                if remainder and not _is_noise(remainder):
                    values.append(remainder)
                elif not remainder:
                    for candidate in lines[index + 1: index + 3]:
                        if not _is_noise(candidate):
                            values.append(candidate.strip())
                            break
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


# Document precedence from FIELD_MANUAL.md, "Trusted Evidence". Pages whose
# type could not be identified rank below every recognised document: an
# unclassified page is often a damaged or decoy fragment, and letting it
# outrank a biometric slip is how a correct value gets overwritten.
_PAGE_RANK = {
    "manual": 6,
    "intake": 5,
    "biometric": 4,
    "fee": 4,
    "sponsor": 3,
    "registry": 2,
    "unknown": 0,
}


def _consensus(
    candidates: list[tuple[str, str]], parser, *, authority_first: bool = False
) -> str | None:
    """Resolve a field from candidates carrying their source page type.

    Corroboration is the primary signal and document authority breaks ties. A
    value repeated across independent documents is evidence in a way that a
    single authoritative page is not, and ranking authority first measurably
    hurt: it replaced 61 correct applicant names with names taken from another
    page, fixing only 17.

    `authority_first` inverts that for fields where the manual's precedence is
    the point - a later signed note supersedes an earlier form outright rather
    than being outvoted by repetition of the stale value.
    """
    parsed: list[tuple[int, str]] = []
    for value, kind in candidates:
        result = parser(value)
        if result:
            parsed.append((_PAGE_RANK.get(kind, 0), result))
    if not parsed:
        return None

    agreement = Counter(value for _, value in parsed)
    authority: dict[str, int] = {}
    for rank, value in parsed:
        authority[value] = max(authority.get(value, 0), rank)

    def key(item):
        _, value = item
        if authority_first:
            return (authority[value], agreement[value])
        return (agreement[value], authority[value])

    return max(parsed, key=key)[1]


# The flag label is itself OCR-damaged on scanned slips ("Observed fiags",
# "Ohserved flags"), so it is matched loosely and its value snapped to the
# closed flag vocabulary.
_FLAG_LABEL_RE = re.compile(
    r"\b[o0][bh]?s[ea]rv[ea]d\s+f[il1]ags?\b|\brisk\s+f[il1]ags?\b", re.IGNORECASE
)
_NONE_RE = re.compile(r"^\W*(?:none|nene|nune|no[nm]e|n/?a)\W*$", re.IGNORECASE)


def _extract_flags(lines: list[str], text: str) -> tuple[frozenset[str], bool]:
    """Return (flags, evidence_seen).

    An explicit `none` is evidence exactly as much as a named flag is: it is
    the difference between a slip stating the applicant is clean and a slip we
    could not read. Only the former may support an approval.
    """
    found: set[str] = set()
    seen = False

    for index, line in enumerate(lines):
        match = _FLAG_LABEL_RE.search(line)
        if not match:
            continue
        seen = True
        tail = line[match.end():].lstrip(" :.|-\t")
        candidates = [tail] if tail.strip() else []
        candidates += [l for l in lines[index + 1: index + 3] if l.strip()]
        for candidate in candidates:
            if _NONE_RE.match(candidate.strip()):
                return frozenset(), True
            for token in re.split(r"[|,;]+", candidate):
                flag = snap_flag(token)
                if flag:
                    found.add(flag)
            if found:
                break

    # Flags named anywhere in trusted text still count: some packets record
    # findings in prose rather than as a labeled slip field.
    normalized = re.sub(r"[^a-z_]+", " ", text.lower())
    found |= {flag for flag in RISK_FLAGS if flag in normalized}

    # A literal scan misses OCR-damaged names ('illegible_biometrics' read as
    # 'llegible_biometrics'). Underscore-joined tokens are distinctive enough
    # here that snapping them is safe - ordinary form prose contains none -
    # and recovering a flag only ever makes the decision more conservative.
    for token in re.findall(r"[a-z]{3,}_[a-z_]{3,}", normalized):
        flag = snap_flag(token)
        if flag:
            found.add(flag)

    return frozenset(found), seen or bool(found)


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

    # Candidates keep the page type they came from so precedence can be
    # applied; a bare list of strings cannot express that a value came from a
    # decoy fragment rather than the intake form.
    raw: dict[str, list[tuple[str, str]]] = {field: [] for field in _LABELS}
    for page in packet.pages:
        page_lines = page.text.splitlines()
        for field, labels in _LABELS.items():
            raw[field].extend(
                (value, page.kind) for value in _labeled_values(page_lines, labels)
            )

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
        prose = [(m, "sponsor") for m in _SPONSOR_PROSE_RE.findall(full_text)]
        sponsor = _consensus(prose, repair_sponsor_id)
    record.sponsor_id = sponsor or UNKNOWN

    fee = _first_parsed([v for v, _ in raw["fee_status"]], _parse_fee)
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

    # Targeted cell re-reads fill fields whole-page OCR never surfaced. They
    # pass through the same vocabulary snapping as any other reading, so a
    # garbled crop is rejected rather than trusted.
    roi = packet.roi_values
    if record.fee_status == UNKNOWN and "fee_status" in roi:
        recovered = _parse_fee(roi["fee_status"])
        if recovered:
            record.fee_status = recovered
            record.fee_explicit_unknown = recovered == "unknown"
    if record.visa_class == UNKNOWN and "visa_class" in roi:
        record.visa_class = snap_visa(roi["visa_class"]) or UNKNOWN
    if record.sponsor_id == UNKNOWN and "sponsor_id" in roi:
        record.sponsor_id = repair_sponsor_id(roi["sponsor_id"]) or UNKNOWN
    if record.arrival_date == UNKNOWN and "arrival_date" in roi:
        record.arrival_date = parse_date(roi["arrival_date"]) or UNKNOWN
    if record.species_code == UNKNOWN and "species_code" in roi:
        record.species_code = snap_species(roi["species_code"]) or UNKNOWN
    if record.declared_purpose == UNKNOWN and "declared_purpose" in roi:
        record.declared_purpose = snap_purpose(roi["declared_purpose"]) or UNKNOWN
    if record.home_world == UNKNOWN and "home_world" in roi:
        record.home_world = snap_home_world(roi["home_world"]) or UNKNOWN

    flags, flags_seen = _extract_flags(all_lines, full_text)
    if not flags_seen and "risk_flags" in roi:
        raw_roi = roi["risk_flags"]
        if raw_roi.strip().lower() == "none":
            flags, flags_seen = frozenset(), True
        else:
            roi_flags, roi_seen = _extract_flags(
                [f"Observed flags: {raw_roi}"], raw_roi
            )
            if roi_seen:
                flags, flags_seen = roi_flags, True
    record.risk_flags = flags
    record.risk_flags_known = flags_seen

    record.has_hardship_waiver = bool(
        re.search(r"HARDSHIP\s+WAIVER|WAIVER\s+APPROVED|DIP-?WAIVER", full_text, re.IGNORECASE)
    )
    record.has_diplomatic_note = bool(
        re.search(r"DIPLOMATIC\s+NOTE", full_text, re.IGNORECASE)
    )
    if match := _REGISTRY_STATUS_RE.search(full_text):
        record.registry_status = " ".join(match.group(1).split()).upper()
    if match := _RECEIPT_RE.search(full_text):
        record.receipt_date = match.group(1)

    record.stamp_verdict = packet.stamp_verdict
    record.stamp_contested = packet.stamp_contested
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
