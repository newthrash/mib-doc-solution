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
    split_fused,
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

# The verdict may precede its label. OCR of a damaged manual note returns the
# words of each line in reverse order - "DENIED Finding:" for "Finding:
# DENIED", "Adjudicator Manual Note" for "Manual Adjudicator Note" - and the
# label-first form alone missed the note on MIB-000519 entirely, approving a
# packet whose own adjudicator had written DENIED on it. Both orders are
# accepted; the caller takes whichever group matched.
_VERDICT = r"APPROVED|DENIED|NEEDS[\s_]*REVIEW"
_FINDING_RE = re.compile(
    rf"FINDING\s*[:\-]?\s*({_VERDICT})|({_VERDICT})\s*[:\-]?\s*FINDING",
    re.IGNORECASE,
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
# Amendment annotations state a field's value in a sentence, so no per-field
# label parser ever sees them. They are authoritative rather than corroborating:
# across the public corpus they match the label on 422 of 422 readable cases,
# never contradict the sponsor letter's clause, and no packet carries two
# conflicting values. A match therefore replaces the form field instead of
# merely filling a blank - worth +54 fields recovered from blanks and, again
# nearly as much, +50 wrong readings corrected.
_CORRECTION_RE = {
    "visa_class": re.compile(
        r"CORRECTION[^.\n]*?VISA\s+CLASS\s+(?:IS|=|:)?\s*([A-Z0-9-]{3,12})", re.IGNORECASE
    ),
    "fee_status": re.compile(
        r"CORRECTION[^.\n]*?FEE\s+STATUS\s+(?:IS|=|:)?\s*([A-Za-z]{3,10})", re.IGNORECASE
    ),
    "applicant_name": re.compile(
        r"CORRECTION[^.\n]*?APPLICANT\s+(?:IS|=|:)?\s*([^.\n]{3,40})", re.IGNORECASE
    ),
}
# Sponsor letters name the visa class inside a compliance clause.
_COMPLIANCE_CLASS_RE = re.compile(
    r"RESPONSIBILITY\s+FOR\s+CLASS\s+([A-Z0-9-]{3,12})\s+COMPLIANCE", re.IGNORECASE
)
# The registry extract states an embargo outright. Reading it beats inferring
# one from a mined world list: on the public corpus EMBARGO REVIEW appears for
# three different home worlds, and a private set may use others entirely.
_REGISTRY_STATUS_RE = re.compile(
    r"REGISTRY\s+STATUS\s*[:\n]?\s*([A-Z][A-Z ]{2,25})", re.IGNORECASE
)
# Packets mark destroyed fields inline: [NAME CUT OUT], [SPONSOR ID BLANK],
# [SPECIES WHITEOUT], [FEE STATUS OBSCURED], [VISA CLASS TORN]. These are the
# document stating a field is unrecoverable - stronger evidence than our own
# failure to read one. Measured on the public corpus: 38 tags across 27 of 400
# packets, and in five of those we were extracting a value anyway for a field
# the document says is gone. A phantom sponsor can invent or erase a
# revocation, so a tagged field is forced blank rather than trusted.
_DAMAGE_TAG_RE = re.compile(r"\[([A-Z][A-Z ]{2,28})\]")
_DAMAGE_WORDS = ("CUT OUT", "BLANK", "WHITEOUT", "LOST", "WASHED", "OBSCURED",
                 "TORN", "ILLEGIBLE", "MISSING", "REDACTED", "UNREADABLE")
_TAG_FIELDS = (
    ("SPONSOR", "sponsor_id"),
    ("SPECIES", "species_code"),
    ("VISA", "visa_class"),
    ("FEE", "fee_status"),
    ("PURPOSE", "declared_purpose"),
    ("WORLD", "home_world"),
    ("DATE", "arrival_date"),
    ("NAME", "applicant_name"),
)


def _damaged_fields(text: str) -> set[str]:
    """Fields the packet itself marks as destroyed."""
    damaged: set[str] = set()
    for tag in _DAMAGE_TAG_RE.findall(text):
        if not any(word in tag for word in _DAMAGE_WORDS):
            continue
        for token, field in _TAG_FIELDS:
            if token in tag:
                damaged.add(field)
                break
    return damaged


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


def _collapsed_label_prefix(stripped: str, label: str) -> str | None:
    """Match a label whose interior spaces OCR dropped ('FeeStatus:paid').

    The second OCR engine frequently omits spaces, leaving label and value
    fused in one token. Compare against the space-collapsed label with the
    same per-character tolerance used elsewhere.
    """
    from .lexicon import weighted_distance

    collapsed = label.replace(" ", "")
    head = re.sub(r"[^A-Za-z]", "", stripped[: len(collapsed) + 2])[: len(collapsed)]
    if len(head) < len(collapsed) - 1:
        return None
    if weighted_distance(head.upper(), collapsed) / len(collapsed) > 0.25:
        return None
    index = 0
    matched = 0
    while index < len(stripped) and matched < len(collapsed):
        if stripped[index].isalpha():
            matched += 1
        index += 1
    return stripped[index:].lstrip(" :.|-\t")


def _labeled_values(
    lines: list[str], labels: tuple[str, ...], *, collapsed: bool = False
) -> list[str]:
    """Collect values for `labels` in both packet layouts.

    Layout A (native form text): the label occupies its own line and the value
    is the next non-noise line. Layout B (OCR of the same page): the label and
    value share a line, separated by a colon. Labels are matched exactly first
    and fuzzily second, so damaged labels ('Sponser ID', 'Visa Cisse') no
    longer strand intact values.

    `collapsed` additionally matches labels whose interior spaces were dropped
    ('FeeStatus:paid'). That is a second-engine artifact, and enabling it on
    ordinary OCR lines measurably polluted open fields - a noise line starting
    with a label word yields a garbled candidate that vocabulary repair then
    turns into a wrong-but-valid value - so it is opt-in for the rapid pass.
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
            if remainder is None and collapsed:
                remainder = _collapsed_label_prefix(stripped, label)
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

# Identity is the one field where the intake form is not the best source. It
# is self-reported, and it is where this corpus puts its identity decoys: on
# the packets where page kinds disagree about the applicant, intake matches
# the label 37% of the time against 86% for the biometric slip and 81% for the
# registry extract. Ranking the identity documents above it is worth +10
# names against 1 lost over 400 packets.
#
# Only the tie-break order changes. Corroboration stays primary: reordering
# authority *and* promoting it above agreement scores +11/-6, and doing that
# with the general ranking is the change that once cost 61 names (+0/-16 when
# re-measured here).
_IDENTITY_RANK = {
    "manual": 6,
    "biometric": 5,
    "registry": 4,
    "sponsor": 3,
    "intake": 2,
    "fee": 2,
    "unknown": 0,
}


def _consensus(
    candidates: list[tuple[str, str]], parser, *, authority_first: bool = False,
    rank: dict[str, int] | None = None,
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
    ranking = rank or _PAGE_RANK
    parsed: list[tuple[int, str]] = []
    for value, kind in candidates:
        result = parser(value)
        if result:
            parsed.append((ranking.get(kind, 0), result))
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
    findings = [m.group(1) or m.group(2) for m in _FINDING_RE.finditer(full_text)]
    if findings:
        final = findings[-1].upper().replace(" ", "_").replace("__", "_")
        record.manual_finding = "NEEDS_REVIEW" if "REVIEW" in final else final
        if record.manual_finding == "APPROVED" and len(findings) > 1:
            record.has_approval_override = True

    # Candidates keep the page type they came from so precedence can be
    # applied; a bare list of strings cannot express that a value came from a
    # decoy fragment rather than the intake form.
    #
    # Second-engine lines feed only closed-vocabulary fields, where snapping
    # filters noise. Open-form fields (names, sponsor digits, dates) are
    # excluded: a garbled read there snaps into a wrong-but-valid value, which
    # measured as a doubling of wrong applicant names before this split.
    _CLOSED = {"species_code", "home_world", "visa_class", "declared_purpose",
               "fee_status", "risk_flags"}
    raw: dict[str, list[tuple[str, str]]] = {field: [] for field in _LABELS}
    rapid_lines: list[str] = []
    for page in packet.pages:
        page_lines = page.text.splitlines()
        page_rapid = page.rapid_text.splitlines() if page.rapid_text else []
        rapid_lines.extend(page_rapid)
        for field, labels in _LABELS.items():
            raw[field].extend(
                (value, page.kind) for value in _labeled_values(page_lines, labels)
            )
            if page_rapid and field in _CLOSED:
                raw[field].extend(
                    (value, page.kind)
                    for value in _labeled_values(page_rapid, labels, collapsed=True)
                )

    # Tokens where OCR lost the space between label and value are unreadable
    # by any per-field parser, so they are decoded before field resolution and
    # merged in as ordinary candidates from their own page.
    for page in packet.pages:
        for line in (page.text + "\n" + (page.rapid_text or "")).splitlines():
            for token in line.split():
                if len(token) < 8:
                    continue
                decoded = split_fused(token)
                if decoded:
                    raw[decoded[0]].append((decoded[1], page.kind))

    record.applicant_name = (
        _consensus(raw["applicant_name"], _parse_name, rank=_IDENTITY_RANK) or UNKNOWN
    )
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

    flags, flags_seen = _extract_flags(all_lines + rapid_lines, full_text)
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
    rapid_blob = "\n".join(rapid_lines)
    if match := _REGISTRY_STATUS_RE.search(full_text + "\n" + rapid_blob):
        record.registry_status = " ".join(match.group(1).split()).upper()
    if match := _RECEIPT_RE.search(full_text):
        record.receipt_date = match.group(1)

    # A field the document marks destroyed is not read, whatever OCR produced
    # for it. Extraction scores blank and wrong identically, but a wrong value
    # reaches the policy and a blank one does not.
    for field in _damaged_fields(full_text):
        setattr(record, field, UNKNOWN)
        if field == "risk_flags":
            record.risk_flags_known = False
    record.documented_damage = bool(_damaged_fields(full_text))

    # Prose evidence is applied after the damage blanking, and outranks it.
    # The two co-occur precisely because a correction is the remedy for the
    # destroyed field it names, so honouring the tag over the correction would
    # discard the fix: on the six packets where both cover the same field the
    # prose matches the label every time. This does not weaken the no-invented
    # -data rule - the value is written in the packet, not inferred from it -
    # and `documented_damage` above still records that the damage occurred.
    #
    # Read from the native text layer only, the source the 422-case agreement
    # was measured on. OCR of these same sentences is excluded: a garbled
    # clause still matches the pattern, and letting it overwrite a good form
    # reading would trade a measured gain for an unmeasured risk. Values still
    # pass their vocabulary snapper, so a malformed capture is dropped.
    annotated = "\n".join(page.visible_native for page in packet.pages)
    if match := _COMPLIANCE_CLASS_RE.search(annotated):
        if snapped := snap_visa(match.group(1)):
            record.visa_class = snapped
    # A correction is an amendment, so it is applied last and wins outright.
    if match := _CORRECTION_RE["visa_class"].search(annotated):
        if snapped := snap_visa(match.group(1)):
            record.visa_class = snapped
    if match := _CORRECTION_RE["fee_status"].search(annotated):
        if snapped := _parse_fee(match.group(1)):
            record.fee_status = snapped
            record.fee_explicit_unknown = snapped == "unknown"
    if match := _CORRECTION_RE["applicant_name"].search(annotated):
        if snapped := _parse_name(match.group(1)):
            record.applicant_name = snapped
    if match := _SPONSOR_CORRECTION_RE.search(annotated):
        if snapped := repair_sponsor_id(match.group(1)):
            record.sponsor_id = snapped

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
