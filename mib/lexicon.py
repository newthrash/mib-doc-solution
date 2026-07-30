"""OCR-tolerant matching against the challenge's closed vocabularies.

Nearly every output field draws from a small fixed vocabulary, which turns
noisy extraction into classification: snap an OCR reading to the nearest legal
value using an edit distance that discounts common glyph confusions, and
refuse to snap when the nearest value is not clearly closest. Refusing matters
as much as matching - forcing a bad snap manufactures confident wrong fields.
"""

from __future__ import annotations

import re
from functools import lru_cache

from .constants import HOME_WORLDS, PURPOSES, SPECIES_CODES, VISA_CLASSES

# Substitution cost for glyph pairs OCR habitually confuses.
#
# tools/fit_confusions.py measures this corpus's real error model by aligning
# OCR of each page footer against the footer text we already know - no labels
# needed. It found the damage concentrated in one cluster of thin vertical
# glyphs (e, i, l, r), which explains the residual name errors. Adding the two
# pairs it surfaced that were absent here, E<->I and L<->R, was measured and
# did NOT help (out-of-fold 117.10 -> 117.09, one more wrong name), so it is
# not applied. The finding is kept; the change is not.
_CONFUSABLE = {
    ("0", "O"), ("0", "Q"), ("0", "D"), ("1", "I"), ("1", "L"), ("1", "T"),
    ("2", "Z"), ("5", "S"), ("6", "G"), ("8", "B"), ("7", "T"), ("4", "A"),
    ("C", "G"), ("E", "F"), ("I", "L"), ("K", "X"), ("M", "N"), ("O", "Q"),
    ("P", "R"), ("U", "V"), ("V", "Y"), ("W", "V"), ("H", "N"), ("R", "B"),
}
_CONFUSABLE_COST = 0.35
_CASE_COST = 0.0


def _sub_cost(a: str, b: str) -> float:
    if a == b:
        return 0.0
    if a.upper() == b.upper():
        return _CASE_COST
    if (a.upper(), b.upper()) in _CONFUSABLE or (b.upper(), a.upper()) in _CONFUSABLE:
        return _CONFUSABLE_COST
    return 1.0


def weighted_distance(a: str, b: str) -> float:
    """Damerau-Levenshtein with discounted confusable substitutions."""
    if a == b:
        return 0.0
    rows, cols = len(a) + 1, len(b) + 1
    previous2: list[float] = []
    previous = [float(j) for j in range(cols)]
    for i in range(1, rows):
        current = [float(i)] + [0.0] * (cols - 1)
        for j in range(1, cols):
            cost = _sub_cost(a[i - 1], b[j - 1])
            current[j] = min(
                previous[j] + 1.0,        # deletion
                current[j - 1] + 1.0,     # insertion
                previous[j - 1] + cost,   # substitution
            )
            if (
                i > 1
                and j > 1
                and a[i - 1].upper() == b[j - 2].upper()
                and a[i - 2].upper() == b[j - 1].upper()
            ):
                current[j] = min(current[j], previous2[j - 2] + 0.7)  # transposition
        previous2, previous = previous, current
    return previous[-1]


def _canon(value: str) -> str:
    return re.sub(r"[^A-Z0-9]+", " ", value.upper()).strip()


@lru_cache(maxsize=4096)
def snap(
    value: str,
    vocabulary: tuple[str, ...],
    max_relative: float = 0.34,
    min_margin: float = 0.15,
) -> str | None:
    """Return the vocabulary entry nearest to `value`, or None.

    A snap is accepted only when the best distance is within `max_relative`
    of the target length AND beats the runner-up by `min_margin` relative -
    similar vocabulary entries (XW-1 vs XW-2, species pairs) must stay
    unmatched rather than guessed.
    """
    cleaned = _canon(value)
    if not cleaned:
        return None
    scored = sorted(
        (weighted_distance(cleaned, _canon(entry)) / max(1, len(_canon(entry))), entry)
        for entry in vocabulary
    )
    best_score, best_entry = scored[0]
    if best_score > max_relative:
        return None
    if len(scored) > 1 and scored[1][0] - best_score < min_margin:
        return None
    return best_entry


def snap_species(value: str) -> str | None:
    return snap(value, SPECIES_CODES)


def snap_home_world(value: str) -> str | None:
    return snap(value, HOME_WORLDS)


# The suffix digit is nearly a primary key: 2, 3 and 7 each belong to exactly
# one visa class, so a damaged prefix costs nothing there. Only digit 1 is
# shared (XW-1 and DIP-1) and needs the letters to disambiguate.
_DIGIT_UNIQUE = {"2": "XW-2", "3": "MED-3", "7": "TRANSIT-7"}
_DIGIT_ONE = ("XW-1", "DIP-1")


def snap_visa(value: str) -> str | None:
    """Resolve a visa class, trusting the digit over the letters.

    OCR damages prefixes readily ('WED-3' for 'MED-3') while the digit, being
    a single well-separated glyph, survives. Exploiting that is safe precisely
    because the digit is the discriminating field: the classes it cannot
    separate - XW-1 from DIP-1 - still require their letters, and a packet
    whose letters are unreadable stays unmatched rather than guessed, since a
    wrong DIP-1 would grant a sponsor exemption that was never earned.
    """
    cleaned = _canon(value)
    exact = re.search(r"\b(XW|DIP|MED|TRANSIT)\s*[-. ]?\s*([1237])\b", cleaned)
    if exact:
        candidate = f"{exact.group(1)}-{exact.group(2)}"
        if candidate in VISA_CLASSES:
            return candidate

    loose = re.search(r"([A-Z]{2,8})\s*[-. ]?\s*([1237])\b", cleaned)
    if loose:
        prefix, digit = loose.group(1), loose.group(2)
        if digit in _DIGIT_UNIQUE:
            # Guard against a prefix that clearly belongs to another class,
            # which would mean the digit itself was misread.
            best = min(
                (weighted_distance(prefix, entry.split("-")[0]) / len(entry.split("-")[0]), entry)
                for entry in VISA_CLASSES
            )
            if best[1] == _DIGIT_UNIQUE[digit] or best[0] > 0.5:
                return _DIGIT_UNIQUE[digit]
            return None
        scored = sorted(
            (weighted_distance(prefix, entry.split("-")[0]) / len(entry.split("-")[0]), entry)
            for entry in _DIGIT_ONE
        )
        if scored[0][0] <= 0.34 and scored[1][0] - scored[0][0] >= 0.25:
            return scored[0][1]
        return None

    return snap(value, VISA_CLASSES, max_relative=0.25, min_margin=0.25)


def snap_purpose(value: str) -> str | None:
    return snap(value, PURPOSES)


_DIGIT_REPAIRS = str.maketrans(
    {"O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "L": "1",
     "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "G": "6", "T": "7"}
)


def repair_sponsor_id(value: str) -> str | None:
    """Normalize a sponsor mention to SPN-#### with constrained repair.

    The prefix is a fixed literal, so OCR damage to it is unambiguous and safe
    to repair (`SPR`, `SPM`, `SFN`, `5PN` all mean `SPN`). The four digits are
    NOT safe to guess beyond known glyph confusions - a wrong sponsor id can
    invent or erase a revocation - so anything still non-numeric is rejected.
    """
    compact = re.sub(r"[^A-Za-z0-9]", "", value.upper())
    match = re.search(r"[S5][PFR][NRM]([0-9OQDILZSBGT]{4})", compact)
    if not match:
        # Bare four digits directly after a SPONSOR label.
        match = re.search(r"^([0-9OQDILZSBGT]{4})$", compact)
        if not match:
            return None
    digits = match.group(1).translate(_DIGIT_REPAIRS)
    return f"SPN-{digits}" if digits.isdigit() else None


def snap_fee(value: str) -> str | None:
    """Match a damaged fee word to its status, asymmetrically.

    `paid` and `unpaid` differ only by a two-letter prefix, and confusing them
    flips a denial into an approval. So the negative prefix is decisive: any
    reading that begins with a plausible `un` resolves within {unpaid,
    unknown} and can never become `paid`. Only a token positively lacking that
    prefix is allowed to reach `paid`.
    """
    token = re.sub(r"[^a-z]", "", value.lower())
    if not token or len(token) < 3:
        return None

    if re.match(r"^[uvn][nmr]", token) or token.startswith("un"):
        rest = token[2:]
        if not rest:
            return None
        for candidate in ("paid", "known"):
            if weighted_distance(rest, candidate) / len(candidate) <= 0.5:
                return "unpaid" if candidate == "paid" else "unknown"
        return None

    for candidate in ("paid", "waived", "unknown"):
        if candidate == "unknown":
            continue
        # A short OCR fragment ("pac") is a prefix of the true word, so score
        # the truncation rather than penalizing the missing tail.
        window = candidate[: len(token)] if len(token) < len(candidate) else candidate
        if weighted_distance(token, window) / max(1, len(window)) <= 0.34:
            return candidate
    return None


_MONTHS = {
    "JAN": 1, "FEB": 2, "MAR": 3, "APR": 4, "MAY": 5, "JUN": 6,
    "JUL": 7, "AUG": 8, "SEP": 9, "OCT": 10, "NOV": 11, "DEC": 12,
}


def parse_date(value: str) -> str | None:
    """Parse ISO or `17 Apr 2026`-style dates to ISO, without guessing."""
    text = value.strip()
    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if match:
        year, month, day = map(int, match.groups())
    else:
        match = re.search(
            r"\b(\d{1,2})\s+([A-Za-z]{3})[A-Za-z]*\.?,?\s+(\d{4})\b", text
        )
        if match:
            day = int(match.group(1))
            month = _MONTHS.get(match.group(2).upper(), 0)
            year = int(match.group(3))
        else:
            match = re.search(
                r"\b([A-Za-z]{3})[A-Za-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\b", text
            )
            if not match:
                return None
            month = _MONTHS.get(match.group(1).upper(), 0)
            day = int(match.group(2))
            year = int(match.group(3))
    if not (1 <= month <= 12 and 1 <= day <= 31 and 1900 <= year <= 2100):
        return None
    import datetime as dt

    try:
        return dt.date(year, month, day).isoformat()
    except ValueError:
        return None


def snap_flag(value: str) -> str | None:
    """Match a possibly OCR-damaged token to a risk flag.

    Flags are long underscore-joined words, so a generous threshold is safe:
    the vocabulary is well separated and there is no `paid`/`unpaid` style
    near-collision to guard against.
    """
    from .constants import RISK_FLAGS

    token = re.sub(r"[^a-z_ ]", "", value.lower()).strip().replace(" ", "_")
    if len(token) < 5:
        return None
    best_score, best_flag = min(
        (weighted_distance(token, flag) / len(flag), flag) for flag in RISK_FLAGS
    )
    return best_flag if best_score <= 0.28 else None


# --- Applicant names -------------------------------------------------------

_NAMES_PATH = None
_NAME_VOCAB: tuple = ()


def load_names(path=None) -> tuple:
    """Load the applicant-name vocabulary: observed tokens plus grammar.

    Names are drawn from a small token set shared between first and last
    position, so a damaged reading can be snapped rather than emitted wrong.
    Generated stem+suffix combinations are included alongside the observed
    tokens so a private set that pairs known parts in new ways is still
    covered.
    """
    global _NAME_VOCAB, _NAMES_PATH
    from pathlib import Path as _Path

    if path is None:
        path = _Path(__file__).resolve().parent.parent / "policy" / "names.json"
    if _NAME_VOCAB and _NAMES_PATH == str(path):
        return _NAME_VOCAB
    try:
        import json

        with open(path) as handle:
            payload = json.load(handle)
        # Observed tokens are kept separate from generated ones and matched
        # first. Mixing them lets a spurious stem+suffix combination sit beside
        # a real name and block the margin test, so a damaged reading of a
        # token we have actually seen would fail to resolve.
        _NAME_VOCAB = (
            tuple(sorted(payload.get("observed_tokens", ()))),
            tuple(sorted(set(payload.get("generated", ())) - set(payload.get("observed_tokens", ())))),
        )
        _NAMES_PATH = str(path)
    except (OSError, ValueError):
        _NAME_VOCAB = ((), ())
    return _NAME_VOCAB


def snap_name_token(value: str) -> str | None:
    """Snap one name token to the vocabulary, refusing ambiguous matches."""
    observed, generated = load_names()
    if not observed:
        return None
    cleaned = re.sub(r"[^A-Za-z]", "", value)
    if len(cleaned) < 3:
        return None
    for vocab in (observed, generated):
        if not vocab:
            continue
        scored = sorted(
            (weighted_distance(cleaned.lower(), entry.lower()) / max(1, len(entry)), entry)
            for entry in vocab
        )
        best, entry = scored[0]
        if best > 0.30:
            continue
        if len(scored) > 1 and scored[1][0] - best < 0.08:
            continue
        return entry
    return None


def snap_name(value: str) -> str | None:
    """Repair a two-token applicant name, preserving its original casing."""
    parts = [p for p in value.split() if re.fullmatch(r"[A-Za-z'’-]+", p)]
    if len(parts) < 2:
        return None
    repaired = []
    for part in parts[:2]:
        snapped = snap_name_token(part)
        repaired.append(snapped if snapped else part)
    return " ".join(repaired)
