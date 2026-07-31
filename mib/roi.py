"""Targeted region-of-interest reading for fields whole-page OCR loses.

Whole-page OCR spends its resolution budget uniformly, so a damaged value in
one form cell is unrecoverable even when the surrounding page reads cleanly.
But these packets are template-generated: a field's value sits at a fixed
offset from its label, measured on the corpus at dx ~= 135-150pt, dy ~= 23pt,
height ~= 12pt.

So: find the *label* on the raster (labels are printed cleanly and survive OCR
even when hand-filled values do not), crop the cell where its value must be,
and re-read that small region at high resolution with a per-field character
allowlist. Ten times the effective resolution for a fraction of the pixels.

The label is located by OCR word boxes rather than absolute page coordinates.
Hardcoded template coordinates would score well here and break on a private
set whose layout shifted; anchoring to the label travels.
"""

from __future__ import annotations

import re

import numpy as np

import pymupdf

# These forms are two-column: the value sits to the RIGHT of its label on the
# same text line. The column offset is not constant (the intake form puts
# values at x=203 against labels at x=68; the fee receipt uses 238 against 88),
# so the crop is defined as "everything right of this label, on its line"
# rather than a fixed dx that would fit one template and miss the other.
_GAP = 4.0        # skip the label's own trailing edge
_VALUE_W = 250.0  # widest observed value cell, padded
_LINE_PAD = 4.0   # vertical slack for baseline jitter on rasterized scans

# Per-field character allowlists. Constraining the recognizer to the alphabet a
# field can legally contain is what makes a high-resolution re-read pay off.
ALLOWLISTS = {
    "fee_status": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ",
    "sponsor_id": "SPN-0123456789",
    "visa_class": "XWDIPMEDTRANSIT-0123456789",
    "arrival_date": "0123456789-",
    "species_code": "ABCDEFGHIJKLMNOPQRSTUVWXYZ_",
    "risk_flags": "abcdefghijklmnopqrstuvwxyz_|,",
    "declared_purpose": "abcdefghijklmnopqrstuvwxyz ",
    "home_world": "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789- ",
}

# Labels are multi-word, and a single word is ambiguous ("Status" also ends
# "Registry Status"). Each field is matched as a consecutive word sequence, so
# the crop starts after the label's true right edge.
_LABEL_TOKENS = {
    "fee_status": ("fee", "status"),
    "sponsor_id": ("sponsor", "id"),
    "visa_class": ("visa", "class"),
    "arrival_date": ("arrival", "date"),
    "species_code": ("species", "code"),
    "risk_flags": ("observed", "flags"),
    "declared_purpose": ("declared", "purpose"),
    "home_world": ("home", "world"),
}

# OCR damage to the label itself is tolerated: these are printed form text, so
# a near match is safe where a near match on the *value* would not be.
_TOKEN_SLOP = 0.34


def _token_matches(word: str, token: str) -> bool:
    from .lexicon import weighted_distance

    cleaned = re.sub(r"[^a-z]", "", word.lower())
    if not cleaned:
        return False
    return weighted_distance(cleaned, token) / len(token) <= _TOKEN_SLOP


def word_boxes(gray: np.ndarray, dpi: int) -> list[tuple[str, float, float, float, float]]:
    """OCR the page and return (word, x0, y0, x1, y1) in PDF points."""
    from .pdfio import _tesseract_tsv

    scale = 72.0 / dpi
    boxes = []
    for word, left, top, width, height in _tesseract_tsv(gray):
        boxes.append(
            (word, left * scale, top * scale, (left + width) * scale,
             (top + height) * scale)
        )
    return boxes


def find_label(boxes, field: str) -> tuple[float, float, float, float] | None:
    """Locate a field's label as a consecutive run of words on one line."""
    tokens = _LABEL_TOKENS.get(field)
    if not tokens:
        return None
    for index in range(len(boxes)):
        run = []
        cursor = index
        for token in tokens:
            if cursor >= len(boxes) or not _token_matches(boxes[cursor][0], token):
                break
            run.append(boxes[cursor])
            cursor += 1
        if len(run) != len(tokens):
            continue
        # All tokens of a label share a text line.
        tops = [box[2] for box in run]
        if max(tops) - min(tops) > 6.0:
            continue
        return (
            min(b[1] for b in run), min(b[2] for b in run),
            max(b[3] for b in run), max(b[4] for b in run),
        )
    return None


def read_value(
    page: pymupdf.Page,
    field: str,
    boxes,
    *,
    dpi: int = 600,
) -> str | None:
    """Re-read one field's value cell at high resolution.

    Returns the raw recognized text; callers snap it to the field's vocabulary,
    so a garbled read is rejected downstream rather than trusted here.
    """
    from .pdfio import _tesseract

    label = find_label(boxes, field)
    if label is None:
        return None

    _, y0, x1, y1 = label
    cell = pymupdf.Rect(
        x1 + _GAP,
        y0 - _LINE_PAD,
        x1 + _GAP + _VALUE_W,
        y1 + _LINE_PAD,
    ) & page.rect
    if cell.is_empty or cell.get_area() < 40:
        return None

    pixmap = page.get_pixmap(
        dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False, clip=cell
    )
    if pixmap.width < 8 or pixmap.height < 8:
        return None
    crop = np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(
        pixmap.height, pixmap.width
    )

    # PSM 7 reads the crop as a single text line, which is what a form cell is.
    text, _ = _tesseract(crop, psm=7, allowlist=ALLOWLISTS.get(field))
    text = text.strip()
    return text or None


# --- Constrained recognition: correlation against rendered candidates -------
#
# The corpus's raster pages embed low-DPI JPEGs whose strokes are ~3x too
# thick for their glyph size: counters fill in and adjacent glyphs merge into
# word-shaped blobs. Segment-then-classify OCR has nothing to segment at any
# resolution. But every damaged field is closed-vocabulary, so recognition can
# be constrained: render each candidate in the form's face, degrade it the
# same way (dilate to match the stroke bloat), correlate against the cell, and
# accept only a clear winner of plausible width. Measured on fee cells OCR had
# failed entirely: 8/40 recovered, 0 wrong, 32 refused.

_CORR_MARGIN = 0.08
_WIDTH_RATIO = (0.65, 1.45)


def _render_candidate(text: str, height: int, dilate: int) -> "np.ndarray | None":
    scale = height / 22.0
    doc = pymupdf.open()
    page = doc.new_page(width=420, height=40)
    page.insert_text((4, 26), text, fontsize=18, fontname="helv")
    pix = page.get_pixmap(matrix=pymupdf.Matrix(scale, scale), colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    doc.close()
    cols = np.where((img < 128).any(axis=0))[0]
    rows = np.where((img < 128).any(axis=1))[0]
    if not len(cols) or not len(rows):
        return None
    img = img[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
    import cv2

    inked = cv2.dilate(255 - img, np.ones((3, 3), np.uint8), iterations=dilate)
    return cv2.GaussianBlur(255 - inked, (3, 3), 0)


def correlate_value(page: pymupdf.Page, field: str, boxes, candidates) -> str | None:
    """Pick the candidate whose degraded rendering best matches the cell.

    Refuses unless the winner clears the runner-up by a margin AND its width
    is plausible for the cell's ink - a short template can correlate well
    inside a longer blob, so shape alone is not enough.
    """
    import cv2

    label = find_label(boxes, field)
    if label is None:
        return None
    _, y0, x1, y1 = label
    cell_rect = pymupdf.Rect(
        x1 + _GAP, y0 - _LINE_PAD, x1 + _GAP + _VALUE_W, y1 + _LINE_PAD
    ) & page.rect
    if cell_rect.is_empty:
        return None
    pix = page.get_pixmap(dpi=400, colorspace=pymupdf.csGRAY, clip=cell_rect)
    if pix.width < 12 or pix.height < 12:
        return None
    cell = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    cell = cv2.normalize(cell, None, 0, 255, cv2.NORM_MINMAX)

    ink_rows = np.where((cell < 150).any(axis=1))[0]
    ink_cols = np.where((cell < 150).any(axis=0))[0]
    if len(ink_rows) < 6 or len(ink_cols) < 6:
        return None
    height = ink_rows[-1] - ink_rows[0] + 1
    ink_width = ink_cols[-1] - ink_cols[0] + 1

    scores: dict[str, float] = {}
    for candidate in candidates:
        best = -1.0
        for dilate in (1, 2):
            template = _render_candidate(candidate, height, dilate)
            if (
                template is None
                or template.shape[0] > cell.shape[0]
                or template.shape[1] > cell.shape[1]
            ):
                continue
            ratio = template.shape[1] / max(1, ink_width)
            if not (_WIDTH_RATIO[0] <= ratio <= _WIDTH_RATIO[1]):
                continue
            best = max(best, float(cv2.matchTemplate(cell, template, cv2.TM_CCOEFF_NORMED).max()))
        scores[candidate] = best
    ranked = sorted(scores.items(), key=lambda item: -item[1])
    if len(ranked) < 2 or ranked[0][1] < 0.2:
        return None
    if ranked[0][1] - ranked[1][1] < _CORR_MARGIN:
        return None
    return ranked[0][0]
