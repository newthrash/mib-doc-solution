"""MIB-OCR: a degradation-matched, vocabulary-constrained word recognizer.

Generic OCR solves the wrong problem for this corpus. Tesseract segments
characters before classifying and PP-OCR reads open text in unknown fonts -
but these packets are template-generated: the font is known, every damaged
field draws from a closed vocabulary, and the raster degradation is synthetic
and consistent within a page. That is the matched-filter regime from
communications: estimate the channel, then ask which of N known messages best
explains the received signal.

The channel estimate is the novel part, and it is self-supervised - with one
correction a page screenshot forced: the footer is NOT usable as known
plaintext, because it is stamped as crisp native text OVER the embedded
raster rather than degraded with it. The known plaintext that genuinely
shares the values' degradation is the form TITLE, large template text inside
the raster image itself. Fitting jointly over (title candidate x degradation
grid) therefore yields the page's channel and its form type in a single
step; candidate values are then rendered, pushed through the fitted channel,
and scored against the observed blob. Recognition never needs character
segmentation, which is exactly what the merged-glyph pages destroy.

Scoring combines normalized cross-correlation with a column-ink-profile
match, after a width prefilter. A candidate wins only with a clear margin;
otherwise the field stays unread. The engine's failure mode is silence, not
a confident wrong value - the property the rest of the pipeline is built on.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

import pymupdf

BOILERPLATE = "Synthetic hiring challenge document"

# Form titles: large known text inside the degraded raster - the genuine
# same-channel plaintext. The footer, by contrast, is native text stamped
# over the image and always crisp; fitting against it yields an identity
# channel and must not be done.
FORM_TITLES = (
    "MIB Fee Receipt",
    "Planetary Registry Extract",
    "FORM I-8090: Extraterrestrial Work Authorization Intake",
    "FORM B-13: Biometric Scan Slip",
    "Sponsor Attestation Letter",
    "Manual Adjudicator Note",
)

# Degradation grid searched against the footer. Small on purpose: the
# generator's damage is stroke thickening plus mild blur, and the footer fit
# only needs to be close, not perfect.
_DILATE_RADII = (0, 1, 2)
_BLUR_SIGMAS = (0.0, 0.8, 1.6)

_MIN_MARGIN = 0.055
_MIN_SCORE = 0.35
_WIDTH_RATIO = (0.60, 1.55)


def _render_word(text: str, target_height: int, fontsize: int = 22) -> np.ndarray | None:
    """Render `text` in the form face, cropped to ink, scaled to height."""
    import cv2

    width = int(len(text) * fontsize * 0.75) + 30
    doc = pymupdf.open()
    page = doc.new_page(width=width, height=fontsize * 2)
    page.insert_text((6, fontsize * 1.3), text, fontsize=fontsize, fontname="helv")
    pix = page.get_pixmap(colorspace=pymupdf.csGRAY, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    doc.close()
    ys, xs = np.where(img < 128)
    if not len(xs):
        return None
    img = img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    scale = target_height / max(1, img.shape[0])
    new_w = max(2, int(img.shape[1] * scale))
    return cv2.resize(img, (new_w, target_height), interpolation=cv2.INTER_AREA)


def degrade(img: np.ndarray, radius: int, sigma: float) -> np.ndarray:
    """Apply the fitted channel: thicken strokes, then blur."""
    import cv2

    out = img
    if radius > 0:
        kernel = np.ones((2 * radius + 1, 2 * radius + 1), np.uint8)
        out = 255 - cv2.dilate(255 - out, kernel)
    if sigma > 0:
        size = int(sigma * 4) | 1
        out = cv2.GaussianBlur(out, (size, size), sigma)
    return out


def _normalize(img: np.ndarray) -> np.ndarray:
    import cv2

    return cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)


def _ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Correlation of b slid over a (a wider or equal)."""
    import cv2

    if a.shape[0] < b.shape[0] or a.shape[1] < b.shape[1]:
        return -1.0
    return float(cv2.matchTemplate(a, b, cv2.TM_CCOEFF_NORMED).max())


def _column_profile(img: np.ndarray, bins: int = 48) -> np.ndarray:
    """Ink mass per column, resampled - survives merging that kills glyphs."""
    ink = (255.0 - img.astype(np.float32)).sum(axis=0)
    total = ink.sum()
    if total <= 0:
        return np.zeros(bins)
    ink /= total
    positions = np.linspace(0, len(ink) - 1e-6, bins + 1)
    return np.array([
        ink[int(positions[i]):max(int(positions[i]) + 1, int(positions[i + 1]))].sum()
        for i in range(bins)
    ])


def _profile_score(a: np.ndarray, b: np.ndarray) -> float:
    pa, pb = _column_profile(a), _column_profile(b)
    return 1.0 - float(np.abs(pa - pb).sum()) / 2.0


@dataclass
class Channel:
    radius: int
    sigma: float
    quality: float  # title-fit NCC; low quality disables the engine for the page
    form_title: str | None = None


def fit_channel(gray: np.ndarray, case_id: str, page_number: int) -> Channel | None:
    """Estimate the page's degradation from its form title.

    The title is the largest known text INSIDE the raster, so it shares the
    values' channel (the footer does not - it is crisp native overlay). The
    joint search over (title, radius, sigma) also identifies the form; a page
    whose best joint fit is weak gets no channel and the engine stays silent.
    """
    # Search the top third; the title is somewhere in the first few ink
    # bands, below whatever stamps or noise sit above it.
    strip = gray[: int(gray.shape[0] * 0.33), :]
    if (strip < 150).sum() < 80:
        return None

    best = Channel(0, 0.0, -1.0)
    best_title = None
    for top, bottom in find_lines(strip)[:4]:
        band = strip[top:bottom, :]
        cols = np.where((band < 150).any(axis=0))[0]
        if len(cols) < 20 or bottom - top < 8:
            continue
        observed = _normalize(band[:, cols[0]:cols[-1] + 1])
        height = observed.shape[0]
        for title in FORM_TITLES:
            reference = _render_word(title, height)
            if reference is None:
                continue
            for radius in _DILATE_RADII:
                for sigma in _BLUR_SIGMAS:
                    candidate = degrade(reference, radius, sigma)
                    score = _ncc(observed, _normalize(candidate)) if (
                        candidate.shape[1] <= observed.shape[1]
                    ) else _ncc(_normalize(candidate), observed)
                    if score > best.quality:
                        best = Channel(radius, sigma, score)
                        best_title = title
    best.form_title = best_title
    return best if best.quality >= 0.35 else None


def read_blob(
    cell: np.ndarray,
    candidates: tuple[str, ...],
    channel: Channel,
) -> str | None:
    """Which vocabulary word best explains this cell, under the fitted channel?

    Returns None unless one candidate wins with both an absolute-quality floor
    and a margin over the runner-up.
    """
    cell = _normalize(cell)
    ink_rows = np.where((cell < 150).any(axis=1))[0]
    ink_cols = np.where((cell < 150).any(axis=0))[0]
    if len(ink_rows) < 6 or len(ink_cols) < 8:
        return None
    blob = cell[ink_rows[0]:ink_rows[-1] + 1, ink_cols[0]:ink_cols[-1] + 1]
    height, width = blob.shape

    scored: list[tuple[float, str]] = []
    for text in candidates:
        rendered = _render_word(text, height)
        if rendered is None:
            continue
        shaped = degrade(rendered, channel.radius, channel.sigma)
        ratio = shaped.shape[1] / max(1, width)
        if not (_WIDTH_RATIO[0] <= ratio <= _WIDTH_RATIO[1]):
            continue
        ncc = _ncc(blob, _normalize(shaped)) if shaped.shape[1] <= width else _ncc(
            _normalize(shaped), blob
        )
        profile = _profile_score(blob, shaped)
        scored.append((0.55 * ncc + 0.45 * profile, text))

    if not scored:
        return None
    scored.sort(reverse=True)
    if scored[0][0] < _MIN_SCORE:
        return None
    if len(scored) > 1 and scored[0][0] - scored[1][0] < _MIN_MARGIN:
        return None
    return scored[0][1]


# --- Line-level reading without any localization dependency -----------------
#
# Word boxes from a segmenting engine fail on exactly the pages this engine
# exists for, so localization must not depend on them. Text LINES survive the
# degradation - merged glyphs still sit in bands separated by vertical
# whitespace - and within a band, label and value separate by a column gap.
# The matched filter identifies the label chunk itself; whatever follows it
# on the band is the value blob.

FIELD_LABELS = {
    "fee_status": "Fee Status",
    "species_code": "Species Code",
    "home_world": "Home World",
    "visa_class": "Visa Class",
    "declared_purpose": "Declared Purpose",
    "risk_flags": "Observed flags",
    "registry_status": "Registry Status",
}


def find_lines(gray: np.ndarray, min_height: int = 8) -> list[tuple[int, int]]:
    """Text-line bands via horizontal ink projection."""
    ink = (gray < 150).sum(axis=1)
    threshold = max(2, int(gray.shape[1] * 0.004))
    bands, start = [], None
    for row, count in enumerate(ink):
        if count >= threshold and start is None:
            start = row
        elif count < threshold and start is not None:
            if row - start >= min_height:
                bands.append((start, row))
            start = None
    if start is not None and len(ink) - start >= min_height:
        bands.append((start, len(ink)))
    return bands


def find_chunks(band: np.ndarray, gap_factor: float = 0.6) -> list[tuple[int, int]]:
    """Column-gap segmentation of a line band into ink chunks."""
    ink = (band < 150).sum(axis=0)
    min_gap = max(4, int(band.shape[0] * gap_factor))
    chunks, start, gap = [], None, 0
    for col, count in enumerate(ink):
        if count > 0:
            if start is None:
                start = col
            gap = 0
        else:
            if start is not None:
                gap += 1
                if gap >= min_gap:
                    chunks.append((start, col - gap + 1))
                    start, gap = None, 0
    if start is not None:
        chunks.append((start, len(ink)))
    return [(a, b) for a, b in chunks if b - a >= 6]


def read_page_fields(
    gray: np.ndarray,
    wanted: dict[str, tuple[str, ...]],
    channel: Channel,
) -> dict[str, str]:
    """Read wanted fields from a raster page, matched-filter end to end.

    For each line band: identify a label chunk by matching against the known
    label set (with margin); if it names a wanted field, read the next chunk
    against that field's vocabulary. A second attempt matches the whole band
    against rendered 'Label: value' composites, which covers inline layouts
    whose label-value gap is too small to chunk.
    """
    found: dict[str, str] = {}
    label_candidates = {f: FIELD_LABELS[f] for f in wanted if f in FIELD_LABELS}
    if not label_candidates:
        return found
    label_vocab = tuple(label_candidates.values())
    label_to_field = {v: k for k, v in label_candidates.items()}

    body_limit = int(gray.shape[0] * 0.88)  # exclude the footer
    for top, bottom in find_lines(gray):
        if bottom > body_limit or not label_candidates:
            continue
        band = gray[top:bottom, :]
        chunks = find_chunks(band)

        if len(chunks) >= 2:
            first = band[:, chunks[0][0]:chunks[0][1]]
            label_hit = read_blob(first, label_vocab, channel)
            if label_hit:
                field = label_to_field[label_hit]
                value = band[:, chunks[1][0]:chunks[-1][1]]
                verdict = read_blob(value, wanted[field], channel)
                if verdict and field not in found:
                    found[field] = verdict
                continue

        # Inline layout: match whole-band composites 'Label: value'.
        for field, label in list(label_candidates.items()):
            if field in found:
                continue
            composites = tuple(f"{label}: {v}" for v in wanted[field])
            verdict = read_blob(band[:, :], composites, channel)
            if verdict:
                found[field] = verdict.split(": ", 1)[1]
                break
    return found
