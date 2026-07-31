"""Undo horizontal strip displacement in scanned pages.

These rasters behave as though the page were cut into horizontal strips and
reassembled with the strips in correct vertical order but each slid sideways
by a different amount. Text lines are therefore broken mid-word wherever a
strip boundary crosses them, and the offsets show up as sharp discontinuities
in the leftmost-ink position: steady at x=82 for eighty rows, then 25, then
138.

That also explains the "vertical bar" artifacts that resisted every filter -
several are not marks on the page at all, but the exposed edges where one
strip ends and the next begins at a different offset.

Recovery is a registration problem. For each pair of adjacent rows, the
horizontal shift that best aligns them is found by cross-correlation; rows
where that shift is small belong to the same strip, and a large shift marks a
boundary. Each strip's displacement accumulates down the page, so shifting
every strip back by its cumulative offset restores the original alignment.

The estimate is only trustworthy where there is ink to correlate. Blank or
near-blank rows carry no signal, so a strip without enough ink inherits its
neighbour's offset rather than inventing one.
"""

from __future__ import annotations

import numpy as np

# Rows closer than this in estimated shift belong to the same strip.
_SAME_STRIP_TOLERANCE = 3
# Displacements beyond this are treated as correlation failures, not strips.
_MAX_SHIFT = 80
# A strip needs this much ink before its shift estimate is believed.
_MIN_INK_COLUMNS = 12
# Strips thinner than this are noise, and get merged into their neighbour.
_MIN_STRIP_ROWS = 4


def _row_shift(above: np.ndarray, below: np.ndarray, limit: int = _MAX_SHIFT) -> int | None:
    """Horizontal displacement aligning `below` to `above`, or None."""
    if above.max() < 6 or below.max() < 6:
        return None
    a = above - above.mean()
    b = below - below.mean()
    denominator = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if denominator < 1e-6:
        return None
    best_shift, best_score = 0, -2.0
    for shift in range(-limit, limit + 1):
        score = float((a * np.roll(b, shift)).sum()) / denominator
        if score > best_score:
            best_shift, best_score = shift, score
    # A confident alignment; otherwise the rows simply differ in content.
    return best_shift if best_score > 0.30 else None


def find_strips(gray: np.ndarray, step: int = 2) -> list[tuple[int, int, int]]:
    """Return (top, bottom, cumulative_shift) for each detected strip."""
    profile = (255.0 - gray.astype(np.float32))
    height = gray.shape[0]

    boundaries = [0]
    offsets = [0]
    running = 0
    for y in range(step, height, step):
        shift = _row_shift(profile[y - step], profile[y])
        if shift is None or abs(shift) <= _SAME_STRIP_TOLERANCE:
            continue
        running += shift
        boundaries.append(y)
        offsets.append(running)
    boundaries.append(height)

    strips: list[tuple[int, int, int]] = []
    for index in range(len(boundaries) - 1):
        top, bottom = boundaries[index], boundaries[index + 1]
        if bottom - top < _MIN_STRIP_ROWS and strips:
            # Too thin to register on its own; extend the previous strip.
            previous = strips[-1]
            strips[-1] = (previous[0], bottom, previous[2])
            continue
        strips.append((top, bottom, offsets[index]))
    return strips


def deshred(gray: np.ndarray, step: int = 2) -> tuple[np.ndarray, int]:
    """Realign horizontally displaced strips. Returns (image, strips_moved)."""
    strips = find_strips(gray, step=step)
    if len(strips) < 2:
        return gray, 0

    output = np.full_like(gray, 255)
    moved = 0
    for top, bottom, offset in strips:
        band = gray[top:bottom]
        ink_columns = int((band < 200).any(axis=0).sum())
        if ink_columns < _MIN_INK_COLUMNS or offset == 0:
            output[top:bottom] = band
            continue
        output[top:bottom] = np.roll(band, -offset, axis=1)
        # Rolling wraps; blank the wrapped margin so it cannot forge ink.
        if offset > 0:
            output[top:bottom, -offset:] = 255
        else:
            output[top:bottom, :(-offset)] = 255
        moved += 1
    return output, moved


def deshred_page(document, page, dpi: int = 300) -> tuple[np.ndarray, int]:
    """Load a page's bitmap (embedded when available) and realign it."""
    import cv2
    import pymupdf

    best = None
    for info in page.get_images(full=True):
        try:
            payload = document.extract_image(info[0])
        except Exception:  # noqa: BLE001
            continue
        decoded = cv2.imdecode(
            np.frombuffer(payload["image"], np.uint8), cv2.IMREAD_GRAYSCALE
        )
        if decoded is not None and (best is None or decoded.size > best.size):
            best = decoded
    if best is None:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        best = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return deshred(best)
