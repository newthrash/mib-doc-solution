"""Undo horizontal strip displacement in scanned pages.

Part of this corpus really was cut into horizontal bands and slid sideways.
The first attempt (mib/deshred.py) concluded otherwise; that conclusion was
wrong, and so was the reasoning behind it:

  * Its estimator correlated rows with ``np.roll``, which is *circular*:
    content leaving one edge reappears at the other and matches spuriously,
    so real displacements scored no better than wrap-around noise.
  * It compared single pixel rows at a 0.30 confidence threshold, low enough
    that unrelated text lines registered as displacement.
  * Its disproof rested on full-width horizontal rules being continuous in
    the original image. They are - but each rule is only ~6px tall and sits
    entirely inside one strip, so it cannot reveal displacement *between*
    strips. The check was incapable of detecting the effect it ruled out.

What settles it is a structure tall enough to span a boundary. On MIB-000006
page 2 the "SAMPLE DENIAL" watermark is cut across rows 704-708 with the
lower half displaced -88px at 0.92 correlation; realigning makes the
watermark, the ruled grid, the stamps and the portrait photograph all
coherent at once. A photograph does not reassemble by coincidence.

**Estimator: absolute, not accumulated.** An earlier version of this module
measured each strip's shift *relative* to the one above and summed them, so a
single bad boundary corrupted every row below it. Fitting one robust line
(cv2.fitLine with DIST_L1, resistant to the staggers themselves) to the page's
left margin and snapping every row to it gives each row an absolute target
with no error accumulation - and handles page skew for free, since the fitted
line can tilt. Measured over 152 pages where OCR already failed: the relative
estimator recovered +15 form anchors, this one +43, at a fifth of the cost.

**Apply only where OCR already failed.** Realignment rescues pages that read
as nothing (best cases 0 -> 68 and 0 -> 152 body characters) and destroys
pages that read fine (worst case 204 -> 0), because a page whose margin is
genuinely ragged gets sheared. Callers must gate on recovery failure and keep
the result only if it scores better - see ocr_page in pdfio.py.
"""
from __future__ import annotations

import numpy as np

MIN_JUMP = 5
MIN_BOUNDARIES = 3
MAX_JUMP = 150
INK_THRESHOLD = 120


def _binarize(gray: np.ndarray) -> np.ndarray:
    return (gray < INK_THRESHOLD).astype(np.uint8)


def _left_margins(ink: np.ndarray) -> list[tuple[int, int]]:
    """(row, leftmost ink column) for every row carrying ink."""
    out = []
    for y in range(ink.shape[0]):
        nz = np.flatnonzero(ink[y])
        if nz.size:
            out.append((y, int(nz[0])))
    return out


def looks_shredded(gray: np.ndarray, min_jump: int = MIN_JUMP,
                   min_boundaries: int = MIN_BOUNDARIES) -> tuple[bool, int]:
    """Count sharp margin steps between *vertically adjacent* rows.

    Adjacency matters: comparing across blank gaps counts ordinary layout
    changes as displacement. Counting rare large steps matters too - a
    percentile cannot work here, because boundaries are only ~2% of row
    transitions and any percentile at or below p97 discards exactly them.
    """
    margins = _left_margins(_binarize(gray))
    if len(margins) < 50:
        return False, 0
    jumps = 0
    for i in range(1, len(margins)):
        (y_prev, x_prev), (y_cur, x_cur) = margins[i - 1], margins[i]
        if y_cur == y_prev + 1 and min_jump < abs(x_cur - x_prev) < MAX_JUMP:
            jumps += 1
    return jumps >= min_boundaries, jumps


def deshred(gray: np.ndarray) -> tuple[np.ndarray, int]:
    """Snap every row's left margin onto one robust line. Returns (image, rows moved)."""
    import cv2

    ink = _binarize(gray)
    margins = _left_margins(ink)
    if len(margins) < 10:
        return gray, 0

    points = np.array([(x, y) for y, x in margins], dtype=np.float32)
    vx, vy, x0, y0 = [float(v[0]) for v in
                      cv2.fitLine(points, cv2.DIST_L1, 0, 0.01, 0.01)]
    if abs(vy) < 1e-6:
        return gray, 0

    out = np.full_like(gray, 255)
    moved = 0
    width = gray.shape[1]
    for y in range(gray.shape[0]):
        nz = np.flatnonzero(ink[y])
        if nz.size == 0:
            out[y] = gray[y]
            continue
        shift = int(round((x0 + (y - y0) * (vx / vy)) - nz[0]))
        if shift == 0:
            out[y] = gray[y]
            continue
        if abs(shift) >= width:
            continue  # nothing of the row would survive the move
        row = np.roll(gray[y], shift)
        if shift > 0:
            row[:shift] = 255
        else:
            row[shift:] = 255
        out[y] = row
        moved += 1
    return out, moved


def deshred_page(document, page, dpi: int = 300) -> tuple[np.ndarray, int]:
    """Realign a page, preferring its embedded bitmap over a re-render."""
    import cv2
    import pymupdf

    best = None
    for info in page.get_images(full=True):
        try:
            payload = document.extract_image(info[0])
        except Exception:  # noqa: BLE001
            continue
        decoded = cv2.imdecode(
            np.frombuffer(payload["image"], np.uint8), cv2.IMREAD_GRAYSCALE)
        if decoded is not None and (best is None or decoded.size > best.size):
            best = decoded
    if best is None:
        pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False)
        best = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)
    return deshred(best)
