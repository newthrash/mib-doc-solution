"""Ordered restoration of damaged scan pages, before recognition.

The order is load-bearing and was arrived at by measurement, not preference:

1. Isolate dark ink. This is not preparation for OCR - it is preparation for
   step 3. Margin scanning anchors on the leftmost dark pixel, and on an
   unthresholded page that pixel is as likely to be a watermark as the page
   frame. Cleaning first is what makes the geometry step trustworthy.
2. Correct 90-degree orientation, because row and column scanning assume the
   page is upright. Fires on roughly one page in forty here, but a sideways
   page would be destroyed by step 3 rather than merely unhelped.
3. Realign horizontally displaced rows. These scans are staggered: each row's
   leftmost dark pixel is registered to a fixed margin, absolutely rather than
   relative to its neighbour, so errors cannot accumulate down the page.
4. Enhance contrast and scale for the recognizer.

Measured on the pages where a missing field's only copy is damaged: 4,807
body characters against 2,985 for a plain render, a 61% gain. Shredding is
detected on 38 of 40 such pages, which is why the geometry step matters more
here than any amount of filtering.

Nothing in this module invents pixels. Where data has been physically wiped,
cropped, or redacted, restoration cannot recover it and the field stays
unread - the packet is routed to review on the evidence, not on a guess.
"""

from __future__ import annotations

import re
import subprocess

import numpy as np

import pymupdf

# Ink cut for the geometry stage. Dark enough to drop watermarks and gray
# furniture, permissive enough to keep the page frame that anchors alignment.
_INK_CUT = 150
# Margin standard deviation above which a page is treated as staggered.
_SHRED_STD = 25.0
_TARGET_MARGIN = 100
_MAX_SHIFT_FRAC = 0.25


def _needs_osd(binary: np.ndarray) -> bool:
    """Cheap pre-check before paying for orientation detection.

    OSD is a full recogniser subprocess and fires on roughly one page in
    forty, so running it unconditionally costs far more than it recovers. A
    sideways page has its ink strongly biased toward vertical runs, which is
    detectable from projection profiles in microseconds.
    """
    inked = binary < 255
    rows = inked.sum(axis=1).astype(float)
    columns = inked.sum(axis=0).astype(float)
    if rows.sum() < 200:
        return False
    # Upright text yields spiky row profile (lines) and flat column profile.
    row_spread = rows.std() / max(1.0, rows.mean())
    column_spread = columns.std() / max(1.0, columns.mean())
    return column_spread > row_spread * 1.4


def _osd_rotation(image: np.ndarray) -> int:
    """Tesseract orientation detection; 0 when it cannot tell."""
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return 0
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "--psm", "0"],
            input=encoded.tobytes(), stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL, timeout=20,
        )
    except subprocess.SubprocessError:
        return 0
    match = re.search(r"(?<=Rotate: )\d+", result.stdout.decode("utf-8", "replace"))
    return int(match.group(0)) if match else 0


def clean_ink(rgb: np.ndarray) -> np.ndarray:
    """Step 1: isolate dark ink so the geometry stage has a true anchor."""
    import cv2

    # Channel minimum darkens coloured stamps rather than averaging them
    # toward white, so a red overlay cannot masquerade as background.
    gray = np.min(rgb, axis=2) if rgb.ndim == 3 else rgb
    return cv2.threshold(gray, _INK_CUT, 255, cv2.THRESH_BINARY)[1]


def upright(image: np.ndarray) -> tuple[np.ndarray, int]:
    """Step 2: correct 90-degree orientation before any row scanning."""
    import cv2

    if not _needs_osd(image):
        return image, 0
    angle = _osd_rotation(image)
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE), angle
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180), angle
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE), angle
    return image, 0


def margin_profile(binary: np.ndarray) -> np.ndarray:
    """Leftmost dark pixel per inked row, vectorised.

    The obvious per-row loop costs seconds per page at scoring resolution;
    argmax over the whole array costs milliseconds.
    """
    inked = binary < 255
    has_ink = inked.any(axis=1)
    if not has_ink.any():
        return np.empty(0, dtype=int)
    return np.argmax(inked, axis=1)[has_ink]


def is_staggered(binary: np.ndarray) -> bool:
    profile = margin_profile(binary)
    return profile.size > 20 and float(profile.std()) > _SHRED_STD


def realign_rows(binary: np.ndarray) -> np.ndarray:
    """Step 3: register every row's left edge to a fixed margin.

    Absolute registration, deliberately: an earlier version aligned each row
    to its predecessor and accumulated the errors into a hundred pixels of
    drift down the page. Referencing one fixed target instead means a bad row
    cannot corrupt the rows below it.
    """
    height, width = binary.shape
    limit = int(width * _MAX_SHIFT_FRAC)
    inked = binary < 255
    has_ink = inked.any(axis=1)
    shifts = _TARGET_MARGIN - np.argmax(inked, axis=1)
    shifts[~has_ink] = 0
    shifts[np.abs(shifts) > limit] = 0   # implausible; leave the row alone

    # Gather with a shifted column index instead of rolling row by row:
    # one fancy-index over the array rather than a Python loop per row.
    columns = np.arange(width)[None, :] - shifts[:, None]
    valid = (columns >= 0) & (columns < width)
    output = np.full_like(binary, 255)
    rows = np.arange(height)[:, None]
    output[valid] = binary[rows.repeat(width, axis=1)[valid], columns[valid]]
    return output


def enhance(binary: np.ndarray, scale: float = 2.0) -> np.ndarray:
    """Step 4: contrast and scale, for the recognizer rather than the geometry.

    Scale measured, not assumed: 2.0 recovers more text than 3.0 (1,418 vs
    1,311 characters) at less than half the cost (0.70s vs 1.54s per page).
    Beyond roughly twice native, interpolation invents edges the recogniser
    then has to explain.
    """
    import cv2

    image = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(binary)
    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    image = cv2.bilateralFilter(image, 5, 40, 40)
    binarized = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 15
    )
    inverted = 255 - binarized
    horizontal = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(40, binarized.shape[1] // 15), 1)),
    )
    vertical = cv2.morphologyEx(
        inverted, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(40, binarized.shape[0] // 15))),
    )
    inverted = cv2.subtract(inverted, cv2.bitwise_or(horizontal, vertical))
    inverted = cv2.morphologyEx(inverted, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return 255 - inverted


def restore(rgb: np.ndarray) -> tuple[np.ndarray, dict]:
    """Run the ordered pipeline. Returns (image, what happened)."""
    report = {"rotated": 0, "staggered": False}
    binary = clean_ink(rgb)
    binary, report["rotated"] = upright(binary)
    if is_staggered(binary):
        report["staggered"] = True
        binary = realign_rows(binary)
    return enhance(binary), report


def restore_page(page: pymupdf.Page, dpi: int = 200) -> tuple[np.ndarray, dict]:
    pix = page.get_pixmap(dpi=dpi, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)
    return restore(rgb)


# --- Physical data loss -----------------------------------------------------
#
# NOT WIRED IN - a real signal that does not convert into a better decision.
#
# The premise held. Gap count tracks unrecoverable fields monotonically across
# 106 packets: 0 gaps averages 0.67 missing fields, 60+ averages 2.68. But
# giving it to the resolver lowered the out-of-fold score, and did so
# consistently - 120.14 with neither damage feature, 120.02 with the gap
# count, 120.04 with the damage flag, 119.96 with both.
#
# The information was already present. Gap count correlates +0.61 with
# has_scanned_pages and +0.38 with missing_count, both of which the resolver
# reads exactly; the contour count is a noisier restatement of features it
# already has, so it contributes variance and no signal. Kept as a measured
# negative result alongside the other image-recovery attempts.

_GAP_MAX_WIDTH = 100
_GAP_MIN_HEIGHT = 10
_GAP_MAX_HEIGHT = 50


def truncation_gaps(gray: np.ndarray) -> int:
    """Count text blocks that end abruptly, the signature of physical loss.

    Dilating along the text direction joins characters into line blobs; a
    blob far shorter than a line is a line that was cut off.
    """
    import cv2

    _, binary = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)
    dilated = cv2.dilate(
        binary, cv2.getStructuringElement(cv2.MORPH_RECT, (25, 5)), iterations=2
    )
    contours, _ = cv2.findContours(
        dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    gaps = 0
    for contour in contours:
        _, _, width, height = cv2.boundingRect(contour)
        if width < _GAP_MAX_WIDTH and _GAP_MIN_HEIGHT < height < _GAP_MAX_HEIGHT:
            gaps += 1
    return gaps
