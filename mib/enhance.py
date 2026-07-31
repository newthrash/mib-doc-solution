"""Prepare a degraded scan page for OCR.

Built against the pages the pipeline actually fails on, and shaped by what
they turned out to contain. A biometric slip like MIB-000006 page 2 carries
`Observed flags: planetary_embargo` - a disqualifying flag that decides the
case - in ink that is nearly black, sitting among artifacts that are merely
gray. Whole-page OCR returns only the footer.

MEASURED SCOPE - read before using. Across 45 raster pages this pipeline
recovers 1,723 body characters against 3,310 for a plain 300 DPI render: 11
pages better, 28 worse. It is NOT a general replacement for rendering. It wins
on pages whose text is near-black against gray furniture, and loses on faint
pages, where the darkness cut that isolates ink on a high-contrast page
discards the text itself. Faint pages are roughly three quarters of the
failures in this corpus, so this must be applied selectively - see
`suits_page` - or it will cost more than it recovers.

The order matters, and each step earns its place by measurement:

1. Read the embedded bitmap when there is one. Rendering the page resamples a
   144 DPI JPEG at 300 DPI; working from the source avoids one resample.
   Native text is composited back separately by the caller, since extracting
   the bitmap alone loses it.
2. Flatten the background. A large morphological opening estimates the page's
   illumination - smudges, soft blobs, uneven toner - and dividing it out
   removes them without touching stroke-sized detail.
3. Separate ink from artifact by darkness. This corpus's field text is far
   darker than its gray furniture, so a percentile-anchored cut keeps the
   text and discards rules, bars and stamps in one step. This is the single
   most effective operation on these pages.
4. Correct stroke weight in whichever direction the page needs, measured from
   its own ink rather than assumed: thin the merged glyphs, thicken the faded
   ones. Applying the wrong one is worse than applying neither.
5. Upscale before recognition, never after. Tesseract's LSTM wants roughly
   30px of x-height; these pages offer about 10.
6. Deskew, then erase long rules from the binary image where they are
   unambiguous.
"""

from __future__ import annotations

import numpy as np

import pymupdf

# Ink darker than this fraction of the page's dynamic range is text; the rest
# is furniture. Measured on failing pages: field text bottoms out near 0-50
# while rules, bars and watermarks sit above 150.
_INK_PERCENTILE = 2.0
_INK_HEADROOM = 60

# Stroke ratio = ink area over erosion boundary. Merged glyphs sit above the
# upper bound, broken faded ones below the lower.
_STROKE_MERGED = 3.1
_STROKE_BROKEN = 2.2

_TARGET_XHEIGHT = 30.0
_MAX_SCALE = 4.0
_MAX_SKEW_DEG = 8.0
_MIN_SKEW_DEG = 0.25


def _embedded_bitmap(document: pymupdf.Document, page: pymupdf.Page):
    """The page's source bitmap, or None when the page is vector-only."""
    import cv2

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
    return best


def _flatten_background(gray: np.ndarray) -> np.ndarray:
    """Divide out slow illumination: smudges, blobs, uneven toner."""
    import cv2

    span = max(15, (min(gray.shape) // 20) | 1)
    background = cv2.morphologyEx(
        gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (span, span))
    )
    background = cv2.GaussianBlur(background, (0, 0), span / 3.0)
    flattened = cv2.divide(gray, background, scale=255)
    return np.clip(flattened, 0, 255).astype(np.uint8)


def _ink_mask(gray: np.ndarray) -> np.ndarray:
    """Keep genuinely dark ink; drop gray furniture. Returns ink-as-white."""
    import cv2

    darkest = float(np.percentile(gray, _INK_PERCENTILE))
    cut = min(200.0, darkest + _INK_HEADROOM)
    mask = (gray <= cut).astype(np.uint8) * 255
    # A single stray pixel is noise, not a stroke.
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))


def _stroke_ratio(mask: np.ndarray) -> float:
    import cv2

    binary = (mask > 0).astype(np.uint8)
    total = int(binary.sum())
    if total < 40:
        return 0.0
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8))
    boundary = total - int(eroded.sum())
    return total / max(1, boundary)


def _xheight(mask: np.ndarray) -> float:
    """Median height of glyph-like components, for scale selection."""
    import cv2

    count, _, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    heights = [
        stats[i, cv2.CC_STAT_HEIGHT]
        for i in range(1, count)
        if 2 < stats[i, cv2.CC_STAT_HEIGHT] < mask.shape[0] * 0.05
        and stats[i, cv2.CC_STAT_AREA] > 4
    ]
    if len(heights) < 8:
        return 0.0
    return float(np.median(heights))


def _deskew(mask: np.ndarray) -> np.ndarray:
    import cv2

    coords = np.column_stack(np.where(mask > 0))
    if len(coords) < 150:
        return mask
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    if angle > 45:
        angle -= 90
    if not (_MIN_SKEW_DEG <= abs(angle) <= _MAX_SKEW_DEG):
        return mask
    matrix = cv2.getRotationMatrix2D((mask.shape[1] / 2, mask.shape[0] / 2), angle, 1.0)
    return cv2.warpAffine(
        mask, matrix, (mask.shape[1], mask.shape[0]),
        flags=cv2.INTER_NEAREST, borderValue=0,
    )


def _drop_bars(mask: np.ndarray) -> np.ndarray:
    """Remove scan-bar artifacts: components far taller than the page's text.

    These pages are littered with short dark vertical bars that survive rule
    removal because they are too short to open with a long kernel, and they
    dominate OCR output as pipes. A glyph is bounded by the page's own median
    glyph height, so anything several times taller is furniture - including
    the tall thin bars, without touching an 'I' or 'l' that sits at text
    height.
    """
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    if count < 3:
        return mask
    # Measure the glyph height from plausible glyphs only. Including the bars
    # in the median lets them define "normal" and immunises them from removal,
    # which is why an earlier version left every pipe in place.
    plausible = [
        stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, count)
        if stats[i, cv2.CC_STAT_HEIGHT] < mask.shape[0] * 0.04
        and stats[i, cv2.CC_STAT_AREA] > 4
    ]
    if len(plausible) < 6:
        return mask
    median_h = float(np.median(plausible))
    if median_h <= 0:
        return mask

    keep = np.zeros_like(mask)
    for index in range(1, count):
        h = stats[index, cv2.CC_STAT_HEIGHT]
        w = stats[index, cv2.CC_STAT_WIDTH]
        area = stats[index, cv2.CC_STAT_AREA]
        if h > median_h * 2.6 and w <= max(6, median_h * 0.9):
            continue  # tall thin bar
        if area < 3:
            continue
        keep[labels == index] = 255
    return keep


def _crop_to_text(mask: np.ndarray) -> np.ndarray:
    """Crop to the densest block of glyph-sized components.

    Scale selection must be driven by field text, not by the large stamp
    lettering elsewhere on the page - measuring x-height across both leaves
    the fields too small to recognise.
    """
    import cv2

    count, _, stats, centroids = cv2.connectedComponentsWithStats(
        (mask > 0).astype(np.uint8), 8
    )
    if count < 8:
        return mask
    heights = [stats[i, cv2.CC_STAT_HEIGHT] for i in range(1, count)]
    median_h = float(np.median(heights))
    glyphs = [
        (stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP],
         stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        for i in range(1, count)
        if 0.5 * median_h <= stats[i, cv2.CC_STAT_HEIGHT] <= 2.2 * median_h
    ]
    if len(glyphs) < 8:
        return mask
    tops = np.array([g[1] for g in glyphs])
    hist, edges = np.histogram(tops, bins=16, range=(0, mask.shape[0]))
    peak = int(np.argmax(hist))
    low, high = edges[max(0, peak - 1)], edges[min(len(edges) - 1, peak + 3)]
    band = [g for g in glyphs if low - median_h <= g[1] <= high + median_h]
    if len(band) < 6:
        return mask
    x0 = max(0, min(g[0] for g in band) - int(median_h))
    x1 = min(mask.shape[1], max(g[0] + g[2] for g in band) + int(median_h * 3))
    y0 = max(0, min(g[1] for g in band) - int(median_h))
    y1 = min(mask.shape[0], max(g[1] + g[3] for g in band) + int(median_h))
    if (y1 - y0) < 10 or (x1 - x0) < 40:
        return mask
    return mask[y0:y1, x0:x1]


def _erase_rules(mask: np.ndarray) -> np.ndarray:
    import cv2

    long_h = max(45, mask.shape[1] // 10)
    long_v = max(45, mask.shape[0] // 10)
    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (long_h, 1))
    )
    vertical = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, long_v))
    )
    cleaned = cv2.subtract(mask, cv2.bitwise_or(horizontal, vertical))
    return cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))


def suits_page(document: pymupdf.Document, page: pymupdf.Page) -> bool:
    """Whether this page has the profile enhance_for_ocr actually helps.

    The separation this pipeline depends on is text much darker than the
    furniture around it. Measure that directly rather than assuming it: if the
    page's ink is broadly gray, the darkness cut would take the text with the
    artifacts.
    """
    gray = _embedded_bitmap(document, page)
    if gray is None:
        return False
    ink = gray[gray < 200]
    if ink.size < 200:
        return False
    darkest = float(np.percentile(gray, _INK_PERCENTILE))
    # High-contrast page: a solidly black floor with gray furniture above it.
    return darkest < 60 and float(ink.mean()) > 90


def enhance_for_ocr(
    document: pymupdf.Document, page: pymupdf.Page, fallback_dpi: int = 300
) -> np.ndarray:
    """Return a black-on-white image of this page prepared for recognition."""
    import cv2

    gray = _embedded_bitmap(document, page)
    if gray is None:
        pix = page.get_pixmap(dpi=fallback_dpi, colorspace=pymupdf.csGRAY, alpha=False)
        gray = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)

    gray = _flatten_background(gray)
    mask = _ink_mask(gray)

    # Correct stroke weight in the direction this page needs.
    ratio = _stroke_ratio(mask)
    if ratio > _STROKE_MERGED:
        mask = cv2.erode(mask, np.ones((2, 2), np.uint8))
    elif 0.0 < ratio < _STROKE_BROKEN:
        mask = cv2.dilate(mask, np.ones((2, 2), np.uint8))

    mask = _deskew(mask)
    mask = _erase_rules(mask)
    mask = _drop_bars(mask)
    mask = _crop_to_text(mask)

    # Scale so glyphs reach the recognizer's preferred x-height.
    measured = _xheight(mask)
    if measured > 0:
        scale = float(np.clip(_TARGET_XHEIGHT / measured, 1.0, _MAX_SCALE))
        if scale > 1.05:
            mask = cv2.resize(
                mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC
            )
            mask = (mask > 110).astype(np.uint8) * 255

    return cv2.bitwise_not(mask)  # black text on white
