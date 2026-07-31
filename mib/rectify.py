"""Rectification front-end: localize the field block, then make it readable.

Looking at the damaged pages settles what they actually need. The field text
occupies roughly five percent of a page - always the upper-left - while the
rest is ruled grid, watermarks, seals and scan bands. Whole-page OCR spends
its resolution and layout budget on decoration and returns the footer.

So this does not recognize anything. It hands the existing engines an image
they can already read: the field block, cropped, upright, de-ruled, free of
coloured overlays, re-rendered from source at high resolution. Fewer pixels
than a full-page pass, which makes it faster as well as more accurate.

Two traps found by inspection are handled explicitly:

- The footer is crisp NATIVE text composited over the degraded raster. It is
  the cleanest glyph cluster on the page and captures any density-based
  localizer, so the footer band is masked before locating anything.
- Damage is per page, not per packet: one page of a packet can be stroke
  bloated into merged blobs while the next is faint but sharp. The treatment
  is chosen from measured ink statistics rather than assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

import pymupdf

# Field text never sits below this fraction of page height; the footer does.
_BODY_LIMIT = 0.70
_LOCATE_DPI = 150
_READ_DPI = 600

# A glyph component at the locate resolution.
_MIN_GLYPH_H = 3
_MAX_GLYPH_H_FRAC = 0.035
_MIN_GLYPH_FILL = 0.12
_MIN_COMPONENTS = 8

_MAX_SKEW_DEG = 8.0
_MIN_SKEW_DEG = 0.3

# Calibrated on the block-ink distribution across pages the pipeline fails:
# ink_mean is bimodal (min 20 / p25 20 / med 111 / max 149), stroke_ratio
# medians 2.75 with the merged-glyph tail above 3.1.
_FAINT_INK = 70.0
_BLOATED_STROKE = 3.1


@dataclass
class Block:
    """Field-text region as page fractions, plus the page's damage profile."""

    x0: float
    y0: float
    x1: float
    y1: float
    faint: bool
    bloated: bool


def _page_gray(page: pymupdf.Page, dpi: int, clip=None) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False, clip=clip)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def _page_rgb(page: pymupdf.Page, dpi: int, clip=None) -> np.ndarray:
    pix = page.get_pixmap(dpi=dpi, alpha=False, clip=clip)
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)


def locate_block(page: pymupdf.Page) -> Block | None:
    """Find the field-text block and characterise the page's damage."""
    import cv2

    gray = _page_gray(page, _LOCATE_DPI)
    height, width = gray.shape
    masked = gray.copy()
    masked[int(height * _BODY_LIMIT):, :] = 255  # footer is native text, not evidence

    dark = (masked < 140).astype(np.uint8)
    count, _, stats, _ = cv2.connectedComponentsWithStats(dark, 8)
    boxes = []
    for index in range(1, count):
        x, y, w, h, area = stats[index]
        if not (_MIN_GLYPH_H <= h <= int(height * _MAX_GLYPH_H_FRAC)):
            continue
        if w > int(width * 0.5) or area < 6:
            continue
        if area / max(1, w * h) < _MIN_GLYPH_FILL:  # hairline rule, not a glyph
            continue
        boxes.append((x, y, w, h))
    if len(boxes) < _MIN_COMPONENTS:
        return None

    # Densest vertical band of glyph components is the form's field block.
    tops = np.array([b[1] for b in boxes])
    hist, edges = np.histogram(tops, bins=24, range=(0, int(height * _BODY_LIMIT)))
    peak = int(np.argmax(hist))
    low = edges[max(0, peak - 2)]
    high = edges[min(len(edges) - 1, peak + 4)]
    selected = [b for b in boxes if low - 10 <= b[1] <= high + 10]
    if len(selected) < 6:
        return None

    x0 = min(b[0] for b in selected)
    x1 = max(b[0] + b[2] for b in selected)
    y0 = min(b[1] for b in selected)
    y1 = max(b[1] + b[3] for b in selected)
    pad = int((y1 - y0) * 0.25) + 6

    # Damage profile from the block's own ink, calibrated against the measured
    # distribution over failing pages rather than invented. Earlier thresholds
    # (max-minus-min spread, 28% coverage) never fired on any page: a stage
    # that cannot trigger is worse than no stage, because it looks like one.
    region = gray[max(0, y0 - pad):min(height, y1 + pad),
                  max(0, x0 - pad):min(width, x1 + pad * 4)]
    ink = region[region < 200]
    ink_mean = float(ink.mean()) if ink.size else 255.0
    binary = (region < 160).astype(np.uint8)
    eroded = cv2.erode(binary, np.ones((3, 3), np.uint8))
    stroke = float(binary.sum() / max(1, binary.sum() - eroded.sum()))

    return Block(
        x0=max(0, x0 - pad) / width,
        y0=max(0, y0 - pad) / height,
        x1=min(width, x1 + pad * 4) / width,
        y1=min(height, y1 + pad) / height,
        # Ink intensity is bimodal across failing pages: black text sits near
        # 20, washed-out text between 110 and 150. Anything above the trough
        # is faint and wants contrast.
        faint=ink_mean > _FAINT_INK,
        # Stroke ratio (ink area over erosion boundary) medians 2.75; the
        # merged-glyph pages sit in the upper tail.
        bloated=stroke > _BLOATED_STROKE,
    )


def _deskew(gray: np.ndarray) -> np.ndarray:
    import cv2

    coords = np.column_stack(np.where((255 - gray) > 60))
    if len(coords) < 100:
        return gray
    angle = cv2.minAreaRect(coords[:, ::-1].astype(np.float32))[-1]
    if angle < -45:
        angle += 90
    if angle > 45:
        angle -= 90
    if not (_MIN_SKEW_DEG <= abs(angle) <= _MAX_SKEW_DEG):
        return gray
    matrix = cv2.getRotationMatrix2D(
        (gray.shape[1] / 2, gray.shape[0] / 2), angle, 1.0
    )
    return cv2.warpAffine(
        gray, matrix, (gray.shape[1], gray.shape[0]),
        flags=cv2.INTER_CUBIC, borderValue=255,
    )


def _strip_rules(gray: np.ndarray) -> np.ndarray:
    import cv2

    binary = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 31, 11
    )
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(30, gray.shape[1] // 12), 1)),
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(30, gray.shape[0] // 8))),
    )
    cleaned = cv2.subtract(binary, cv2.bitwise_or(horizontal, vertical))
    return cv2.bitwise_not(cleaned)


def render_block(page: pymupdf.Page, block: Block) -> np.ndarray | None:
    """Re-render the block from source at high DPI and rectify it.

    Colour matters here: watermarks, COPY/FILED stamps and wax seals are red,
    and a grayscale conversion averages them into the text. Taking the red
    channel erases them, which is free accuracy on stamped pages.
    """
    import cv2

    rect = page.rect
    clip = pymupdf.Rect(
        rect.x0 + block.x0 * rect.width,
        rect.y0 + block.y0 * rect.height,
        rect.x0 + block.x1 * rect.width,
        rect.y0 + block.y1 * rect.height,
    ) & rect
    if clip.is_empty or clip.width < 4 or clip.height < 4:
        return None

    rgb = _page_rgb(page, _READ_DPI, clip=clip)
    if rgb.shape[0] < 12 or rgb.shape[1] < 12:
        return None
    gray = rgb[:, :, 0]  # red channel: red overlays vanish, black text stays

    gray = _deskew(gray)
    if block.faint:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(gray)
    if block.bloated:
        # Merged glyphs: thin the strokes back toward separable characters.
        gray = cv2.erode(gray, np.ones((2, 2), np.uint8))
    return _strip_rules(gray)


def rectified_text(page: pymupdf.Page) -> str:
    """Localize, rectify and read a page's field block. '' when not locatable."""
    from .pdfio import _tesseract

    block = locate_block(page)
    if block is None:
        return ""
    image = render_block(page, block)
    if image is None:
        return ""
    text, _ = _tesseract(image, psm=6)
    return text or ""
