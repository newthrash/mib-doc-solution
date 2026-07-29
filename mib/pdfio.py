"""PDF ingestion with the rendered page as the trust boundary.

Native PDF text is extracted span-by-span and kept only when the span would
actually be visible to a human reviewer: inside the crop box, readable size,
and real contrast against white. Everything else - white-on-white injections,
off-crop decoys, invisible render modes - is dropped before any downstream
code sees it, and its presence is recorded as `injection_detected`.

Every page is also rasterized and OCRed, so field evidence ultimately comes
from visible pixels. OCR retries are bounded and gated so clean pages pay a
single pass (budget: ~6s/PDF average on 4 vCPUs).
"""

from __future__ import annotations

import os
import re
import subprocess
from dataclasses import dataclass, field

import numpy as np

import pymupdf

# Instruction-shaped decoys are removed from all text views; their presence is
# a document-quality signal, never evidence.
UNTRUSTED_RE = re.compile(
    r"(?:ANSWER\s*KEY|BARCODE\s+PAYLOAD|OUTPUT\s+THIS|\bSYSTEM\s*:"
    r"|FORCE\s+ADJUDICATION|IGNORE\s+(?:ALL|PREVIOUS)|DO\s+NOT\s+OCR)",
    re.IGNORECASE,
)

_MIN_FONT_SIZE = 4.5
_MIN_CONTRAST = 28.0
_MIN_CROP_OVERLAP = 0.75


@dataclass
class Page:
    number: int
    kind: str = "unknown"
    visible_native: str = ""
    hidden_native: str = ""
    ocr_text: str = ""
    ocr_confidence: float = 0.0
    is_scanned: bool = False
    injection_seen: bool = False

    # Merged, injection-scrubbed view for parsers.
    @property
    def text(self) -> str:
        return "\n".join(part for part in (self.visible_native, self.ocr_text) if part)


@dataclass
class Packet:
    case_id: str
    pdf_name: str
    pages: list[Page] = field(default_factory=list)
    error: str | None = None

    @property
    def injection_detected(self) -> bool:
        return any(page.injection_seen for page in self.pages)

    @property
    def has_scanned_pages(self) -> bool:
        return any(page.is_scanned for page in self.pages)

    def full_text(self) -> str:
        return "\n".join(page.text for page in self.pages)


def _rgb(color: int) -> tuple[int, int, int]:
    return ((color >> 16) & 255, (color >> 8) & 255, color & 255)


def _span_visible(span: dict, crop: pymupdf.Rect) -> bool:
    text = str(span.get("text", "")).strip()
    if not text or float(span.get("size", 0.0)) < _MIN_FONT_SIZE:
        return False
    bbox = pymupdf.Rect(span.get("bbox", (0, 0, 0, 0)))
    overlap = bbox & crop
    if bbox.is_empty or overlap.is_empty:
        return False
    if overlap.get_area() < _MIN_CROP_OVERLAP * max(1.0, bbox.get_area()):
        return False
    # Render mode 3 is explicitly invisible text.
    if int(span.get("flags", 0)) and span.get("render_mode", 0) == 3:
        return False
    alpha = int(span.get("alpha", 255))
    red, green, blue = _rgb(int(span.get("color", 0)))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    contrast = (255.0 - luminance) * (alpha / 255.0)
    return contrast >= _MIN_CONTRAST


def native_text_split(page: pymupdf.Page) -> tuple[str, str]:
    """Return (visible, hidden) native text for the page."""
    visible_lines: list[str] = []
    hidden_lines: list[str] = []
    crop = page.rect
    for block in page.get_text("dict", sort=True).get("blocks", []):
        for line in block.get("lines", []):
            visible_parts, hidden_parts = [], []
            for span in line.get("spans", []):
                text = str(span.get("text", "")).strip()
                if not text:
                    continue
                (visible_parts if _span_visible(span, crop) else hidden_parts).append(text)
            if visible_parts:
                visible_lines.append(" ".join(visible_parts))
            if hidden_parts:
                hidden_lines.append(" ".join(hidden_parts))
    return "\n".join(visible_lines), "\n".join(hidden_lines)


def scrub(text: str) -> tuple[str, bool]:
    """Remove instruction-shaped lines; report whether any were present."""
    kept: list[str] = []
    seen = False
    for line in text.splitlines():
        if UNTRUSTED_RE.search(line):
            seen = True
            continue
        kept.append(line)
    return "\n".join(kept), seen


def _tesseract(gray: np.ndarray, psm: int, timeout: int = 25) -> tuple[str, float]:
    import cv2

    ok, encoded = cv2.imencode(".png", gray)
    if not ok:
        return "", 0.0
    env = os.environ.copy()
    env.setdefault("OMP_THREAD_LIMIT", "1")
    try:
        result = subprocess.run(
            ["tesseract", "stdin", "stdout", "--dpi", "220", "--psm", str(psm),
             "-l", "eng", "-c", "preserve_interword_spaces=1", "tsv"],
            input=encoded.tobytes(),
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            check=False,
            env=env,
            timeout=timeout,
        )
    except subprocess.SubprocessError:
        return "", 0.0
    if result.returncode != 0:
        return "", 0.0
    return _tsv_lines(result.stdout.decode("utf-8", errors="replace"))


def _tsv_lines(tsv: str) -> tuple[str, float]:
    groups: dict[tuple[str, str, str, str], list[tuple[int, str]]] = {}
    confidences: list[float] = []
    for raw in tsv.splitlines():
        parts = raw.split("\t", 11)
        if len(parts) != 12 or parts[0] == "level":
            continue
        word = parts[11].strip()
        if not word:
            continue
        try:
            confidence = float(parts[10])
        except ValueError:
            confidence = -1.0
        if confidence >= 0:
            confidences.append(confidence)
        try:
            left = int(parts[6])
        except ValueError:
            left = 0
        groups.setdefault((parts[1], parts[2], parts[3], parts[4]), []).append((left, word))
    lines = [" ".join(w for _, w in sorted(words)) for words in groups.values()]
    return "\n".join(lines), float(np.mean(confidences)) if confidences else 0.0


def _enhance(gray: np.ndarray) -> np.ndarray:
    import cv2

    low, high = np.percentile(gray, (1.0, 99.0))
    if high - low < 80:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(12, 12)).apply(gray)
    return gray


def render_gray(page: pymupdf.Page, dpi: int = 220) -> np.ndarray:
    pixmap = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY, alpha=False, annots=True)
    return np.frombuffer(pixmap.samples, dtype=np.uint8).reshape(pixmap.height, pixmap.width)


_FOOTER_RE = re.compile(r"Packet\s+MIB-\d+.*|Synthetic hiring.*", re.IGNORECASE)

# Field labels that indicate the page's substance was actually recovered.
# Footer boilerplate OCRs cleanly even when the form body is unreadable, so
# raw confidence alone cannot tell a good read from a page of noise.
_ANCHORS = (
    "FEE STATUS", "APPLICANT", "SPECIES", "SPONSOR", "ARRIVAL", "VISA",
    "HOME WORLD", "OBSERVED", "REGISTRY", "BIOMETRIC", "FINDING", "AMOUNT",
)


def _anchor_count(text: str) -> int:
    normalized = re.sub(r"[^A-Z ]+", " ", _FOOTER_RE.sub("", text).upper())
    return sum(anchor in normalized for anchor in _ANCHORS)


def _score(text: str, confidence: float) -> float:
    body = _FOOTER_RE.sub("", text)
    return confidence + 14.0 * _anchor_count(text) + min(18.0, len(body) / 30.0)


def ocr_page(page: pymupdf.Page, dpi: int = 220) -> tuple[str, float]:
    """OCR with bounded escalation, ordered by what actually works here.

    PSM 3 (full auto segmentation) reads these structured forms far better
    than PSM 11 (sparse text): on a damaged fee receipt PSM 11 at 220 DPI
    misses the status line entirely while PSM 3 at 300 DPI recovers it. Each
    rung is gated on anchor recovery so clean pages pay a single pass.
    """
    gray = render_gray(page, dpi=dpi)
    best = _tesseract(gray, psm=3)

    if _anchor_count(best[0]) < 2:
        candidate = _tesseract(gray, psm=11)
        if _score(*candidate) > _score(*best):
            best = candidate

    # A page whose body is still unrecovered earns one high-resolution,
    # contrast-stretched pass - the expensive rung, reserved for pages that
    # would otherwise contribute nothing.
    if _anchor_count(best[0]) < 2:
        dense = render_gray(page, dpi=300)
        for variant in (dense, _enhance(dense)):
            candidate = _tesseract(variant, psm=3)
            if _score(*candidate) > _score(*best):
                best = candidate
            if _anchor_count(best[0]) >= 2:
                break

    # Rotated scans: PDF metadata says upright but the raster is turned.
    if _anchor_count(best[0]) < 1 and len(re.sub(r"\W", "", best[0])) < 100:
        for turns in (1, 3, 2):
            candidate = _tesseract(np.ascontiguousarray(np.rot90(gray, turns)), psm=3)
            if _score(*candidate) > _score(*best):
                best = candidate
            if _anchor_count(best[0]) >= 2:
                break
    return best


def classify_page(text: str) -> str:
    normalized = re.sub(r"[^A-Z0-9]+", " ", text.upper())
    if "MANUAL ADJUDICATOR" in normalized or "FINDING" in normalized:
        return "manual"
    if any(anchor in normalized for anchor in
           ("WORK AUTHORIZATION", "PRIMARY INTAKE", "FORM I 8090", "FORM J 8090")):
        return "intake"
    if "BIOMETRIC" in normalized or "BIOHAZARD" in normalized:
        return "biometric"
    if "FEE RECEIPT" in normalized or ("FEE STATUS" in normalized and "AMOUNT" in normalized):
        return "fee"
    if "SPONSOR ATTESTATION" in normalized or ("SPONSOR" in normalized and "ATTESTS" in normalized):
        return "sponsor"
    if "REGISTRY EXTRACT" in normalized or "REGISTRY STATUS" in normalized:
        return "registry"
    return "unknown"


def load_packet(pdf_path: str, dpi: int = 220) -> Packet:
    from pathlib import Path

    path = Path(pdf_path)
    packet = Packet(case_id=path.stem, pdf_name=path.name)
    try:
        with pymupdf.open(path) as document:
            for index, page in enumerate(document):
                visible, hidden = native_text_split(page)
                visible, injected_visible = scrub(visible)
                _, injected_hidden = scrub(hidden)
                ocr_text, ocr_confidence = "", 0.0
                is_scanned = len(visible.strip()) < 40
                try:
                    ocr_text, ocr_confidence = ocr_page(page, dpi=dpi)
                    ocr_text, injected_ocr = scrub(ocr_text)
                except Exception:
                    injected_ocr = False
                packet.pages.append(
                    Page(
                        number=index + 1,
                        kind=classify_page("\n".join((visible, ocr_text))),
                        visible_native=visible,
                        hidden_native=hidden,
                        ocr_text=ocr_text,
                        ocr_confidence=ocr_confidence,
                        is_scanned=is_scanned,
                        injection_seen=bool(hidden.strip()) or injected_visible
                        or injected_hidden or injected_ocr,
                    )
                )
    except Exception as error:  # noqa: BLE001 - one bad packet must not kill the run
        packet.error = f"{type(error).__name__}: {error}"
    return packet
