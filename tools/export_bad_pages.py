#!/usr/bin/env python3
"""Export the pages the pipeline actually fails on, plus a rectified version.

Two failure classes, and nothing else:

  cfa/         packets we approved that are truly DENIED - the catastrophic
               false approvals, where the evidence that should have stopped
               the approval sat on a page we misread or never read.
  confident/   any other wrong adjudication we reported at high confidence.
               Being wrong is survivable; being wrong and sure of it is the
               failure that would mislead a human reviewer.
  unreadable/  pages where OCR recovered essentially nothing beyond the
               footer boilerplate, despite the page carrying a raster image.

Every manifest row carries the confidence we reported, so the cost of each
failure is visible rather than implied.

An earlier version selected any raster page in a packet with a blank field,
which swept in pages that read perfectly well. These are the pages where the
pipeline is demonstrably blind.

Each is written as a standalone single-page PDF with a sibling
<name>_fixed.pdf holding the rectified image, so the two open side by side.

Fixing applies the same treatment as mib/rectify, at whole-page scale so the
result stays recognisable: red-channel selection to drop watermarks and
stamps, deskew, morphological removal of the ruled grid, and a per-page
contrast or stroke-thinning pass chosen from the block's measured ink.

Usage:
  python tools/export_bad_pages.py --cache cache/train_v13.jsonl \
      --labels ../mib-doc-challenge/data/train_labels.csv \
      --pdf-dir ../mib-doc-challenge/data/train --out bad --limit 40
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

from mib.rectify import _deskew, locate_block  # noqa: E402
from tools.build_cache import load_cache  # noqa: E402

# Fields whose absence costs a decision, and the page that should carry them.
POLICY_FIELDS = ("fee_status", "visa_class", "sponsor_id", "species_code",
                 "home_world", "declared_purpose", "arrival_date")

RENDER_DPI = 300
FIX_DPI = 400


import re as _re

_FOOTER = _re.compile(r"Packet MIB-\d+.*|Synthetic hiring.*", _re.IGNORECASE)


def page_is_raster(page: pymupdf.Page) -> bool:
    body = _FOOTER.sub("", page.get_text()).strip()
    return len(body) < 60 and bool(page.get_images())


def recovered_chars(page: pymupdf.Page) -> int:
    """Body characters OCR actually recovers from this page, footer excluded.

    The footer is crisp native text stamped over the raster and always reads,
    so counting it would mark every destroyed page as readable.
    """
    from mib.pdfio import _tesseract, render_gray

    text, _ = _tesseract(render_gray(page, dpi=300), psm=3)
    native = _FOOTER.sub("", page.get_text())
    return len(_re.sub(r"\W", "", _FOOTER.sub("", text) + native))


def _erase_rules_gray(gray: np.ndarray) -> np.ndarray:
    """Whiten long thin rules while leaving grayscale text untouched.

    Binarizing first destroys faint thin strokes - measured: a page whose
    sponsor id read correctly before rectification read as noise after. So
    rules are detected on a binary copy but erased from the GRAYSCALE image,
    and only where they do not overlap text-sized ink.
    """
    import cv2

    binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    long_h = max(60, gray.shape[1] // 8)
    long_v = max(60, gray.shape[0] // 8)
    horizontal = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (long_h, 1))
    )
    vertical = cv2.morphologyEx(
        binary, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, long_v))
    )
    rules = cv2.bitwise_or(horizontal, vertical)
    # An earlier guard subtracted all dilated text-sized ink from this mask to
    # protect glyphs a rule passes through. Measured: it cancelled the mask
    # entirely - rules were erased on 0% of pages. Rules are thin by
    # construction (they survived a 1-pixel-tall opening), so erasing them
    # outright costs at most a stroke-width of a crossed glyph, which the
    # closing below restores.
    out = gray.copy()
    out[rules > 0] = 255
    repaired = cv2.morphologyEx(255 - out, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
    return 255 - repaired


def fix_page(page: pymupdf.Page) -> np.ndarray:
    """Whole-page rectification, deliberately conservative.

    Each step is reversible in spirit: nothing is binarized, nothing small is
    deleted. An earlier aggressive version scored worse than the original on
    pages that were already readable, which is the failure mode to avoid -
    a "fix" must never make a legible page illegible.
    """
    import cv2

    pix = page.get_pixmap(dpi=FIX_DPI, alpha=False)
    rgb = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, 3)

    # Red channel: red watermarks, COPY/FILED stamps and wax seals go white
    # while black text is unaffected. Grayscale would average them into it.
    gray = rgb[:, :, 0]

    block = locate_block(page)
    faint = block.faint if block else False
    bloated = block.bloated if block else False

    gray = _deskew(gray)
    gray = _erase_rules_gray(gray)
    if faint:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
        gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    if bloated:
        # Merged glyphs only: thin strokes back toward separability.
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return gray


def write_page_pdf(page: pymupdf.Page, destination: Path) -> None:
    out = pymupdf.open()
    out.insert_pdf(page.parent, from_page=page.number, to_page=page.number)
    out.save(destination)
    out.close()


def write_image_pdf(image: np.ndarray, destination: Path, rect: pymupdf.Rect) -> None:
    import cv2

    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        return
    out = pymupdf.open()
    page = out.new_page(width=rect.width, height=rect.height)
    page.insert_image(pymupdf.Rect(0, 0, rect.width, rect.height),
                      stream=io.BytesIO(encoded.tobytes()))
    out.save(destination)
    out.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--out", default="bad")
    parser.add_argument("--limit", type=int, default=40)
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        labels = {row["case_id"]: row for row in csv.DictReader(f)}

    records = {r.case_id: r for r in load_cache(args.cache)}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Reproduce the shipped decision to find catastrophic false approvals.
    from mib.policy import (
        Calibration, apply_corpus_context, corpus_reference_date,
        corpus_revoked_sponsors, emitted_guardrail,
    )

    values = list(records.values())
    reference = corpus_reference_date(
        [r.arrival_date for r in values if r.arrival() is not None]
    )
    revoked = corpus_revoked_sponsors([r.sponsor_id for r in values])
    for record in values:
        apply_corpus_context(record, reference, revoked)
    calibration = Calibration()

    CONFIDENT = 0.70
    cfa_cases, confident_errors = [], []
    for record in values:
        truth = labels.get(record.case_id)
        if not truth:
            continue
        verdict, confidence, _ = calibration.adjudicate(record)
        row = {
            f: ("|".join(sorted(record.flag_set())) or "none")
            if f == "risk_flags" else str(getattr(record, f))
            for f in POLICY_FIELDS + ("risk_flags",)
        }
        row["adjudication"] = verdict
        demotion = emitted_guardrail(row, record)
        if demotion:
            verdict, confidence = demotion
        gold = truth["adjudication"].strip()
        if verdict == "APPROVED" and gold == "DENIED":
            cfa_cases.append((record.case_id, verdict, confidence))
        elif verdict != gold and confidence >= CONFIDENT:
            confident_errors.append((record.case_id, verdict, confidence))

    for name in ("cfa", "confident", "unreadable"):
        (out_dir / name).mkdir(exist_ok=True)
    manifest = []

    # 1 & 2. Wrong decisions: every page of the packet, worst class first.
    for folder, cases in (("cfa", cfa_cases), ("confident", confident_errors)):
        for case_id, verdict, confidence in cases[: args.limit]:
            document = pymupdf.open(f"{args.pdf_dir}/{case_id}.pdf")
            truth = labels[case_id]
            for page in document:
                stem = f"{case_id}_p{page.number + 1}"
                write_page_pdf(page, out_dir / folder / f"{stem}.pdf")
                try:
                    write_image_pdf(fix_page(page),
                                    out_dir / folder / f"{stem}_fixed.pdf", page.rect)
                except Exception:  # noqa: BLE001
                    pass
                manifest.append((
                    folder, stem, f"{confidence:.2f}",
                    f"said {verdict}, truth {truth['adjudication']} "
                    f"(visa={truth['visa_class']} flags={truth['risk_flags'] or 'none'} "
                    f"fee={truth['fee_status']} sponsor={truth['sponsor_id']})",
                ))
            document.close()
    print(f"catastrophic false approvals: {len(cfa_cases)}")
    print(f"confident wrong decisions:    {len(confident_errors)}")

    # 2. Pages OCR cannot read at all.
    exported = 0
    for case_id in records:
        if case_id not in labels or exported >= args.limit:
            continue
        document = pymupdf.open(f"{args.pdf_dir}/{case_id}.pdf")
        for page in document:
            if exported >= args.limit or not page_is_raster(page):
                continue
            chars = recovered_chars(page)
            if chars > 25:
                continue  # OCR read this page; not a failure
            stem = f"{case_id}_p{page.number + 1}"
            write_page_pdf(page, out_dir / "unreadable" / f"{stem}.pdf")
            try:
                fixed = fix_page(page)
                write_image_pdf(fixed, out_dir / "unreadable" / f"{stem}_fixed.pdf",
                                page.rect)
            except Exception as error:  # noqa: BLE001
                print(f"  {stem}: fix failed ({type(error).__name__})")
                continue
            manifest.append(("unreadable", stem, "", f"OCR recovered {chars} body chars"))
            exported += 1
        document.close()

    with open(out_dir / "MANIFEST.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["class", "page", "confidence", "why"])
        writer.writerows(manifest)

    print(f"unreadable pages: {exported}")
    print(f"total pairs in {out_dir}/: {len(manifest)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
