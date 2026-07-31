#!/usr/bin/env python3
"""Measure the matched recognizer on fields the shipped pipeline left blank.

For each blank closed-vocabulary field with known truth, locate the value
cell (label word-boxes first, template coordinates as fallback), fit the
page's degradation channel from its footer, and ask the recognizer which
vocabulary word explains the blob. Report correct / wrong / refused per
field. Wrong answers are the metric that matters: this engine is only
shippable if its failure mode is silence.

Usage:
  python tools/probe_matched.py --cache cache/train_v13.jsonl \
      --labels ../mib-doc-challenge/data/train_labels.csv --limit 60
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pymupdf  # noqa: E402

from mib.constants import (  # noqa: E402
    FEE_VALUES,
    HOME_WORLDS,
    PURPOSES,
    RISK_FLAGS,
    SPECIES_CODES,
)
from mib.matched import fit_channel, read_blob  # noqa: E402
from mib.pdfio import _tesseract_tsv, render_gray  # noqa: E402
from mib.roi import _GAP, _LINE_PAD, _VALUE_W, find_label  # noqa: E402
from tools.build_cache import load_cache  # noqa: E402

CANDIDATES = {
    "fee_status": tuple(FEE_VALUES),
    "species_code": tuple(SPECIES_CODES),
    "home_world": tuple(HOME_WORLDS),
    "declared_purpose": tuple(PURPOSES),
    "risk_flags": ("none",) + tuple(RISK_FLAGS),
}


def cell_for(page, gray, dpi, field):
    boxes = [
        (w, x * 72 / dpi, y * 72 / dpi, (x + bw) * 72 / dpi, (y + bh) * 72 / dpi)
        for w, x, y, bw, bh in _tesseract_tsv(gray)
    ]
    label = find_label(boxes, field)
    if label is None:
        return None
    _, y0, x1, y1 = label
    rect = pymupdf.Rect(x1 + _GAP, y0 - _LINE_PAD, x1 + _GAP + _VALUE_W, y1 + _LINE_PAD)
    rect = rect & page.rect
    if rect.is_empty:
        return None
    pix = page.get_pixmap(dpi=400, colorspace=pymupdf.csGRAY, clip=rect)
    if pix.width < 12 or pix.height < 12:
        return None
    return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", required=True)
    parser.add_argument("--labels", required=True)
    parser.add_argument("--pdf-dir", default="../mib-doc-challenge/data/train")
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    with open(args.labels, newline="") as f:
        labels = {row["case_id"]: row for row in csv.DictReader(f)}

    records = {r.case_id: r for r in load_cache(args.cache)}
    results = {f: Counter() for f in CANDIDATES}
    channel_stats = Counter()
    examined = 0

    for case_id, record in records.items():
        if case_id not in labels:
            continue
        wanted = []
        for field in CANDIDATES:
            if field == "risk_flags":
                if not record.risk_flags_known and labels[case_id][field].strip():
                    wanted.append(field)
            elif (
                str(getattr(record, field)) == "unknown"
                and labels[case_id][field].strip()
            ):
                wanted.append(field)
        if not wanted:
            continue
        examined += 1
        if examined > args.limit:
            break

        doc = pymupdf.open(f"{args.pdf_dir}/{case_id}.pdf")
        for index, page in enumerate(doc):
            if not wanted:
                break
            gray = render_gray(page, dpi=300)
            channel = fit_channel(gray, case_id, index + 1)
            if channel is None:
                channel_stats["no_channel"] += 1
                continue
            channel_stats["fitted"] += 1
            for field in list(wanted):
                cell = cell_for(page, gray, 300, field)
                if cell is None:
                    continue
                verdict = read_blob(cell, CANDIDATES[field], channel)
                truth = labels[case_id][field].strip() or "none"
                if verdict is None:
                    results[field]["refused"] += 1
                elif field == "risk_flags":
                    ok = (verdict == "none" and truth == "none") or (
                        verdict != "none" and verdict in truth
                    )
                    results[field]["correct" if ok else "WRONG"] += 1
                    wanted.remove(field)
                else:
                    results[field]["correct" if verdict.lower() == truth.lower() else "WRONG"] += 1
                    wanted.remove(field)
        doc.close()

    print(f"packets examined: {min(examined, args.limit)}")
    print(f"channel fits: {dict(channel_stats)}\n")
    total_c = total_w = 0
    for field, counts in results.items():
        if not counts:
            continue
        print(f"  {field:18s} {dict(counts)}")
        total_c += counts.get("correct", 0)
        total_w += counts.get("WRONG", 0)
    if total_c + total_w:
        print(f"\nprecision when deciding: {total_c}/{total_c + total_w} "
              f"= {total_c / (total_c + total_w):.0%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
