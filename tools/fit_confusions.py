#!/usr/bin/env python3
"""Estimate this corpus's OCR character-confusion model, self-supervised.

Every page carries a footer whose exact text is known in advance - the packet's
own case id, its page number, and a fixed boilerplate line. That is known
plaintext sitting on the same damaged rasters as the fields we actually care
about, so OCR of the footer can be aligned against ground truth without a
single label.

Aligning them yields the substitutions this generator's damage actually
produces, replacing the hand-picked confusable list in `lexicon` with measured
costs. It also transfers: the model is re-derivable from any corpus, including
a private one, because the footer is part of the document rather than something
we had to be told.

Usage:
  python tools/fit_confusions.py --pdf-dir ../mib-doc-challenge/data/train \
      --output policy/confusions.json --limit 300
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

BOILERPLATE = "Synthetic hiring challenge document"
_FOOTER_HINT = re.compile(r"(synth|hiring|challenge|document|packet|page)", re.IGNORECASE)


def _distance(a: str, b: str) -> int:
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1,
                               previous[j - 1] + (ca != cb)))
        previous = current
    return previous[-1]


def align_ops(truth: str, observed: str):
    """Yield ('sub', a, b) pairs from an edit-distance backtrace."""
    n, m = len(truth), len(observed)
    if not n or not m or abs(n - m) > max(6, n // 3):
        return []
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if truth[i - 1] == observed[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    ops = []
    i, j = n, m
    while i > 0 and j > 0:
        cost = 0 if truth[i - 1] == observed[j - 1] else 1
        if dp[i][j] == dp[i - 1][j - 1] + cost:
            if cost:
                ops.append((truth[i - 1], observed[j - 1]))
            i, j = i - 1, j - 1
        elif dp[i][j] == dp[i - 1][j] + 1:
            i -= 1
        else:
            j -= 1
    return ops


def _scan(pdf_path: str) -> Counter:
    from mib.pdfio import Page, ocr_page, native_text_split, _FOOTER_RE  # noqa: F401
    import pymupdf

    pairs: Counter = Counter()
    case_id = Path(pdf_path).stem
    try:
        with pymupdf.open(pdf_path) as document:
            for index, page in enumerate(document):
                visible, _ = native_text_split(page)
                # Only raster pages carry useful damage; digital pages OCR
                # perfectly and would bias the model toward the identity.
                if len(_FOOTER_RE.sub("", visible).strip()) >= 40:
                    continue
                text, _, _, _ = ocr_page(page)
                expected = [BOILERPLATE, f"Packet {case_id} / page {index + 1}"]
                lines = [l.strip() for l in text.splitlines()
                         if l.strip() and _FOOTER_HINT.search(l)]
                for truth in expected:
                    if not lines:
                        break
                    # Only align a line that is genuinely a damaged copy of the
                    # expected text. Picking the nearest by length alone let an
                    # unrelated line align against the footer and manufacture
                    # impossible confusions ('y' -> 'P', 'd' -> '/').
                    scored = [(_distance(truth, l) / max(1, len(truth)), l)
                              for l in lines]
                    ratio, best = min(scored)
                    if ratio > 0.35:
                        continue
                    for a, b in align_ops(truth, best):
                        if a.strip() and b.strip():
                            pairs[(a, b)] += 1
    except Exception:  # noqa: BLE001
        pass
    return pairs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pdf-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=300)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    pdfs = sorted(str(p) for p in Path(args.pdf_dir).glob("*.pdf"))[: args.limit]
    totals: Counter = Counter()
    with mp.get_context("spawn").Pool(args.workers) as pool:
        for result in pool.imap_unordered(_scan, pdfs, chunksize=4):
            totals.update(result)

    # Per-character normalisation: how often is `a` misread as `b`, given that
    # `a` was misread at all. Rare pairs are dropped as noise.
    by_source: Counter = Counter()
    for (a, _b), n in totals.items():
        by_source[a] += n

    confusions = {}
    for (a, b), n in totals.items():
        if n < 3:
            continue
        confusions[f"{a}|{b}"] = round(n / by_source[a], 4)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(
            {
                "source": "page-footer known plaintext",
                "packets_scanned": len(pdfs),
                "observations": sum(totals.values()),
                "confusions": confusions,
            },
            f, indent=2, sort_keys=True,
        )
        f.write("\n")

    print(f"scanned {len(pdfs)} packets, {sum(totals.values())} substitutions")
    print(f"kept {len(confusions)} pairs seen 3+ times")
    for (a, b), n in totals.most_common(25):
        if n >= 3:
            print(f"  {a!r} -> {b!r}  n={n:4d}  p={n / by_source[a]:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
