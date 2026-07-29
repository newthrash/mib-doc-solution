"""End-to-end run: PDFs in, predictions.jsonl out.

Two-phase design for timeout resilience:

Phase 1 - extraction (the expensive part) runs across worker processes;
a provisional, schema-valid row is flushed to disk the moment each case
finishes. A container killed at the runtime limit is scored on these.

Phase 2 - corpus context (staleness reference, revoked-sponsor outliers) is
computed over all extracted records, every case is re-adjudicated with it,
output-fill and the one-way emitted-fields guardrail are applied, and the
final file replaces the provisional one atomically.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import sys
import traceback
from pathlib import Path

from .extract import extract_record
from .output import IncrementalWriter, apply_priors, load_priors
from .pdfio import load_packet
from .policy import (
    Calibration,
    apply_corpus_context,
    corpus_reference_date,
    corpus_revoked_sponsors,
    emitted_guardrail,
)
from .record import Record


def _workers() -> int:
    if os.environ.get("MIB_WORKERS"):
        return max(1, int(os.environ["MIB_WORKERS"]))
    return max(1, min(4, os.cpu_count() or 1))


def _process_case(pdf_path: str) -> Record:
    try:
        packet = load_packet(pdf_path)
        return extract_record(packet)
    except Exception:  # noqa: BLE001 - per-case isolation
        traceback.print_exc(file=sys.stderr)
        record = Record(case_id=Path(pdf_path).stem)
        record.risk_flags_known = False
        return record


def _row_from(record: Record, adjudication: str, confidence: float) -> dict:
    return {
        "case_id": record.case_id,
        "applicant_name": record.applicant_name,
        "species_code": record.species_code,
        "home_world": record.home_world,
        "visa_class": record.visa_class,
        "sponsor_id": record.sponsor_id,
        "arrival_date": record.arrival_date,
        "declared_purpose": record.declared_purpose,
        "risk_flags": "|".join(sorted(record.flag_set())) or "none",
        "fee_status": record.fee_status,
        "adjudication": adjudication,
        "confidence": confidence,
    }


def run(input_dir: str, output_path: str) -> int:
    pdfs = sorted(str(p) for p in Path(input_dir).glob("*.pdf"))
    calibration = Calibration()
    priors = load_priors()
    writer = IncrementalWriter(output_path)
    records: list[Record] = []

    if not pdfs:
        writer.close()
        return 0

    with mp.get_context("spawn").Pool(processes=_workers()) as pool:
        for record in pool.imap_unordered(_process_case, pdfs, chunksize=4):
            records.append(record)
            # Provisional decision without corpus context; overwritten in phase 2.
            adjudication, confidence, _ = calibration.adjudicate(record)
            writer.write(_row_from(record, adjudication, confidence))

    # Phase 2: corpus context + final adjudication + guardrail.
    reference = corpus_reference_date(
        [r.arrival_date for r in records if r.arrival() is not None]
    )
    revoked = corpus_revoked_sponsors([r.sponsor_id for r in records])
    final_rows: list[dict] = []
    for record in records:
        apply_corpus_context(record, reference, revoked)
        adjudication, confidence, _path = calibration.adjudicate(record)
        row = _row_from(record, adjudication, confidence)
        # The guardrail inspects the evidence-backed row, before any prior is
        # applied, so a filled slot can never suppress a demotion.
        demotion = emitted_guardrail(row, record)
        if demotion:
            row["adjudication"], row["confidence"] = demotion
        final_rows.append(apply_priors(row, priors))

    written = writer.rewrite_all(final_rows)
    print(f"wrote {written} predictions to {output_path}", file=sys.stderr)
    return 0


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print("usage: python -m mib.pipeline <input_pdf_dir> <output_path>", file=sys.stderr)
        return 2
    return run(argv[1], argv[2])


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
