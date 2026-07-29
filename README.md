# MIB Doc Challenge Solution

Offline, CPU-only document pipeline for 8090's MIB Doc Challenge: extracts
applicant records from adversarial PDF case packets and adjudicates each case
as `APPROVED`, `DENIED`, or `NEEDS_REVIEW`.

## Design

- **Render-first trust boundary** (`mib/pdfio.py`): native PDF text is kept
  only when a human reviewer would see it (crop-box overlap, font size,
  luminance contrast). Hidden text and instruction-shaped decoys are
  quarantined at ingestion and recorded as document-quality signals - they can
  never populate a field or drive a decision.
- **Closed-vocabulary extraction** (`mib/extract.py`, `mib/lexicon.py`):
  nearly every field draws from a small fixed vocabulary, so noisy OCR snaps
  to the nearest legal value with glyph-confusion-weighted edit distance, and
  refuses to snap when the nearest value is not clearly closest.
- **Path-based policy engine** (`mib/policy.py`): trusted evidence maps to a
  named decision path; each path carries an empirical outcome distribution
  fitted on the public training labels; the decision is the expected-value
  argmax under the evaluator's payoff matrix, and confidence is P(decision
  correct) - the exact quantity the Brier term scores.
- **Corpus-relative constants**: revoked sponsors and the staleness reference
  are re-derived from the corpus being scored (frequency outliers, arrival
  percentiles), so the pipeline transfers to a private test set whose
  constants differ from the public data.
- **Timeout-resilient output** (`mib/output.py`, `mib/pipeline.py`):
  schema-clamped rows are flushed per case; the final corpus-context pass
  replaces the file atomically. A container stopped at the runtime limit
  still scores on everything processed so far.

## Run

```bash
docker build -t mib-submission .
docker run --rm --network none \
  --mount type=bind,src=/path/to/pdfs,dst=/input,readonly \
  --mount type=bind,src=/path/to/output,dst=/output \
  mib-submission /input /output/predictions.jsonl
```

## Develop

```bash
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python -m unittest discover -s tests
.venv/bin/python tools/fit_calibration.py \
  --labels ../mib-doc-challenge/data/train_labels.csv \
  --output policy/calibration.json
```

See `ATTRIBUTION.md` for design-idea provenance.
