"""Policy invariants that must hold regardless of extraction quality."""

import csv
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.constants import UNKNOWN
from mib.policy import (
    Calibration,
    apply_corpus_context,
    corpus_reference_date,
    corpus_revoked_sponsors,
    decision_path,
    emitted_guardrail,
)
from mib.record import Record

LABELS = Path(__file__).resolve().parent.parent.parent / "mib-doc-challenge" / "data" / "train_labels.csv"


def gold_record(row: dict) -> Record:
    flags = row["risk_flags"].strip()
    return Record(
        case_id=row["case_id"],
        visa_class=row["visa_class"],
        sponsor_id=row["sponsor_id"],
        home_world=row["home_world"],
        arrival_date=row["arrival_date"],
        fee_status=row["fee_status"],
        risk_flags=frozenset() if flags in ("", "none") else frozenset(flags.split("|")),
        fee_explicit_unknown=row["fee_status"] == "unknown",
    )


class PolicyInvariants(unittest.TestCase):
    def test_disqualifying_flag_denies(self):
        record = Record(case_id="MIB-000001", risk_flags=frozenset({"biohazard_red"}))
        self.assertEqual(decision_path(record), "disqualifying_flag")

    def test_unknown_visa_is_not_dip_exempt(self):
        record = Record(case_id="MIB-000001", sponsor_id="SPN-0007", visa_class=UNKNOWN)
        self.assertNotEqual(decision_path(record), "sponsor_revoked")

    def test_unread_risk_never_reaches_clean(self):
        record = Record(
            case_id="MIB-000001", visa_class="XW-1", sponsor_id="SPN-1234",
            fee_status="paid", arrival_date="2026-06-01", receipt_date="2026-07-01",
            risk_flags_known=False, has_scanned_pages=True,
        )
        self.assertEqual(decision_path(record), "risk_page_unreadable")

    def test_guardrail_demotes_contradictory_approval(self):
        record = Record(case_id="MIB-000001")
        row = {"adjudication": "APPROVED", "risk_flags": "planetary_embargo",
               "visa_class": "XW-1", "sponsor_id": "SPN-1234",
               "home_world": "Proxima-b", "fee_status": "paid"}
        decision = emitted_guardrail(row, record)
        self.assertIsNotNone(decision)
        self.assertEqual(decision[0], "DENIED")

    def test_guardrail_never_creates_approval(self):
        record = Record(case_id="MIB-000001")
        for adjudication in ("DENIED", "NEEDS_REVIEW"):
            row = {"adjudication": adjudication, "risk_flags": "none",
                   "visa_class": "DIP-1", "sponsor_id": "SPN-1234",
                   "home_world": "Proxima-b", "fee_status": "paid"}
            self.assertIsNone(emitted_guardrail(row, record))


@unittest.skipUnless(LABELS.exists(), "challenge labels not present")
class OracleFloor(unittest.TestCase):
    """Scored against public labels with gold fields: the policy ceiling."""

    @classmethod
    def setUpClass(cls):
        with open(LABELS, newline="") as f:
            cls.rows = list(csv.DictReader(f))
        cls.records = [gold_record(row) for row in cls.rows]
        reference = corpus_reference_date([r.arrival_date for r in cls.records])
        revoked = corpus_revoked_sponsors([r.sponsor_id for r in cls.records])
        for record in cls.records:
            apply_corpus_context(record, reference, revoked)
        cls.calibration = Calibration()

    def test_oracle_accuracy_floor(self):
        correct = catastrophic = 0
        for record, row in zip(self.records, self.rows):
            prediction, _, _ = self.calibration.adjudicate(record)
            truth = row["adjudication"].strip()
            correct += prediction == truth
            catastrophic += truth == "DENIED" and prediction == "APPROVED"
        self.assertGreaterEqual(correct / len(self.rows), 0.93)
        self.assertLessEqual(catastrophic, 10)


if __name__ == "__main__":
    unittest.main()
