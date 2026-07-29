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

    def test_gold_field_sanity_and_safety(self):
        """Sanity floor on perfect fields, and the safety bound that matters.

        The shipped table is fitted on *extracted* records, so feeding it gold
        fields is off-distribution and accuracy here is a loose sanity check,
        not a tight bound - the number to judge the system by is the
        out-of-fold total from tools/fit_oof.py. The catastrophic-approval
        bound is the real invariant: EVALUATION.md makes it a minimum-bar
        criterion and the second tie-breaker, so it must hold on any input.
        """
        correct = catastrophic = 0
        for record, row in zip(self.records, self.rows):
            prediction, _, _ = self.calibration.adjudicate(record)
            truth = row["adjudication"].strip()
            correct += prediction == truth
            catastrophic += truth == "DENIED" and prediction == "APPROVED"
        self.assertGreaterEqual(correct / len(self.rows), 0.85)
        self.assertLessEqual(catastrophic, 5)


class TransferSafety(unittest.TestCase):
    """Rules must key on read evidence, not on constants mined from train."""

    def test_stated_embargo_denies_any_world(self):
        """A world absent from the mined list is still caught when stated.

        EMBARGO REVIEW appears for three different home worlds in the public
        corpus; a private set may use others. Reading the status travels,
        a hardcoded world list does not.
        """
        record = Record(
            case_id="MIB-000001", visa_class="XW-1", sponsor_id="SPN-1234",
            home_world="Some-Unseen-World", registry_status="EMBARGO REVIEW",
        )
        self.assertEqual(decision_path(record), "registry_embargo")

    def test_diplomatic_exemption_survives(self):
        record = Record(
            case_id="MIB-000001", visa_class="DIP-1",
            home_world="Some-Unseen-World", registry_status="EMBARGO REVIEW",
        )
        self.assertNotEqual(decision_path(record), "registry_embargo")

    def test_clear_status_does_not_license_approval(self):
        """CLEAR accompanies approvals only 35% of the time; it is not proof."""
        record = Record(
            case_id="MIB-000001", visa_class="XW-1", sponsor_id="SPN-1234",
            registry_status="CLEAR", risk_flags_known=False,
            has_scanned_pages=True, arrival_date="2026-06-01",
            receipt_date="2026-07-01",
        )
        self.assertEqual(decision_path(record), "risk_page_unreadable")


class ApprovalSafety(unittest.TestCase):
    """Approvals are fail-closed: evidence gaps can never grant authorization."""

    def test_missing_evidence_paths_cannot_approve(self):
        from mib.policy import NO_APPROVAL_PATHS, decide
        # A distribution that would otherwise approve outright.
        probs = {"APPROVED": 0.9, "DENIED": 0.05, "NEEDS_REVIEW": 0.05}
        for path in sorted(NO_APPROVAL_PATHS):
            choice, _ = decide(probs, allow_approval=False)
            self.assertNotEqual(choice, "APPROVED", path)

    def test_approval_requires_margin_over_denial(self):
        from mib.policy import decide
        # Wins on raw expected value, but the denial mass is too close.
        thin = {"APPROVED": 0.45, "DENIED": 0.35, "NEEDS_REVIEW": 0.20}
        self.assertNotEqual(decide(thin)[0], "APPROVED")
        clear = {"APPROVED": 0.80, "DENIED": 0.05, "NEEDS_REVIEW": 0.15}
        self.assertEqual(decide(clear)[0], "APPROVED")


if __name__ == "__main__":
    unittest.main()
