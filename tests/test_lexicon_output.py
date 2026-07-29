import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.lexicon import (
    parse_date,
    repair_sponsor_id,
    snap_home_world,
    snap_species,
    snap_visa,
)
from mib.output import clamp_row


class LexiconTests(unittest.TestCase):
    def test_species_glyph_confusion(self):
        self.assertEqual(snap_species("0RION_GRAYS"), "ORION_GRAYS")
        self.assertEqual(snap_species("TR1ANGULAN"), "TRIANGULAN")
        self.assertEqual(snap_species("JOVIAN GASF0RM"), "JOVIAN_GASFORM")

    def test_species_refuses_garbage(self):
        self.assertIsNone(snap_species("ZZZZZZZZ"))
        self.assertIsNone(snap_species(""))

    def test_visa_requires_discriminating_digit(self):
        self.assertEqual(snap_visa("XW-1"), "XW-1")
        self.assertEqual(snap_visa("XW 2"), "XW-2")
        self.assertEqual(snap_visa("TRANSIT 7"), "TRANSIT-7")
        self.assertIsNone(snap_visa("XW-9"))

    def test_home_world(self):
        self.assertEqual(snap_home_world("Wolf-1O61c"), "Wolf-1061c")
        self.assertEqual(snap_home_world("Europa Statlon"), "Europa Station")

    def test_sponsor_repair(self):
        self.assertEqual(repair_sponsor_id("SPN-1042"), "SPN-1042")
        self.assertEqual(repair_sponsor_id("SPN 1O42"), "SPN-1042")
        self.assertIsNone(repair_sponsor_id("nonsense"))

    def test_date_parsing(self):
        self.assertEqual(parse_date("2026-04-17"), "2026-04-17")
        self.assertEqual(parse_date("17 Apr 2026"), "2026-04-17")
        self.assertEqual(parse_date("Apr 17, 2026"), "2026-04-17")
        self.assertIsNone(parse_date("2026-99-99"))
        self.assertIsNone(parse_date("no date here"))


class OutputClampTests(unittest.TestCase):
    def test_clamps_invalid_enums(self):
        row = clamp_row({
            "case_id": "MIB-123456", "adjudication": "approved",
            "fee_status": "PAID!", "confidence": 3.0,
            "risk_flags": "biohazard_red|made_up_flag",
            "sponsor_id": "SPN-12", "arrival_date": "17 Apr",
        })
        self.assertEqual(row["adjudication"], "APPROVED")
        self.assertEqual(row["fee_status"], "unknown")
        self.assertEqual(row["confidence"], 1.0)
        self.assertEqual(row["risk_flags"], "biohazard_red")
        self.assertEqual(row["sponsor_id"], "SPN-0000")
        self.assertEqual(row["arrival_date"], "1900-01-01")

    def test_rejects_bad_case_id(self):
        self.assertIsNone(clamp_row({"case_id": "MIB-12"}))

    def test_valid_row_passes_through(self):
        row = clamp_row({
            "case_id": "MIB-000001", "applicant_name": "Zed Zarnax",
            "species_code": "ORION_GRAYS", "home_world": "Kepler-186f",
            "visa_class": "XW-2", "sponsor_id": "SPN-1042",
            "arrival_date": "2026-04-17", "declared_purpose": "research",
            "risk_flags": "none", "fee_status": "paid",
            "adjudication": "APPROVED", "confidence": 0.91,
        })
        self.assertEqual(row["adjudication"], "APPROVED")
        self.assertEqual(row["confidence"], 0.91)


class FeeSnapSafety(unittest.TestCase):
    """A damaged `unpaid` must never resolve to `paid` - that flips a denial."""

    def test_negative_prefix_is_decisive(self):
        from mib.lexicon import snap_fee
        for token in ("unpaid", "unpai", "unpad", "vnpaid", "unpaic"):
            self.assertEqual(snap_fee(token), "unpaid", token)

    def test_truncated_positive_resolves(self):
        from mib.lexicon import snap_fee
        self.assertEqual(snap_fee("pac"), "paid")
        self.assertEqual(snap_fee("pald"), "paid")
        self.assertEqual(snap_fee("waivec"), "waived")

    def test_unknown_and_garbage(self):
        from mib.lexicon import snap_fee
        self.assertEqual(snap_fee("unknovn"), "unknown")
        self.assertIsNone(snap_fee("xyz"))
        self.assertIsNone(snap_fee(""))


class VisaSnapping(unittest.TestCase):
    """The suffix digit is trusted over the letters, except where it cannot be."""

    def test_damaged_prefix_resolves_via_unique_digit(self):
        from mib.lexicon import snap_visa
        for token in ("WED-3", "MEO-3", "MED-3"):
            self.assertEqual(snap_visa(token), "MED-3", token)
        self.assertEqual(snap_visa("XVV-2"), "XW-2")
        self.assertEqual(snap_visa("TRANS1T 7"), "TRANSIT-7")

    def test_shared_digit_still_requires_letters(self):
        """Digit 1 is XW-1 or DIP-1; a wrong DIP-1 grants a sponsor exemption."""
        from mib.lexicon import snap_visa
        self.assertEqual(snap_visa("D1P-1"), "DIP-1")
        self.assertEqual(snap_visa("XW-1"), "XW-1")
        self.assertIsNone(snap_visa("ZZ-1"))

    def test_garbage_rejected(self):
        from mib.lexicon import snap_visa
        self.assertIsNone(snap_visa("garbage"))
        self.assertIsNone(snap_visa(""))


class FlagSnapping(unittest.TestCase):
    def test_damaged_flag_token_recovers(self):
        from mib.lexicon import snap_flag
        self.assertEqual(snap_flag("llegible_biometrics"), "illegible_biometrics")
        self.assertEqual(snap_flag("biohazard_rcd"), "biohazard_red")

    def test_unrelated_token_rejected(self):
        from mib.lexicon import snap_flag
        self.assertIsNone(snap_flag("reactor_maintenance"))
        self.assertIsNone(snap_flag("abc"))


class NameSnapping(unittest.TestCase):
    """Applicant names are a closed vocabulary, so damage is repairable."""

    def test_repairs_ocr_damage_to_truth(self):
        from mib.lexicon import snap_name
        self.assertEqual(snap_name("Mirequell Qcrul"), "Miraquell Qorul")
        self.assertEqual(snap_name("Zavoss lxomora"), "Zavoss Ixomora")
        self.assertEqual(snap_name("Arinax Qommora"), "Arinax Qormora")

    def test_field_bleed_is_not_snapped_into_a_name(self):
        """A home_world read into the name slot must not become an applicant."""
        from mib.lexicon import snap_name
        self.assertEqual(snap_name("Home Europa"), "Home Europa")

    def test_single_token_rejected(self):
        from mib.lexicon import snap_name
        self.assertIsNone(snap_name("Solo"))


class OutputPriorSafety(unittest.TestCase):
    """Priors improve the serialized row and never touch a decision."""

    def test_fills_only_empty_slots(self):
        from mib.output import apply_priors
        priors = {"fee_status": "paid", "visa_class": "MED-3"}
        filled = apply_priors(
            {"fee_status": "unknown", "visa_class": "XW-1"}, priors
        )
        self.assertEqual(filled["fee_status"], "paid")
        self.assertEqual(filled["visa_class"], "XW-1")  # evidence untouched

    def test_prior_cannot_suppress_a_guardrail_demotion(self):
        """The guardrail sees the evidence row, so a filled fee cannot hide."""
        from mib.output import apply_priors
        from mib.policy import emitted_guardrail
        from mib.record import Record

        record = Record(case_id="MIB-000001")
        evidence_row = {
            "adjudication": "APPROVED", "risk_flags": "none",
            "visa_class": "XW-1", "sponsor_id": "SPN-1234",
            "home_world": "Proxima-b", "fee_status": "unknown",
        }
        demotion = emitted_guardrail(evidence_row, record)
        self.assertIsNotNone(demotion)
        self.assertEqual(demotion[0], "NEEDS_REVIEW")
        # Filling afterwards must not change what the guardrail already decided.
        filled = apply_priors(evidence_row, {"fee_status": "paid"})
        self.assertEqual(filled["fee_status"], "paid")


if __name__ == "__main__":
    unittest.main()
