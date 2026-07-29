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


if __name__ == "__main__":
    unittest.main()
