"""Vector-layer stamp and cancellation reading."""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.marks import Mark, MarkEvidence, classify_color


class ColorClassification(unittest.TestCase):
    def test_primary_stamp_colors(self):
        self.assertEqual(classify_color((1.0, 0.0, 0.0)), "RED")
        self.assertEqual(classify_color((0.0, 0.5, 0.0)), "GREEN")
        self.assertEqual(classify_color((0.0, 0.0, 1.0)), "BLUE")

    def test_form_rules_and_text_are_not_stamps(self):
        for gray in ((0, 0, 0), (0.5, 0.5, 0.5), (1, 1, 1), (0.8, 0.8, 0.75)):
            self.assertIsNone(classify_color(gray), gray)
        self.assertIsNone(classify_color(None))


def stamp(color, verdict, page=1, rect=(340, 89, 501, 156), is_strike=False):
    return Mark(color=color, verdict=verdict, page=page, rect=rect, is_strike=is_strike)


class VerdictResolution(unittest.TestCase):
    def test_single_stamp_resolves(self):
        evidence = MarkEvidence([stamp("GREEN", "APPROVED")])
        self.assertEqual(evidence.live_verdict()[0], "APPROVED")

    def test_struck_stamp_is_not_a_live_verdict(self):
        """A denial crossed out by a later signed approval is not a denial."""
        evidence = MarkEvidence([
            stamp("RED", "DENIED"),
            stamp("RED", "DENIED", rect=(330, 120, 520, 121), is_strike=True),
        ])
        self.assertIsNone(evidence.live_verdict())

    def test_disagreeing_stamps_do_not_resolve(self):
        evidence = MarkEvidence([
            stamp("GREEN", "APPROVED"),
            stamp("RED", "DENIED", page=2),
        ])
        self.assertIsNone(evidence.live_verdict())

    def test_strike_on_another_page_does_not_cancel(self):
        evidence = MarkEvidence([
            stamp("RED", "DENIED", page=1),
            stamp("RED", "DENIED", page=3, rect=(330, 120, 520, 121), is_strike=True),
        ])
        self.assertEqual(evidence.live_verdict()[0], "DENIED")

    def test_no_stamps(self):
        self.assertIsNone(MarkEvidence([]).live_verdict())


if __name__ == "__main__":
    unittest.main()
