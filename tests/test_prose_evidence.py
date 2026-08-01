"""Prose evidence: precedence, and the limits on trusting it.

The ordering here is the fragile part. Corrections are applied after the
damage-tag blanking and read only from the native text layer, and both of
those are invisible in the output until a packet happens to combine the
conditions - so they are pinned by test rather than by comment alone.
"""

import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mib.constants import UNKNOWN
from mib.extract import extract_record
from mib.pdfio import Packet, Page


def _packet(native: str, ocr: str = "") -> Packet:
    packet = Packet(case_id="MIB-TEST", pdf_name="MIB-TEST.pdf")
    packet.pages.append(Page(number=1, visible_native=native, ocr_text=ocr))
    return packet


class ProseEvidenceTests(unittest.TestCase):
    def test_compliance_clause_supplies_visa_class(self):
        record = extract_record(_packet(
            "The sponsor acknowledges responsibility for class MED-3 "
            "compliance and immediate repatriation."
        ))
        self.assertEqual(record.visa_class, "MED-3")

    def test_correction_overrides_a_readable_form_field(self):
        # The form is legible and says one thing; the amendment says another.
        # The amendment is the later document, and on the public corpus it
        # matches the label every time, so it must win rather than defer.
        record = extract_record(_packet(
            "Visa Class: MED-3\nManual correction: visa class is XW-1."
        ))
        self.assertEqual(record.visa_class, "XW-1")

    def test_correction_outranks_a_damage_tag(self):
        # These co-occur because the correction is the remedy for the very
        # field the tag reports destroyed. Blanking it would discard the fix.
        record = extract_record(_packet(
            "Sponsor ID: [SPONSOR ID BLANK]\n"
            "Manual correction: sponsor is SPN-4705."
        ))
        self.assertEqual(record.sponsor_id, "SPN-4705")
        self.assertTrue(record.documented_damage)

    def test_correction_is_not_read_from_ocr(self):
        # A garbled clause still matches the pattern, so OCR of these
        # sentences may not overwrite a form field.
        record = extract_record(_packet(
            "Visa Class: MED-3", ocr="Manual correction: visa class is XW-1."
        ))
        self.assertEqual(record.visa_class, "MED-3")

    def test_unreadable_prose_value_is_dropped_not_trusted(self):
        # Vocabulary snapping is the backstop: a capture that is not a real
        # visa class leaves the field alone rather than inventing one.
        record = extract_record(_packet(
            "Manual correction: visa class is QQ-9."
        ))
        self.assertEqual(record.visa_class, UNKNOWN)

    def test_fee_and_name_corrections_apply(self):
        record = extract_record(_packet(
            "Manual correction: fee status is paid.\n"
            "Manual correction: applicant is Oridane Soltari."
        ))
        self.assertEqual(record.fee_status, "paid")
        self.assertEqual(record.applicant_name, "Oridane Soltari")


if __name__ == "__main__":
    unittest.main()
