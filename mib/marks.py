"""Read adjudicator stamps and cancellation marks from the PDF vector layer.

These packets draw stamps and strike-throughs as vector graphics, not raster
ink. Reading them from `page.get_drawings()` is exact and costs nothing: no
rasterization, no OCR, no threshold to tune. On the public training corpus a
green stamp is APPROVED in 18/18 packets and a blue stamp is NEEDS_REVIEW in
23/23; red is DENIED in 79% alone, the shortfall being the documented
rescinded-denial trap, which needs the cancellation geometry to resolve.

This is evidence, not a decision. It feeds the same precedence rules as any
other visible mark, and a stamp is only as good as the absence of a later
signed note overriding it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pymupdf

APPROVED, DENIED, NEEDS_REVIEW = "APPROVED", "DENIED", "NEEDS_REVIEW"

# Stamp ink is saturated primary colour; form rules and text are gray/black.
_STAMP_COLORS = {
    "GREEN": APPROVED,
    "BLUE": NEEDS_REVIEW,
    "RED": DENIED,
}

# A cancellation stroke is long, thin, and roughly horizontal.
_STRIKE_MAX_THICKNESS = 3.0
_STRIKE_MIN_LENGTH = 40.0


def classify_color(color) -> str | None:
    if not color or len(color) < 3:
        return None
    red, green, blue = float(color[0]), float(color[1]), float(color[2])
    if red > 0.6 and green < 0.4 and blue < 0.4:
        return "RED"
    if green > 0.35 and red < 0.4 and blue < 0.4:
        return "GREEN"
    if blue > 0.6 and red < 0.4 and green < 0.4:
        return "BLUE"
    return None


@dataclass
class Mark:
    color: str
    verdict: str
    page: int
    rect: tuple[float, float, float, float]
    is_strike: bool


@dataclass
class MarkEvidence:
    marks: list[Mark] = field(default_factory=list)

    @property
    def stamps(self) -> list[Mark]:
        return [m for m in self.marks if not m.is_strike]

    @property
    def strikes(self) -> list[Mark]:
        return [m for m in self.marks if m.is_strike]

    def verdicts(self) -> set[str]:
        return {m.verdict for m in self.stamps}

    def struck(self, mark: Mark) -> bool:
        """Whether a cancellation stroke crosses this stamp.

        A denial stamp crossed out by a later signed approval is not
        disqualifying (FIELD_MANUAL.md, "Known Document Traps"), so a struck
        stamp must not be read as a live verdict.
        """
        box = pymupdf.Rect(mark.rect)
        for strike in self.strikes:
            if strike.page != mark.page:
                continue
            if not (box & pymupdf.Rect(strike.rect)).is_empty:
                return True
        return False

    def live_verdict(self) -> tuple[str, float] | None:
        """The single uncancelled stamp verdict, with a purity-based weight.

        Returns None when there is no stamp, when every stamp is struck, or
        when uncancelled stamps disagree - a contradiction is evidence of a
        contested packet, which the policy should route to review rather than
        resolve by guessing.
        """
        live = [m for m in self.stamps if not self.struck(m)]
        if not live:
            return None
        verdicts = {m.verdict for m in live}
        if len(verdicts) != 1:
            return None
        verdict = verdicts.pop()
        # Weights are the observed purity of each colour on public training
        # data; red is lower because it carries the rescinded-denial trap.
        return verdict, (0.97 if verdict in (APPROVED, NEEDS_REVIEW) else 0.85)


def read_marks(document: pymupdf.Document) -> MarkEvidence:
    evidence = MarkEvidence()
    for index, page in enumerate(document):
        try:
            drawings = page.get_drawings()
        except Exception:  # noqa: BLE001 - a damaged content stream is not fatal
            continue
        for item in drawings:
            color = classify_color(item.get("color") or item.get("fill"))
            if not color:
                continue
            rect = item.get("rect")
            if rect is None:
                continue
            width, height = rect.x1 - rect.x0, rect.y1 - rect.y0
            is_strike = (
                min(width, height) < _STRIKE_MAX_THICKNESS
                and max(width, height) >= _STRIKE_MIN_LENGTH
            )
            evidence.marks.append(
                Mark(
                    color=color,
                    verdict=_STAMP_COLORS[color],
                    page=index + 1,
                    rect=(rect.x0, rect.y0, rect.x1, rect.y1),
                    is_strike=is_strike,
                )
            )
    return evidence
