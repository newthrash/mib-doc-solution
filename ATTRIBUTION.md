# Attribution

This solution is an independent implementation, but several design ideas come
from public, MIT-licensed solutions to the same challenge, studied before this
repository was written. Credit where the idea originated:

- **zubalr / mib-intake** (MIT): the decision-path -> empirical-distribution ->
  expected-value adjudication structure, and the corpus-relative inference of
  revoked sponsors (frequency outliers in the scored corpus) and date context
  rather than hardcoded train-mined constants.
- **tylergibbs1 / mib-doc-challenge-solution** (MIT): the span-level visibility
  filter (font size, crop-box overlap, luminance contrast), the bounded OCR
  escalation ladder, and the one-way emitted-fields guardrail that lets output
  completion improve extraction without ever creating an approval.
- **strobl / mib-doc-solution** (MIT): the render-first trust-boundary framing
  and the fail-closed finalizer discipline.

Mined policy constants (three additional revoked sponsors, the Wolf-1061c
embargo) are inferred from the public training labels, which the challenge PRD
explicitly invites; they are independently re-derived at score time from the
scored corpus where possible.

No code was copied from any competitor's repository. No answer-key or hidden
text channel is used for field values or decisions.
