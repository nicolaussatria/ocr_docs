"""Conservative date/volume readings for handwritten Indonesian forms.

Keep these repairs out of the general parser: dotted form leaders and a printed
`20 ...` year prefix have different semantics from prices and timestamps.
"""
from __future__ import annotations

import re
from collections import defaultdict
from datetime import date

from ..normalize import clean, find_dates, to_iso
from ..ocr import Line
from .base import FV, Draft


def handwritten_dates(text: str) -> set[date]:
    """Require a complete year; accept leaders/spaces between its printed digits.

    DMY order is explicit. No guessed century, current year, month/day swapping,
    or missing separator between day and month (a signature can look like a day).
    """
    text = clean(text)
    year = r"(?:19|20)[.\s]*\d[.\s]*\d(?!\d)"
    out = set()
    # A form may read `4.07 20..26` or `4.0720.26` when the leader vanishes.
    pattern = rf"(?<![\d/.\-])(\d{{1,2}})\s*[/.\-]\s*(\d{{1,2}})[.\s/\-]*({year})"
    for m in re.finditer(pattern, text):
        try:
            out.add(date(int(re.sub(r"\D", "", m[3])), int(m[2]), int(m[1])))
        except ValueError:
            pass
    # Retain textual month names and ISO dates supported by the shared parser.
    for found in find_dates(text):
        if re.search(r"[A-Za-z]", found.raw) or re.match(r"(?:19|20)\d{2}[-/.]", found.raw):
            out.add(found.value)
    return out


def set_handwritten_date(d: Draft, readings: list[tuple[str, float, str, list[Line]]]) -> None:
    """Only select a date if credible readings agree. Variants are not independent proof."""
    choices = defaultdict(list)
    for text, score, source, lines in readings:
        if score < 0.55:
            continue
        for value in handwritten_dates(text):
            choices[value].append((text, score, source, lines))
    if not choices:
        return
    if len(choices) > 1:
        values = sorted(v.isoformat() for v in choices)
        d.set("transactionDateCandidates", FV(values, 0.4, "ocr_refined",
              raw=" | ".join(dict.fromkeys(r[0] for rs in choices.values() for r in rs)),
              lines=list(dict.fromkeys(ln for rs in choices.values() for r in rs for ln in r[3]))))
        d.review.append("Handwritten date has competing readings: " + ", ".join(values) + " - confirm")
        return
    value, votes = next(iter(choices.items()))
    # Once an explicit date establishes the separators, an otherwise identical
    # compact reading can back it up. `210/2026` alone never establishes 2/10.
    compact_day_month = f"{value.day}{value.month:02d}"
    for text, score, source, lines in readings:
        if score < 0.65:
            continue
        if re.search(rf"(?<!\d){compact_day_month}\s*/\s*{value.year}(?!\d)", text):
            votes.append((text, score, source, lines))
    # An isolated cleanup hallucination must not become a date.
    if len(votes) < 2 and not any(r[2] == "ocr" for r in votes):
        return
    text, score, source, lines = max((r for r in votes if value in handwritten_dates(r[0])), key=lambda r: r[1])
    confidence = min(score, 0.6)
    d.set("transactionDate", FV(value.isoformat(), confidence, source, text, lines))
    d.set("transactionAt", FV(to_iso(value, None), confidence, source, text, lines))


def handwritten_volume(text: str) -> float | None:
    """An isolated volume, with its decimal separator preserved. Never turn 35 into 3.5."""
    text = clean(text).strip(" .")
    match = re.fullmatch(r"(\d{1,3}(?:[.,]\d{1,3})?)\s*(?:[lL]|[lL]iter)?", text)
    if not match:
        return None
    value = float(match[1].replace(",", "."))
    return value if 0 < value <= 200 else None


def choose_volume(readings: list[tuple[str, float, str, list[Line]]]) -> FV | None:
    # Cleanup often removes the decimal point. Two explicit readings can anchor
    # otherwise identical digits, but the assumed fuel price never supplies it.
    decimals = defaultdict(list)
    for text, score, source, lines in readings:
        value = handwritten_volume(text)
        if value is not None and score >= 0.65 and re.search(r"\d[.,]\d", text):
            decimals[re.sub(r"\D", "", text)].append((value, score))
    anchors = {digits: votes[0][0] for digits, votes in decimals.items()
               if len(votes) >= 2 and max(sc for _, sc in votes) >= 0.75
               and len({v for v, _ in votes}) == 1}
    choices = defaultdict(list)
    for text, score, source, lines in readings:
        value = handwritten_volume(text)
        if source == "ocr_refined" and re.fullmatch(r"\d+", text.strip()):
            value = anchors.get(text.strip(), value)
        if value is not None and score >= 0.65:
            choices[value].append((text, score, source, lines))
    if not choices:
        return None
    value, votes = max(choices.items(), key=lambda kv: sum(r[1] for r in kv[1]))
    support = sum(r[1] for r in votes)
    total = sum(r[1] for rs in choices.values() for r in rs)
    # Provenance must point to a reading that actually contains the selected value.
    text, score, source, lines = max((r for r in votes if handwritten_volume(r[0]) == value), key=lambda r: r[1])
    if support / total < 0.7:
        return None
    if len(votes) < 2 and source != "ocr" and not (score >= 0.85 and re.search(r"\d[.,]\d", text)):
        return None
    return FV(value, min(score, 0.85), source, text, lines)
