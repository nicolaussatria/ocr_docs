"""Template registry. To support a new receipt layout, add a class with
`doc_type`, `category`, `score(segment) -> float` and `parse(segment, ctx) -> Draft`."""
from __future__ import annotations

import re

from ..layout import Segment
from .base import Ctx, Draft
from .fuel import FuelRetail, FuelSubsidized
from .other import CashVoucher, Generic, Kwitansi, Parking, Toll
from .medical import OpticalForm
from .mobile import MobileTopup
from .receipt import Receipt, money_in

# the layout-specific templates outscore `Receipt` (any receipt with labelled money rows),
# which outscores `Generic` (anything else)
REGISTRY = [FuelSubsidized(), FuelRetail(), Toll(), Parking(), Kwitansi(), CashVoucher(), OpticalForm(), MobileTopup(), Receipt(),
            Generic()]

# fields that must be present and trusted for a receipt to count as verified
KEY_FIELDS = {
    "fuel_subsidized": ["amountPaid", "transactionDate"],
    "fuel_retail": ["totalPrinted", "transactionDate"],
    "toll": ["fare", "transactionDate"],
    "parking": ["fee", "transactionDate"],
    "kwitansi": ["amount", "transactionDate"],
    "fuel_cash_voucher": ["amount", "transactionDate"],
    "mobile_topup": ["amountPaid", "nominal", "adminFee", "transactionDate", "phoneNumber", "transactionStatus"],
    "receipt": ["total", "transactionDate"],
    "optical": ["amount", "transactionDate"],
    "unknown": ["amount", "transactionDate"],
}


def assemble(segments: list[Segment]) -> list[Segment]:
    """Put a single document back together.

    An invoice, or a photo of one receipt, comes out of the layout pass in pieces:
    letterhead, header table, lines, a stray column heading. When exactly one piece
    carries the money and another brings no amount of its own, they are the same
    document. Several receipts on a page are never merged here - each has amounts the
    others lack - and a piece with amounts nobody else has stays on its own for review."""
    segments = _join_form(segments)
    bodies = [s for s in segments if classify(s)[0].doc_type != "unknown"]
    if len(bodies) != 1 or not isinstance(classify(bodies[0])[0], Receipt):
        return segments
    body = bodies[0]

    def priced(seg: Segment) -> set[int]:
        """Amounts a piece states: a number in a cell of its own, or one with thousands separators.
        A postcode in an address, a year or a quantity like '1.0000' is not a price."""
        out = set()
        for ln in seg.lines:
            alone = re.fullmatch(r"(?i)[\s:\-]*(rp\.?)?[\s\d.,\-]+", ln.text) is not None
            out |= {abs(v) for v, raw, approx in money_in(ln.text) if not approx and (alone or re.search(r"[.,]\d{3}", raw))}
        return out

    known, out = {abs(v) for ln in body.lines for v, *_ in money_in(ln.text)}, []
    for seg in segments:
        if seg is body or not priced(seg) <= known:
            out.append(seg)
        else:
            body.units = body.units + seg.units
    body.units.sort(key=lambda u: (u.y0, u.x0))
    return out


def _join_form(segments: list[Segment]) -> list[Segment]:
    """A filled-in form is one paper: the date and number written top right, or the second half of its
    table, come out as pieces nobody recognises. Beside the one form on the page, at its height, they are its."""
    forms = [s for s in segments if isinstance(classify(s)[0], OpticalForm)]
    if len(forms) != 1:
        return segments
    form, out = forms[0], []
    fx0, fy0, fx1, fy1 = form.bbox
    for seg in segments:
        x0, y0, x1, y1 = seg.bbox
        beside = y0 < fy1 and y1 > fy0 and max(x0 - fx1, fx0 - x1) <= 0.6 * (fx1 - fx0)
        if seg is not form and beside and classify(seg)[0].doc_type == "unknown":
            form.units = sorted(form.units + seg.units, key=lambda u: (u.y0, u.x0))
        else:
            out.append(seg)
    return out


def classify(seg: Segment):
    scored = sorted(((t.score(seg), i, t) for i, t in enumerate(REGISTRY)), key=lambda x: (-x[0], x[1]))
    return scored[0][2], scored[0][0]


__all__ = ["REGISTRY", "KEY_FIELDS", "assemble", "classify", "Ctx", "Draft"]
