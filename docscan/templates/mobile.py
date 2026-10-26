"""Printed m-Commerce mobile top-ups (pulsa), including bank admin fees."""
from __future__ import annotations

import re

from ..layout import Segment, Unit, match_label
from ..normalize import clean, compact, parse_printed_idr
from .base import FAIL, PASS, WARN, Cand, Check, Ctx, Draft, FV, settle, text_field, unit_conf
from .other import _set_datetime

_AMOUNTS = {"nominal": "topupNominal", "adminFee": "adminFee", "amountPaid": "paymentTotal"}
_PHONE = re.compile(r"(?:08\d{8,11}|\+?628\d{7,10})")
_PRODUCT = re.compile(r"(?:TELKOMSEL|SIMPATI|AS|INDOSAT|IM3|MENTARI|XL|AXIS|TRI|THREE|SMARTFREN)\s*PULSA", re.I)
_STATES = {
    "BERHASIL": "success", "SUKSES": "success", "SUCCESS": "success",
    "GAGAL": "failed", "TIDAKBERHASIL": "failed", "FAILED": "failed",
    "PENDING": "pending", "DALAMPROSES": "pending", "DIPROSES": "pending",
}


def _amount(u: Unit | None, ctx: Ctx) -> FV | None:
    if u is None:
        return None
    value = parse_printed_idr(u.value_text)
    direct = FV(value, unit_conf(u), raw=u.value_text, lines=list(u.lines)) if value is not None else None
    if not ctx.refine or (direct is not None and direct.confidence >= 0.85):
        return direct
    # Use multiple image variants only when the original amount is unreadable.
    # A single new reading cannot establish a value; refined values need review.
    candidates = []
    for text, confidence in ctx.reread(u.value_lines or u.lines):
        hit = match_label(text)
        body = hit.inline if hit and hit.key == u.key else text
        value = parse_printed_idr(body)
        if value is not None:
            candidates.append(Cand(value, confidence, "ocr_refined", body, list(u.lines)))
    result = settle(candidates)
    if result is None or result[1] < 0.8:
        return direct
    c, _ = result
    if direct is not None and direct.value != c.value:
        # Preserve disagreement for a person to resolve; do not silently replace it.
        direct.confidence = min(direct.confidence, 0.49)
        return direct
    return FV(int(c.value), min(c.conf, 0.84), c.source, c.raw, c.lines)


class MobileTopup:
    doc_type, category = "mobile_topup", "pulsa"

    def score(self, seg: Segment) -> float:
        keys = {u.key for u in seg.units}
        product = any(_PRODUCT.fullmatch(clean(u.text)) for u in seg.units)
        commerce = any(compact(u.text) == "MCOMMERCE" for u in seg.units)
        amounts = len(keys & set(_AMOUNTS.values()))
        return 4.0 + amounts if product and (commerce or amounts >= 2) else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="amountPaid")
        for field, label in _AMOUNTS.items():
            d.set(field, _amount(seg.first(label), ctx))
        for u in seg.units:
            if _set_datetime(d, u.text, list(u.lines)):
                break

        product = next((u for u in seg.units if _PRODUCT.fullmatch(clean(u.text))), None)
        if product is not None:
            name = clean(product.text).upper()
            d.set("product", FV(name, product.score, raw=product.text, lines=list(product.lines)))
            d.set("merchantName", FV(re.sub(r"\s*PULSA$", "", name), product.score,
                                     raw=product.text, lines=list(product.lines)))
        phone = seg.first("phoneNumber")
        candidates = [phone] if phone is not None else [
            u for u in seg.units if not u.label and product is not None and u.y0 > product.y0
        ]
        for u in candidates:
            value = re.sub(r"[\s-]", "", u.value_text if u.label else u.text)
            if _PHONE.fullmatch(value):
                d.set("phoneNumber", FV(value, unit_conf(u), raw=u.text, lines=list(u.lines)))
                break
        serial = seg.first("topupSerial")
        if serial is not None:
            value = re.sub(r"\s", "", serial.value_text).lstrip("=")
            if re.fullmatch(r"\d{8,30}", value):
                d.set("transactionNo", FV(value, unit_conf(serial), raw=serial.text, lines=list(serial.lines)))
        d.set("purpose", text_field(seg.first("description"), upper=False))

        states = [(u, _STATES[compact(u.text)]) for u in seg.units if compact(u.text) in _STATES]
        if states:
            # An explicit failure/pending message takes precedence over a success line.
            u, status = next(((u, s) for u, s in states if s != "success"), states[0])
            d.set("transactionStatus", FV(status, u.score, raw=u.text, lines=list(u.lines)))
            d.checks.append(Check("TRANSACTION_SUCCESS", PASS if status == "success" else FAIL,
                                  f"Transaction status: {status}", ["transactionStatus"]))
            if status != "success":
                d.amount_key = None   # failed/pending payments are not expenses
        else:
            d.checks.append(Check("TRANSACTION_SUCCESS", WARN, "Transaction success status is unreadable",
                                  ["transactionStatus"]))

        keys = list(_AMOUNTS)
        nominal, fee, paid = (d.val(k) for k in keys)
        if paid is None and nominal is not None and fee is not None and nominal > 0:
            confidence = min(d.fields["nominal"].confidence, d.fields["adminFee"].confidence, 0.84)
            d.set("amountPaid", FV(nominal + fee, confidence, "derived"))
            d.checks.append(Check("NOMINAL_PLUS_ADMIN_EQ_PAID", WARN,
                                  "Paid total calculated from nominal + admin fee; confirm printed total", keys))
        elif None in (nominal, fee, paid):
            d.checks.append(Check("NOMINAL_PLUS_ADMIN_EQ_PAID", WARN,
                                  "Cannot check nominal + admin fee = paid total: missing amount", keys))
        else:
            good = nominal > 0 and paid > 0 and nominal + fee == paid
            d.checks.append(Check("NOMINAL_PLUS_ADMIN_EQ_PAID", PASS if good else FAIL,
                                  f"Nominal {nominal:,} + admin fee {fee:,} = paid total {paid:,}", keys))
        return d
