"""Pertamina SPBU receipts.

* FuelSubsidized – 'Informasi Harga BBM' layout (Pertalite/Solar with subsidy block).
  Five arithmetic relations tie seven numbers together, so a single misread digit
  is detected and usually repaired from the others.
* FuelRetail     – 'Harga/Liter / Volume / Total Harga' layout (Pertamax, Dexlite ...).
  Also detects a handwritten amount written over a struck-through printed total.
"""
from __future__ import annotations

import re

from rapidfuzz import fuzz, process

from ..layout import Segment, Unit
from ..normalize import clean, compact, find_datetime, parse_amounts, parse_int, parse_volume, to_iso, volume_decimals
from .base import (FV, Cand, Ctx, Draft, Rel, apply_solution, amount_cands, failed, merchant_header, refine_candidates,
                   repair_candidates, solve, text_field, unit_conf)

PRODUCTS = ["PERTALITE", "PERTAMAX", "PERTAMAX TURBO", "PERTAMAX GREEN 95", "DEXLITE", "PERTAMINA DEX",
            "BIO SOLAR", "SOLAR", "PERTAMAX RACING"]


def norm_product(text: str | None) -> str | None:
    if not text:
        return None
    t = re.sub(r"[^A-Z0-9 ]", "", clean(text).upper()).strip()
    hit = process.extractOne(t, PRODUCTS, scorer=fuzz.WRatio, score_cutoff=80)
    return hit[0] if hit else t or None


def _vol_tol(dec: int) -> float:
    return 10 ** (-max(dec, 1))  # displayed volume may be truncated, not rounded


def fuel_common(d: Draft, seg: Segment) -> None:
    for k, fv in merchant_header(seg).items():
        d.set(k, fv)
    full = seg.full_text
    u = seg.first("shift")
    shift = parse_int(u.value_text) if u else None
    if shift is None:
        m = re.search(r"(?i)shi[fl1t]+\s*[:.]?\s*(\d)", full)
        shift = int(m.group(1)) if m else None
    if shift is not None:
        d.set("shift", FV(shift, unit_conf(u) if u else 0.8, lines=list(u.lines) if u else []))
    u = seg.first("transNo")
    m = re.search(r"(\d{5,})", u.value_text) if u else None
    if not m:
        m = re.search(r"(?i)no\.?\s*[ti1l]rans\S*\s*[:.]?\s*(\d{5,})", full)
    if m:
        d.set("transactionNo", FV(m.group(1), unit_conf(u) if u else 0.85, lines=list(u.lines) if u else []))
    # date/time: prefer the 'Waktu' unit, otherwise anything that looks like a date
    dt_units = seg.get("datetime")
    src = dt_units[0].value_text if dt_units else ""
    dd, tt, dconf = find_datetime(src)
    lines = list(dt_units[0].value_lines or dt_units[0].lines) if dt_units else []
    if dd is None:
        for u in seg.units:
            dd, tt, dconf = find_datetime(u.text)
            if dd:
                lines = list(u.lines)
                break
    if dd:
        conf = min(dconf, min((ln.score for ln in lines), default=0.9))
        d.set("transactionDate", FV(dd.isoformat(), conf, raw=src or None, lines=lines))
        if tt:
            d.set("transactionTime", FV(tt.isoformat(), conf, lines=lines))
        d.set("transactionAt", FV(to_iso(dd, tt), conf, lines=lines))
    u = seg.first("pump")
    p = parse_int(u.value_text) if u else None
    if p is not None:
        d.set("pump", FV(p, unit_conf(u), lines=list(u.lines)))
    d.set("operator", text_field(seg.first("operator")))
    pf = text_field(seg.first("product"))
    if pf:
        pf.value = norm_product(pf.value)
        d.set("product", pf)
    if re.search(r"(?m)^\s*CASH\b", full, re.I):
        d.set("paymentMethod", FV("CASH", 0.95))


def _volume_cand(u: Unit | None) -> tuple[list[Cand], int]:
    if u is None:
        return [], 2
    v = parse_volume(u.value_text)
    return ([Cand(v, unit_conf(u), "ocr", u.value_text, list(u.lines))] if v else []), volume_decimals(u.value_text)


class FuelSubsidized:
    doc_type, category = "fuel_subsidized", "fuel"
    KEYS = {"priceNonSubsidized", "subsidy", "totalNonSubsidized", "paid"}

    def score(self, seg: Segment) -> float:
        keys = {u.key for u in seg.units if u.key}
        s = len(keys & self.KEYS) + (1.0 if "SUBSIDI" in compact(seg.full_text) else 0.0)
        return s if s >= 2 else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="amountPaid")
        fuel_common(d, seg)
        units = self._units(seg)
        vol_c, dec = _volume_cand(units["volumeLiters"])
        tol = _vol_tol(dec)
        cands = {k: amount_cands(u) for k, u in units.items() if k != "volumeLiters"}
        cands["volumeLiters"] = vol_c
        # a label the OCR split or missed leaves its number orphaned: offer unlabelled
        # amounts to the empty fields and let the arithmetic decide where (if anywhere) they fit
        loose = [Cand(a.value, u.score * 0.7, "ocr", u.text, list(u.lines))
                 for u in seg.units if not u.label and not u.kind
                 for a in parse_amounts(u.text) if a.value and a.value >= 1000 and not a.approx
                 and len(re.sub(r"[\d.,\s:Rp]", "", u.text)) <= 1]
        for k in cands:
            if k != "volumeLiters" and not cands[k]:
                cands[k] = list(loose)
        rels = [
            Rel("PRICE_MINUS_SUBSIDY_EQ_PRICE", ("priceNonSubsidized", "subsidyPerLiter", "pricePerLiter"),
                lambda v: abs(v["priceNonSubsidized"] - v["subsidyPerLiter"] - v["pricePerLiter"]), True,
                "Harga Non Subsidi - Subsidi/liter = Harga Jual",
                {"pricePerLiter": lambda v: v["priceNonSubsidized"] - v["subsidyPerLiter"],
                 "priceNonSubsidized": lambda v: v["pricePerLiter"] + v["subsidyPerLiter"],
                 "subsidyPerLiter": lambda v: v["priceNonSubsidized"] - v["pricePerLiter"]}),
            Rel("TOTAL_MINUS_SUBSIDY_EQ_PAID", ("totalNonSubsidized", "subsidyTotal", "amountPaid"),
                lambda v: abs(v["totalNonSubsidized"] - v["subsidyTotal"] - v["amountPaid"]) / 2, True,
                "Tanpa Subsidi - Subsidi Pemerintah = Dibayar Konsumen",
                {"amountPaid": lambda v: v["totalNonSubsidized"] - v["subsidyTotal"],
                 "totalNonSubsidized": lambda v: v["amountPaid"] + v["subsidyTotal"],
                 "subsidyTotal": lambda v: v["totalNonSubsidized"] - v["amountPaid"]}),
            Rel("VOLUME_X_PRICE_EQ_PAID", ("volumeLiters", "pricePerLiter", "amountPaid"),
                lambda v: abs(v["volumeLiters"] * v["pricePerLiter"] - v["amountPaid"]) / (tol * v["pricePerLiter"] + 2),
                False, "Volume x Harga Jual ~ Dibayar Konsumen"),
            Rel("VOLUME_X_NONSUB_EQ_TOTAL", ("volumeLiters", "priceNonSubsidized", "totalNonSubsidized"),
                lambda v: abs(v["volumeLiters"] * v["priceNonSubsidized"] - v["totalNonSubsidized"])
                / (tol * v["priceNonSubsidized"] + 2), False, "Volume x Harga Non Subsidi ~ Tanpa Subsidi"),
            Rel("VOLUME_X_SUBSIDY_EQ_SUBSIDY_TOTAL", ("volumeLiters", "subsidyPerLiter", "subsidyTotal"),
                lambda v: abs(v["volumeLiters"] * v["subsidyPerLiter"] - v["subsidyTotal"])
                / (tol * v["subsidyPerLiter"] + 2), False, "Volume x Subsidi/liter ~ Subsidi Pemerintah"),
        ]
        assign = solve(cands, rels)
        probe = Draft(self.doc_type, self.category, seg)
        apply_solution(probe, cands, assign, rels)
        bad = failed(probe) | {k for k in cands if k not in probe.fields}
        if bad and ctx.refine:
            for k, extra in refine_candidates(ctx, units, sorted(bad - {"volumeLiters"})).items():
                cands[k] = cands.get(k, []) + extra
            assign = solve(cands, rels)
        apply_solution(d, cands, assign, rels, cast={"volumeLiters": float, **{k: int for k in cands if k != "volumeLiters"}})
        if "volumeLiters" not in d.fields and d.val("amountPaid") and d.val("pricePerLiter"):
            d.set("volumeLiters", FV(round(d.val("amountPaid") / d.val("pricePerLiter"), 2), 0.8, "derived"))
        return d

    @staticmethod
    def _units(seg: Segment) -> dict[str, Unit | None]:
        price, tns, hns = seg.first("price"), seg.first("totalNonSubsidized"), seg.first("priceNonSubsidized")
        subs = sorted(seg.get("subsidy"), key=lambda u: u.y0)
        per_liter = total = None
        for u in subs:
            if price is not None:
                is_pl = u.y0 < price.y0
            elif tns is not None:
                is_pl = u.y0 < tns.y0
            elif hns is not None:
                is_pl = abs(u.y0 - hns.y1) < 2 * u.h
            else:
                is_pl = u is subs[0] and len(subs) == 2
            if is_pl and per_liter is None:
                per_liter = u
            elif not is_pl and total is None:
                total = u
        return {"priceNonSubsidized": hns, "subsidyPerLiter": per_liter, "pricePerLiter": price,
                "totalNonSubsidized": tns, "subsidyTotal": total, "amountPaid": seg.first("paid"),
                "volumeLiters": seg.first("volume")}


class FuelRetail:
    doc_type, category = "fuel_retail", "fuel"
    KEYS = {"unitPrice", "totalPrice", "product", "volume"}

    def score(self, seg: Segment) -> float:
        keys = {u.key for u in seg.units if u.key}
        if "subsidy" in keys or "priceNonSubsidized" in keys:
            return 0.0
        s = len(keys & self.KEYS) + (0.5 if re.search(r"PERTAMAX|DEXLITE|PERTAMINADEX", compact(seg.full_text)) else 0)
        return s if {"unitPrice", "totalPrice"} & keys and s >= 2 else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="totalPrinted")
        fuel_common(d, seg)
        price_u, total_u, vol_u = seg.first("unitPrice"), seg.first("totalPrice"), seg.first("volume")
        vol_c, dec = _volume_cand(vol_u)
        tol = _vol_tol(dec)
        price_c = amount_cands(price_u, min_value=1000)
        line_amts = amount_cands(total_u, min_value=1000)
        # stray amount boxes inside the receipt (stamped / handwritten corrections)
        strays = []
        for u in seg.units:
            if u.label or u.kind:
                continue
            amts = [a for a in parse_amounts(u.text) if a.value and a.value >= 1000]
            if len(amts) == 1 and len(re.sub(r"[\d.,\sRrPp]", "", u.text)) <= 2:
                strays.append(Cand(amts[0].value, u.score * 0.8, "ocr", u.text, list(u.lines)))
        total_c = list(line_amts) + strays
        rel = Rel("VOLUME_X_PRICE_EQ_TOTAL", ("volumeLiters", "pricePerLiter", "totalPrinted"),
                  lambda v: abs(v["volumeLiters"] * v["pricePerLiter"] - v["totalPrinted"]) / (tol * v["pricePerLiter"] + 2),
                  False, "Volume x Harga/Liter ~ Total Harga (printed)")
        cands = {"volumeLiters": vol_c, "pricePerLiter": price_c, "totalPrinted": total_c}
        if vol_c and price_c:
            # a struck-through printed total is often misread by one digit (319,000 -> 349.000)
            cands["totalPrinted"] += repair_candidates(
                line_amts, lambda t: rel.ok({"volumeLiters": vol_c[0].value, "pricePerLiter": price_c[0].value,
                                             "totalPrinted": t}))
        assign = solve(cands, [rel])
        if (assign.get("totalPrinted") is None or not rel.ok({k: c.value for k, c in assign.items() if c})) \
                and ctx.refine and total_u is not None:
            extra = refine_candidates(ctx, {"totalPrinted": total_u}, ["totalPrinted"], min_value=1000)
            cands["totalPrinted"] += extra.get("totalPrinted", [])
            assign = solve(cands, [rel])
        apply_solution(d, cands, assign, [rel], cast={"volumeLiters": float, "pricePerLiter": int, "totalPrinted": int})
        printed = d.val("totalPrinted")
        # handwritten override: other amounts on the total line / stray boxes
        def is_printed_reading(v: int) -> bool:
            if printed is None:
                return False
            if v == printed:
                return True
            sv, sp = str(v), str(printed)
            one_digit = len(sv) == len(sp) and sum(a != b for a, b in zip(sv, sp)) == 1
            near = d.val("volumeLiters") and d.val("pricePerLiter") and rel.ok(
                {"volumeLiters": d.val("volumeLiters"), "pricePerLiter": d.val("pricePerLiter"), "totalPrinted": v})
            return one_digit or bool(near)

        others = [c for c in total_c if not is_printed_reading(int(c.value))]
        if others:
            vals = {}
            for c in others:
                vals.setdefault(c.value, []).append(c)
            best_v, group = max(vals.items(), key=lambda kv: (len(kv[1]), max(ln.score for c in kv[1] for ln in c.lines)))
            agree = len(group) >= 2
            conflict = len(vals) > 1
            conf = 0.8 if agree and not conflict else (0.5 if conflict else 0.6)
            d.set("amountClaimed", FV(int(best_v), conf, "ocr", ", ".join(c.raw for c in others),
                                      [ln for c in group for ln in c.lines]))
            d.handwritten = True
            readings = " / ".join(f"{int(v):,}" for v in vals)
            d.review.append(
                f"Handwritten amount {int(best_v):,} written over printed total"
                + (f" {printed:,}" if printed else "") + (f" (OCR readings: {readings})" if conflict else "")
                + " - confirm which amount is claimed")
        return d
