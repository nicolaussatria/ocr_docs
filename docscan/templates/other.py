"""Toll (e-Toll), parking, kwitansi, SPBU 'bon kontan' and a generic fallback."""
from __future__ import annotations

import re
from datetime import datetime

import numpy as np
from rapidfuzz import fuzz

from ..layout import Segment
from ..normalize import (clean, compact, find_card_numbers, find_datetime, find_partial_date, find_plate, glyph_amount,
                         hand_amount, parse_amounts, parse_duration_minutes, station_code, strip_value, terbilang_to_int,
                         terbilang_vote, to_iso)
from ..ocr import Line
from .base import (FAIL, FV, PASS, SKIP, WARN, Cand, Check, Ctx, Draft, amount_cands, merchant_header, row_zone,
                   seg_slope, settle, text_field, unit_conf)
from .fuel import PRODUCTS, norm_product
from .medical import is_medical, medical_fields
from .handwriting import choose_volume, handwritten_dates, set_handwritten_date

ISSUERS = [("FLAZZ", "BCA Flazz"), ("BCA", "BCA Flazz"), ("MANDIRI", "Mandiri e-Money"), ("EMONEY", "Mandiri e-Money"),
           ("BRIZZI", "BRI Brizzi"), ("BRI", "BRI Brizzi"), ("TAPCASH", "BNI TapCash"), ("BNI", "BNI TapCash"),
           ("JAKCARD", "Bank DKI JakCard"), ("DKI", "Bank DKI JakCard")]


def row_mates(seg: Segment, line: Line, tol: float = 0.45, right_only: bool = True) -> list[Line]:
    out = []
    for ln in seg.lines:
        if ln is line:
            continue
        pred = line.cy + line.slope * (ln.cx - line.cx)
        if abs(ln.cy - pred) <= tol * max(line.h, ln.h) and (not right_only or ln.x0 >= line.x1 - 0.5 * line.h):
            out.append(ln)
    return sorted(out, key=lambda ln: ln.x0)


def issuer_of(text: str) -> str | None:
    c = compact(text)
    for key, name in ISSUERS:
        if key in c:
            return name
    return None


def _set_datetime(d: Draft, text: str, lines: list[Line], prefix: str = "transaction") -> bool:
    dd, tt, conf = find_datetime(text)
    if dd is None:
        return False
    conf = min(conf, min((ln.score for ln in lines), default=0.9))
    d.set(f"{prefix}Date", FV(dd.isoformat(), conf, raw=text, lines=lines))
    if tt:
        d.set(f"{prefix}Time", FV(tt.isoformat(), conf, lines=lines))
    d.set(f"{prefix}At", FV(to_iso(dd, tt), conf, lines=lines))
    return True


def _balance_after_rp(text: str) -> tuple[int | None, str | None]:
    t = clean(text)
    i = t.upper().rfind("RP")
    if i < 0:
        return None, None
    amts = parse_amounts(t[i:])
    if not amts:
        return None, None
    a = amts[0]
    return a.value, a.pattern


# =========================================================================== toll
class Toll:
    doc_type, category = "toll", "toll"
    # The words of an e-toll slip, as the recogniser reads them off a phone photo too: 'l' comes back
    # as '1' or 'I' and 'O' as '0' ('e-To11', 'GO1-1'), a letter goes missing ('No se-i'), and the
    # card line keeps its 'CN:' when the number itself is cut off.
    SIGNS = [r"JASAMARGA", r"ET[O0][L1I]", r"GERBANG", r"N[O0]SER?[I1L]", r"INF[O0]T[O0][L1I]", r"GT[O0]", r"T[O0][L1I]{2}"]
    GOL = r"G[O0][L1I]\s*[-.]?\s*([1-5I])"
    CARD_LINE = r"\bC[NM]\s*[:;.]\s*[0-9OCD]{4}"

    def score(self, seg: Segment) -> float:
        c = compact(seg.full_text)
        s = sum(1.0 for w in self.SIGNS if re.search(w, c))
        s += 1.0 if re.search(self.GOL, seg.full_text, re.I) else 0.0
        s += 1.0 if re.search(self.CARD_LINE, seg.full_text) else 0.0
        return s if s >= 2 else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="fare")
        lines = sorted(seg.lines, key=lambda ln: (ln.y0, ln.x0))
        if any(fuzz.ratio(compact(ln.text), "JASAMARGA") >= 75 for ln in lines):
            d.set("merchantName", FV("JASA MARGA", 0.95))
        g = seg.first("gate")
        if g is not None:
            m = re.search(r"\[([^\]]+)\]?", g.value_text)
            gate = (m.group(1) if m else strip_value(g.value_text)).strip(" []").upper()
            d.set("gate", FV(gate, unit_conf(g), raw=g.value_text, lines=list(g.lines)))
        # GOL line: class, issuer and (usually) the fare
        gol = next((ln for ln in lines if re.search(rf"{self.GOL}|e-?\s*T[o0][l1I]", ln.text, re.I)), None)
        if gol is not None:
            m = re.search(self.GOL, gol.text, re.I)
            if m:
                d.set("vehicleClass", FV(f"Gol {m.group(1).replace('I', '1')}", gol.score, lines=[gol]))
            texts = [gol.text] + [ln.text for ln in row_mates(seg, gol)]
            issuer = issuer_of(" ".join(texts))
            if issuer:
                d.set("paymentMethod", FV(issuer, gol.score, lines=[gol]))
            fare_c = self._fares(" ".join(texts), gol.score, [gol])
            if (not fare_c or gol.score < 0.95) and ctx.refine:
                fare_c += self._zone_fares(seg, [gol] + row_mates(seg, gol), ctx)
            best = settle(fare_c)
            if best:
                c, share = best
                d.set("fare", FV(int(c.value), c.conf, c.source, c.raw, [gol]))
                if share < 0.6:
                    seen = dict.fromkeys(int(x.value) for x in fare_c)
                    d.review.append("Toll fare readings disagree: " + " / ".join(f"{k:,}" for k in seen))
        # card number + balance
        for ln in lines:
            cards = find_card_numbers(ln.text)
            if cards and "cardNumber" not in d.fields:
                d.set("cardNumber", FV(cards[0], ln.score * (1.0 if "?" not in cards[0] else 0.6), raw=ln.text, lines=[ln]))
                bal, pat = _balance_after_rp(ln.text + " " + " ".join(x.text for x in row_mates(seg, ln)))
                if bal is not None:
                    d.set("balanceAfter", FV(bal, ln.score, raw=ln.text, lines=[ln]))
                elif pat:
                    d.fields["balanceAfter"] = FV(None, 0.0, "ocr", pat, [ln])  # pattern kept for the batch solver
        # date / time: take the strongest date on the slip; 'dd/mm/yy' strings are usually
        # booth/shift codes, so they only count when nothing better exists
        best = None
        for ln in lines:
            dd, tt, conf = find_datetime(ln.text)
            if dd and (best is None or conf > best[2]):
                best = (dd, tt, conf, ln)
        partial = next((p_ for p_ in (find_partial_date(ln.text) for ln in lines) if p_), None)
        if best and (best[2] >= 0.9 or partial is None):
            _set_datetime(d, best[3].text, [best[3]])
        if "transactionDate" not in d.fields:
            for ln in lines:
                p = find_partial_date(ln.text)
                if p:
                    d.fields["datePartial"] = FV({"day": p.day, "month": p.month, "year": p.year}, 0.5, raw=p.raw, lines=[ln])
                    _, tt, _ = find_datetime(ln.text.replace(p.raw, " "))
                    if tt:
                        d.set("transactionTime", FV(tt.isoformat(), ln.score * 0.9, lines=[ln]))
                    d.review.append(f"Date partly unreadable ('{p.raw}')")
                    break
        s = seg.first("serialNo")
        if s is not None:
            d.set("serialNo", FV(strip_value(s.value_text), unit_conf(s), lines=list(s.lines)))
        fare = d.val("fare")
        if fare is not None:
            plausible = 500 <= fare <= 2_000_000 and fare % 500 == 0
            d.checks.append(Check("TOLL_FARE_PLAUSIBLE", PASS if plausible else WARN,
                                  "Toll fare is a multiple of Rp500 within 500..2,000,000", ["fare"]))
        return d

    _ISSUER = r"[B38]CA|MANDIRI|BRI|BNI|DKI|FLAZZ|E-?MONEY|BRIZZI|TAPCASH"

    @classmethod
    def _fares(cls, text: str, conf: float, lines, source="ocr") -> list[Cand]:
        t = re.sub(rf"(?i){cls.GOL}", " ", text)
        m = re.search(rf"(?i)({cls._ISSUER}|RP)", t)
        tail = t[m.end():] if m else t
        out = []
        for a in parse_amounts(tail):
            if a.value and a.value >= 500:
                out.append(Cand(a.value, conf * (0.6 if a.approx else 1.0), source, a.raw, list(lines)))
        return out

    @classmethod
    def _zone_fares(cls, seg: Segment, row: list[Line], ctx: Ctx) -> list[Cand]:
        """Re-read the fare row from the image. A highlighter over the amount hides it from
        the text detector, and the double-height font is too narrow for the recogniser
        unless the crop is widened first."""
        gol = row[0]
        slope = seg_slope(seg)
        tall = gol.h / max(float(np.median([ln.h for ln in seg.lines])), 1.0)
        base = float(np.clip(1.5 * tall, 1.5, 3.2))
        stretch = (0.8 * base, base, 1.2 * base)
        left = min(ln.x0 for ln in row) - 0.1 * gol.h
        right = max(seg.bbox[2], max(ln.x1 for ln in row)) + 0.4 * gol.h
        out = []
        # the whole row: the fare is whatever follows the issuer name
        for txt, sc in ctx.read_zone(row_zone(gol, left, right, slope), h=gol.h, stretch=stretch):
            m = None
            for m in re.finditer(rf"(?i){cls._ISSUER}", txt):
                pass
            v, subs = glyph_amount(txt[m.end():]) if m else (None, 0)
            if v is not None and 500 <= v <= 2_000_000 and v % 500 == 0:
                out.append(Cand(v, sc * 0.95 * 0.85 ** subs, "ocr_refined", txt, [gol]))
        # the part right of the printed words on its own, when the detector left it out
        words = [ln for ln in row if re.search(rf"(?i)G[O0][L1I]|e-?\s*T|{cls._ISSUER}", ln.text)]
        words_end = max((ln.x1 for ln in words if not cls._fares(ln.text, 1.0, [])), default=None)
        if words_end is not None and right - words_end >= 1.5 * gol.h:
            for txt, sc in ctx.read_zone(row_zone(gol, words_end, right, slope), h=gol.h, stretch=stretch):
                v, subs = glyph_amount(txt)
                if v is not None and 500 <= v <= 2_000_000 and v % 500 == 0:
                    out.append(Cand(v, sc * 0.95 * 0.85 ** subs, "ocr_refined", txt, [gol]))
        return out


# =========================================================================== parking
class Parking:
    doc_type, category = "parking", "parking"
    KEYS = {"entryTime", "exitTime", "duration", "parkingFee", "balance", "plate"}

    def score(self, seg: Segment) -> float:
        keys = {u.key for u in seg.units if u.key}
        s = len(keys & self.KEYS) + (1.0 if "PARKIR" in compact(seg.full_text) else 0.0)
        return s if s >= 2 and {"entryTime", "exitTime", "duration", "parkingFee"} & keys else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="fee")
        anchors = [u for u in seg.units if u.kind == "anchor" or (u.y0 <= seg.bbox[1] + 3 * u.h and not u.label)]
        for u in anchors:
            t = clean(u.text)
            if re.fullmatch(r"[A-Za-z .&'-]{5,40}", t) and "TERIMA" not in t.upper():
                d.set("merchantName", FV(t.upper(), u.score, lines=list(u.lines)))
                break
        for u in seg.units:
            if not u.label and compact(u.text) in {"MOBIL", "MOTOR", "CAR", "MOTORCYCLE"}:
                d.set("vehicleType", FV("CAR" if compact(u.text) in {"MOBIL", "CAR"} else "MOTORCYCLE", u.score))
        p = seg.first("plate")
        plate = find_plate(p.value_text) if p else None
        if plate:
            d.set("plateNumber", FV(plate, unit_conf(p), raw=p.value_text, lines=list(p.lines)))
            d.set("vehicleType", d.fields.get("vehicleType") or FV("CAR", 0.7))
        for key, pref in (("entryTime", "entry"), ("exitTime", "exit")):
            u = seg.first(key)
            if u:
                _set_datetime(d, u.value_text, list(u.lines), pref)
        u = seg.first("duration")
        if u:
            mins = parse_duration_minutes(u.value_text)
            if mins is not None:
                d.set("durationMinutes", FV(mins, unit_conf(u), raw=u.value_text, lines=list(u.lines)))
        fee_u = seg.first("parkingFee") or seg.first("total")
        c = amount_cands(fee_u, min_value=500)
        if c:
            d.set("fee", FV(int(c[0].value), c[0].conf, raw=c[0].raw, lines=c[0].lines))
        bal_u = seg.first("balance")
        b = amount_cands(bal_u, min_value=0)
        if b:
            d.set("balanceAfter", FV(int(b[0].value), b[0].conf, raw=b[0].raw, lines=b[0].lines))
        card_u = seg.first("cardNo")
        cards = find_card_numbers(card_u.value_text) if card_u else []
        if cards:
            d.set("cardNumber", FV(cards[0], unit_conf(card_u), raw=card_u.value_text, lines=list(card_u.lines)))
        pm = seg.first("paymentMethod")
        issuer = issuer_of(pm.value_text) if pm else issuer_of(seg.full_text) if cards else None
        if issuer:
            d.set("paymentMethod", FV(issuer, unit_conf(pm) if pm else 0.8))
        # the 'transaction' moment of a parking ticket is the exit
        if "exitAt" in d.fields:
            for k in ("Date", "Time", "At"):
                if f"exit{k}" in d.fields:
                    d.fields[f"transaction{k}"] = d.fields[f"exit{k}"]
        ein, eout = d.val("entryAt"), d.val("exitAt")
        if ein and eout and "T" in ein and "T" in eout:
            a, b2 = datetime.fromisoformat(ein), datetime.fromisoformat(eout)
            d.checks.append(Check("EXIT_AFTER_ENTRY", PASS if b2 > a else FAIL, "Exit time is after entry time",
                                  ["entryAt", "exitAt"]))
            dur = d.val("durationMinutes")
            if dur is not None:
                real = (b2 - a).total_seconds() / 60
                ok = abs(real - dur) <= 1.5
                d.checks.append(Check("DURATION_MATCHES", PASS if ok else FAIL,
                                      f"Printed duration {dur} min vs exit-entry {real:.0f} min",
                                      ["durationMinutes", "entryAt", "exitAt"]))
                if ok:
                    for k in ("entryAt", "exitAt", "durationMinutes", "transactionAt", "transactionDate"):
                        if k in d.fields:
                            d.fields[k].confidence = max(d.fields[k].confidence, 0.97)
        return d


# =========================================================================== kwitansi
class Kwitansi:
    doc_type, category = "kwitansi", "other"
    KEYS = {"receivedFrom", "amountWords", "purpose"}

    def score(self, seg: Segment) -> float:
        keys = {u.key for u in seg.units if u.key}
        s = len(keys & self.KEYS) + (2.0 if re.search(r"KWITANSI|KUITANSI", compact(seg.full_text)) else 0.0)
        # the form is its captions; a printed hospital bill titled 'Kwitansi' has none of them and
        # is read as the receipt it is, items and total
        return s + 0.5 if s >= 2 and keys & self.KEYS else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="amount", handwritten=True)
        d.set("receivedFrom", text_field(seg.first("receivedFrom"), upper=False))
        d.set("purpose", text_field(seg.first("purpose"), upper=False))
        slope, right = seg_slope(seg), seg.bbox[2]
        # 'Untuk pembayaran' is handwriting across ruled lines: read as part of the page it comes back
        # as noise ('9r oo, mgg vind mn'), cut out on its own it reads 'uang pulsn bulan ...'
        p, said = seg.first("purpose"), [str(d.val("purpose") or "")]
        if p is not None and ctx.refine:
            lab = p.lines[0]
            reads = ctx.read_zone(row_zone(lab, lab.x1 + 0.3 * lab.h, right, slope, above=1.2, below=0.9), h=lab.h,
                                  rules=True, tighten=2.5)
            said += [t for t, _ in reads]
            text, score = max(reads, key=lambda r: r[1], default=("", 0.0))
            if score >= 0.7 and score > (d.fields["purpose"].confidence if "purpose" in d.fields else 0.0):
                d.fields["purpose"] = FV(clean(text), score * 0.9, "ocr_refined", raw=p.value_text or None, lines=list(p.lines))
        w = seg.first("amountWords")
        words_val = None
        if w is not None:
            words_val, ratio = terbilang_to_int(w.value_text)
            if ratio < 1.0:
                words_val = None       # a number built from half the words is a wrong number
            d.set("amountInWords", FV(strip_value(w.value_text), unit_conf(w) * max(ratio, 0.1), lines=list(w.lines)))
            if words_val is None and ctx.refine:
                # the terbilang sits in a hatched box: every reading gets a different word wrong
                lab = w.lines[0]
                zone = row_zone(lab, lab.x1 + 0.3 * lab.h, right, slope, above=1.2, below=0.9)
                reads = [t for t, _ in ctx.read_zone(zone, h=lab.h, rules=True, tighten=2.5)]
                words_val, voted = terbilang_vote(reads + ([w.value_text] if w.value_lines else []))
                if words_val:
                    d.set("amountInWords", FV(voted, 0.8, "ocr_refined", strip_value(w.value_text), list(w.lines)))
            if words_val:
                d.set("amountInWordsValue", FV(words_val, d.fields["amountInWords"].confidence, "derived"))
        amt_u = seg.first("amountTotal")
        cands = [Cand(c.value, c.conf * 0.9, c.source, c.raw, c.lines) for c in amount_cands(amt_u, min_value=1000)]
        if amt_u is not None and ctx.refine:
            cands += self._zone_amounts(amt_u, slope, right, ctx)
        best = settle(cands)
        if best:
            c = best[0]
            d.set("amount", FV(int(c.value), c.conf, c.source, c.raw, c.lines))
        # A billing period in the purpose is not the payment/signing date.
        date_lines = [ln for u in seg.units if u.key in (None, "datetime") for ln in u.lines
                      if (p is None or ln.cy > max(l.cy for l in p.lines))
                      and re.search(r"(?:19|20)[.\s]*\d[.\s]*\d", ln.text)]
        dates = [(ln.text, ln.score, "ocr", [ln]) for ln in date_lines]
        if ctx.refine:
            for ln in date_lines[:2]:
                for start in (ln.x0, ln.x0 + 0.45 * ln.w):
                    zone = row_zone(ln, start, ln.x1, ln.slope, above=0.45, below=0.25)
                    dates += [(t, sc, "ocr_refined", [Line(-1, t, sc, zone)])
                              for t, sc in ctx.read_zone(zone, h=ln.h, stretch=(0.7, 1.0), date_line=True)]
        set_handwritten_date(d, dates)
        for ln in date_lines:
            if any(v.isoformat() == d.val("transactionDate") for v in handwritten_dates(ln.text)) and "," in ln.text:
                place = clean(ln.text).split(",")[0].strip()
                if place and not re.search(r"\d", place):
                    d.set("place", FV(place.title(), min(ln.score, 0.6), raw=ln.text, lines=[ln]))
                break
        amt = d.val("amount")
        if amt and words_val:
            ok = amt == words_val
            d.checks.append(Check("AMOUNT_MATCHES_WORDS", PASS if ok else FAIL,
                                  f"Jumlah {amt:,} vs terbilang {words_val:,}", ["amount", "amountInWords"]))
            if ok:
                d.fields["amount"].confidence = 0.97
        elif not amt and words_val:
            d.set("amount", FV(words_val, 0.6, "derived", "from terbilang"))
        else:
            d.checks.append(Check("AMOUNT_MATCHES_WORDS", SKIP, "Jumlah or terbilang not readable",
                                  ["amount", "amountInWords"]))
        # 'uang pulsa bulan Oktober': the claim form has a column for it. Two readings have to say so,
        # or the only one there is - 'pulsn' is as close as handwriting gets.
        hits = sum(1 for t in said if any(fuzz.ratio(word.lower(), "pulsa") >= 80 for word in re.findall(r"[A-Za-z]+", t)))
        if hits >= min(2, len(said)) and hits:
            d.category = "pulsa"
        elif any(is_medical(t) for t in said) or is_medical(seg.full_text):
            # 'biaya berobat', a clinic's stamp: a medical claim, with what was treated as its description
            d.category = "medical"
            medical_fields(d, seg)
            if "serviceDescription" not in d.fields and d.val("purpose"):
                d.set("serviceDescription", FV(d.val("purpose"), d.fields["purpose"].confidence, "ocr", lines=d.fields["purpose"].lines))
        d.review.append("Handwritten kwitansi - verify payer, purpose, amount and date against the image")
        return d

    @staticmethod
    def _zone_amounts(amt_u, slope: float, right: float, ctx: Ctx) -> list[Cand]:
        """Re-read the 'Jumlah Rp.' box from the image: the hatching behind the handwriting
        makes the detector return fragments ('000') or nothing."""
        lab = amt_u.lines[0]
        inline = bool(re.search(r"\d", amt_u.label.inline)) and not amt_u.value_lines
        if inline:      # label and amount came back as one line: its box already frames the amount
            reads = ctx.read_zone(row_zone(lab, lab.x0, lab.x1 + 0.5 * lab.h, slope, above=0.8, below=0.8),
                                  h=lab.h, rules=True)
            reads = [(re.split(r"(?i)rp\.?", t)[-1], sc) for t, sc in reads if re.search(r"(?i)rp", t)]
        else:
            reads = ctx.read_zone(row_zone(lab, lab.x1 + 0.3 * lab.h, right, slope, above=1.3, below=0.9),
                                  h=lab.h, rules=True, tighten=1.0)
        out = []
        for txt, sc in reads:
            a = hand_amount(txt)
            if a is not None and a.value >= 1000:
                out.append(Cand(a.value, sc * (0.6 if a.approx else 0.95), "ocr_refined", txt.strip(), list(amt_u.lines),
                                weak="." not in a.raw and "," not in a.raw))
        return out


# =========================================================================== bon kontan
class CashVoucher:
    doc_type, category = "fuel_cash_voucher", "fuel"
    # Legacy sample reference, not a price read from this voucher or guaranteed
    # for another date/location. Used only for an explicitly marked estimate.
    FIXED_PRICE = {"PERTALITE": 10000}

    def score(self, seg: Segment) -> float:
        return 3.0 if fuzz.partial_ratio("BONKONTAN", compact(seg.full_text)) >= 85 else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="amount", handwritten=True)
        for k, fv in merchant_header(seg).items():
            d.set(k, fv)
        for u in seg.units:
            t = clean(u.text)
            if re.match(r"(?i)^PT\.?\s", t):
                d.set("merchantCompany", FV(t.upper(), u.score, lines=list(u.lines)))
            if re.match(r"(?i)^SPBU", t) and station_code(t):
                d.set("merchantId", FV(station_code(t), u.score, lines=list(u.lines)))
        slope, right = seg_slope(seg), seg.bbox[2]
        # the handwritten amounts sit on the dotted leader right of the printed 'Rp.' column
        # (measured from the left edge: the box of an 'Rp.' often swallows the leader behind it)
        rp = [ln for ln in seg.lines if compact(ln.text) == "RP"]
        rp_x = float(np.median([ln.x0 for ln in rp]) + 1.4 * np.median([ln.h for ln in rp])) if rp and ctx.refine else None

        def zone_cands(label: Line) -> list[Cand]:
            if rp_x is None or right - rp_x < 2 * label.h:
                return []
            zone = row_zone(label, rp_x, right + 0.2 * label.h, slope, above=1.3, below=0.45)
            out = []
            for txt, sc in ctx.read_zone(zone, h=label.h, stretch=(1.0, 1.4)):
                a = hand_amount(txt)
                if a is not None and a.value >= 1000:
                    out.append(Cand(a.value, sc * (0.7 if a.approx else 0.95), "ocr_refined", txt.strip(), [label],
                                    weak="." not in a.raw and "," not in a.raw))
            return out

        rows = []
        for ln in seg.lines:
            # '.....Liter Pertalite' comes back as 'Lier Dexlite', 'ter ertamina Dex', 'iter Pertamax'
            unit, _, name = re.sub(r"^[\d.,\s]+", "", clean(ln.text)).partition(" ")
            if name and max(fuzz.ratio(unit.upper(), w) for w in ("LITER", "BOTOL")) >= 60:
                prod = norm_product(name)
                if prod in PRODUCTS:
                    rows.append((ln, prod))
        tot_u = seg.first("amountTotal")
        labels = [ln for ln, _ in rows] + ([tot_u.lines[0]] if tot_u is not None else [])

        def written_on(label: Line) -> list[Cand]:
            """Amounts the first pass found on this row. Handwriting floats above its dotted
            leader, halfway to the row above, so each one goes to the nearest row only."""
            def off(lab: Line, v: Line) -> float:
                return abs(v.cy - (lab.cy + slope * (v.cx - lab.cx) - 0.4 * lab.h)) / lab.h
            out = []
            for v in seg.lines:
                if any(v is lab for lab in labels) or v.x0 < label.x1 - 0.5 * label.h:
                    continue
                if off(label, v) > 0.9 or min(labels, key=lambda lab: off(lab, v)) is not label:
                    continue
                out += [Cand(a.value, v.score * (0.5 if a.approx else 0.8), "ocr", a.raw, [v])
                        for a in parse_amounts(v.text) if a.value and a.value >= 1000][:1]
            return out

        items = []
        for ln, prod in rows:
            got = settle(written_on(ln) + zone_cands(ln))
            if got:
                items.append({"product": prod, "amount": int(got[0].value), "conf": got[0].conf, "row": ln})
        if items:
            d.set("items", FV([{k: v for k, v in it.items() if k in ("product", "amount")} for it in items],
                              min(it["conf"] for it in items)))
            d.set("product", FV(items[0]["product"], 0.7))
        tot_c = [Cand(c.value, c.conf * 0.8, c.source, c.raw, c.lines) for c in amount_cands(tot_u, min_value=1000)]
        if tot_u is not None:
            tot_c += [c for c in written_on(tot_u.lines[0]) if c.lines[0] not in tot_u.lines]
            tot_c += zone_cands(tot_u.lines[0])
        item_sum = sum(it["amount"] for it in items) if items else None
        item_conf = min((it["conf"] for it in items), default=0.0)
        agree = [c for c in tot_c if item_sum is not None and int(c.value) == item_sum]
        total = (settle(tot_c) or (None,))[0]
        if agree:
            # the total line and the item lines were written (and read) separately: agreement
            # is the best evidence a handwritten voucher can give
            best = max(agree, key=lambda c: c.conf)
            d.set("amount", FV(item_sum, min(0.9, max(best.conf, item_conf) + 0.1), best.source, best.raw, best.lines))
            d.checks.append(Check("ITEMS_SUM_EQ_TOTAL", PASS, f"Item lines add up to Jumlah {item_sum:,}", ["items", "amount"]))
        elif total is not None and items:
            # two readings of the same money that differ: keep the stronger one as a guess only
            if total.conf >= item_conf:
                d.set("amount", FV(int(total.value), total.conf * 0.5, total.source, total.raw, total.lines))
            else:
                d.set("amount", FV(item_sum, item_conf * 0.5, "derived", "sum of item lines"))
            d.checks.append(Check("ITEMS_SUM_EQ_TOTAL", FAIL,
                                  f"Item lines add up to {item_sum:,}, Jumlah reads {int(total.value):,}",
                                  ["items", "amount"]))
        elif total is not None:
            d.set("amount", FV(int(total.value), total.conf * 0.8, total.source, total.raw, total.lines))
        elif items:
            d.set("amount", FV(item_sum, item_conf, "derived", "sum of item lines"))
        self._liters(d, seg, items, slope, ctx)
        u = seg.first("datetime")
        if u is not None:
            lab = u.lines[0]
            mates = row_mates(seg, lab, tol=1.3)
            dates = [(u.value_text + " " + " ".join(x.text for x in mates),
                      min([u.score] + [x.score for x in mates]), "ocr", list(u.lines) + mates)]
            if ctx.refine:
                # Include floating handwriting and the preprinted `20` year prefix,
                # stopping before the signature on the right.
                zone = row_zone(lab, lab.x1, min(right, lab.x1 + 7 * lab.h), slope, above=1.45, below=0.2)
                dates += [(t, sc, "ocr_refined", [Line(-1, t, sc, zone)])
                          for t, sc in ctx.read_zone(zone, h=lab.h, stretch=(0.7, 1.0), handwriting=True)]
            set_handwritten_date(d, dates)
        d.review.append("Handwritten cash voucher (bon kontan) - verify liters, amount and date against the image")
        return d

    @classmethod
    def _liters(cls, d: Draft, seg: Segment, items: list[dict], slope: float, ctx: Ctx) -> None:
        """Read liters before calculating a price; never overwrite ink with a reference price."""
        amount = d.fields.get("amount")
        if len(items) != 1:
            return
        it = items[0]
        row: Line = it["row"]
        readings = []
        prefix = re.match(r"^([\d.,\s]+)(?=[A-Za-z])", clean(row.text))
        if prefix and re.search(r"\d", prefix[1]):
            readings.append((prefix[1], row.score, "ocr", [row]))
        if ctx.refine:
            # Exclude the previous fuel row's dotted rule, which makes `3,5` read `33`.
            zones = [row_zone(row, max(seg.bbox[0], row.x0 - 0.6 * row.h), row.x0 + 1.4 * row.h,
                              slope, above=0.8, below=0.25),
                     row_zone(row, seg.bbox[0], row.x0 + 0.3 * row.w, slope, above=0.8, below=0.25)]
            for zone in zones:
                readings += [(t, sc, "ocr_refined", [Line(-1, t, sc, zone)])
                             for t, sc in ctx.read_zone(zone, h=row.h, stretch=(0.7, 1.0), handwriting=True)]
                if choose_volume(readings) is not None:
                    break
        volume = choose_volume(readings)
        if volume is not None:
            d.set("volumeLiters", volume)
            if amount is not None and amount.value is not None:
                unit_price = round(amount.value / volume.value, 2)
                d.set("pricePerLiter", FV(unit_price, min(amount.confidence, volume.confidence), "derived",
                                         f"Rp {amount.value:,} / {volume.value:g} L"))
                d.review.append("Price per liter calculated from amount / handwritten liters - confirm")
                reference = cls.FIXED_PRICE.get(it["product"])
                if reference and abs(unit_price - reference) > reference * 0.02:
                    d.checks.append(Check("REFERENCE_PRICE_MISMATCH", WARN,
                                          "Written liters and amount imply a different unit price from the sample reference; verify both",
                                          ["volumeLiters", "amount", "pricePerLiter"]))
            return
        price = cls.FIXED_PRICE.get(it["product"])
        if price and amount is not None and amount.value is not None:
            liters = round(amount.value / price, 2)
            if not 0.5 <= liters <= 200:
                return
            d.set("pricePerLiter", FV(price, 0.65, "inferred", "sample reference price; not read from this receipt"))
            d.set("volumeLiters", FV(liters, min(0.65, amount.confidence), "derived",
                                    f"Rp {amount.value:,} / assumed Rp {price:,}"))
            d.review.append(f"Liters ({liters:g}) estimated using assumed Rp {price:,} per liter; handwriting was inconclusive - confirm")


# =========================================================================== generic
class Generic:
    doc_type, category = "unknown", "other"

    def score(self, seg: Segment) -> float:
        return 0.1

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="amount")
        best = None
        for u in seg.units:
            if re.search(r"(?i)total|jumlah|bayar|grand|tagihan", u.text):
                c = amount_cands(u, min_value=100) or [
                    Cand(a.value, ln.score * 0.8, "ocr", a.raw, [ln])
                    for ln in row_mates(seg, u.lines[-1]) for a in parse_amounts(ln.text) if a.value and a.value >= 100]
                if c:
                    best = max(c, key=lambda x: x.value)
        if best:
            d.set("amount", FV(int(best.value), best.conf * 0.8, raw=best.raw, lines=best.lines))
        for u in seg.units:
            if _set_datetime(d, u.text, list(u.lines)):
                break
        d.review.append("Unrecognised document layout - fields extracted heuristically")
        return d


TEMPLATES = [Toll(), Parking(), Kwitansi(), CashVoucher(), Generic()]
