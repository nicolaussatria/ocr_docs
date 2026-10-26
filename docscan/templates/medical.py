"""What the medical claim form needs from a hospital, clinic, doctor or pharmacy receipt.

Claim Medical Karyawan asks for the patient named on the receipt, what was paid for (the service or
the medicines) and, under Keterangan, the hospital or clinic with its address and the doctor. The
amount and date are read by the receipt's own template; this adds the rest where the paper prints
it, and leaves a field out rather than guess it.
"""
from __future__ import annotations

import re
from datetime import date

import numpy as np

from ..describe import MEDICAL
from ..layout import Segment
from rapidfuzz import fuzz

from ..normalize import clean, compact, find_dates
from .base import FV, Ctx, Draft, row_zone, seg_slope

_CAPTION = r"(?:[:.\-]\s*|\s+)"
_PATIENT = re.compile(rf"(?i)^\W*(?:nama\s+pasien|pasien|patient(?:\s+name)?|atas\s+nama|a\.?\s?n\.)\s*{_CAPTION}(.*)$")
_DOCTOR = re.compile(rf"(?i)^\W*(?:nama\s+dokter|dokter(?:\s+pemeriksa)?|dpjp|doctor|physician)\s*{_CAPTION}(.*)$")
_DR_TITLE = re.compile(r"(?<![A-Za-z])(drg?\.?\s+[A-Z][A-Za-z.'\-]+(?:\s+[A-Z][A-Za-z.'\-]+){0,4}(?:,\s*Sp\.?\s?[A-Z.]{1,8})?)")
_SERVICE = re.compile(rf"(?i)^\W*(?:pelayanan|layanan|tindakan|jasa(?:\s+medis)?|keterangan|diagnosa|resep|obat)\s*{_CAPTION}(.+)$")
_ADDRESS = re.compile(r"(?i)\b(jl|jln|jalan|ruko|komp|komplek|kompleks|gedung|gd|blok|kav|perum|perumahan)\b\.?")
_REGION = re.compile(r"(?i)\b(kota|kab|kabupaten|kec|kecamatan|kel|kelurahan|jakarta|bogor|depok|tangerang|bekasi|bandung"
                     r"|cimahi|surabaya|semarang|yogyakarta|medan|makassar|denpasar|malang)\b|\b\d{5}\b")
_PHONE = re.compile(r"(?i)\s*[,;-]?\s*\b(telp|tel|phone|hp|wa|fax)\b.*$")


_NOT_A_PERSON = re.compile(r"(?i)^(umum|bpjs|asuransi|baru|lama|rawat (jalan|inap)|spesialis|gigi|anak|jaga|praktek|-)$")


def is_medical(text: str) -> bool:
    return re.search(MEDICAL, text, re.I) is not None


def _name(text: str) -> str | None:
    """A person's name: letters, spaces, dots, commas and titles; a number or a long sentence is not one."""
    t = clean(text).strip(" :.-")
    letters = sum(ch.isalpha() for ch in t)
    if letters < 3 or letters < 0.7 * len(t.replace(" ", "")) or len(t) > 60 or find_dates(t) or _NOT_A_PERSON.match(t):
        return None
    return t


def _value_beside(seg: Segment, line, tail: str) -> str:
    """The caption's own value, or the box right of it on the same row ('Pasien :' | 'Andi Setiawan')."""
    if clean(tail).strip(" :.-"):
        return tail
    mates = sorted((o for o in seg.lines if o is not line and abs(o.cy - line.cy) <= 0.5 * max(o.h, line.h)
                    and 0 <= o.x0 - line.x1 <= 15 * line.h), key=lambda o: o.x0)
    return mates[0].text if mates else ""


def medical_fields(d: Draft, seg: Segment) -> None:
    lines = sorted(seg.lines, key=lambda ln: (ln.cy, ln.x0))
    for ln in lines:
        text = clean(ln.text)
        for key, pattern in (("patientName", _PATIENT), ("doctorName", _DOCTOR)):
            m = pattern.match(text)
            if m and key not in d.fields:
                name = _name(_value_beside(seg, ln, m.group(1)))
                if name:
                    d.set(key, FV(name, ln.score * 0.9, raw=ln.text, lines=[ln]))
    if "doctorName" not in d.fields:                  # 'dr. Andi Setiawan, Sp.PD' anywhere on the paper
        hit = next(((m.group(1), ln) for ln in lines for m in [_DR_TITLE.search(clean(ln.text))] if m), None)
        if hit:
            d.set("doctorName", FV(hit[0].strip(" ,"), hit[1].score * 0.85, raw=hit[1].text, lines=[hit[1]]))
    # what was paid for: the services and medicines the bill lists, else a line that names them
    names = [i["name"] for i in d.val("items") or []] or list(seg.extra.get("item_names", []))
    if names:
        d.set("serviceDescription", FV(", ".join(names)[:300], 0.75, "derived", "item lines"))
    else:
        for ln in lines:
            m = _SERVICE.match(clean(ln.text))
            if m and len(clean(m.group(1))) >= 3:
                d.set("serviceDescription", FV(clean(m.group(1)), ln.score * 0.85, raw=ln.text, lines=[ln]))
                break
    if "merchantAddress" not in d.fields:
        _address(d, lines)


def _address(d: Draft, lines) -> None:
    """The street line under the hospital's name, with the town line after it when there is one."""
    merchant = d.fields.get("merchantName")
    top = merchant.lines[0].cy if merchant and merchant.lines else None
    for i, ln in enumerate(lines[:12]):
        if top is not None and ln.cy < top:
            continue
        text = _PHONE.sub("", clean(ln.text)).strip(" ,;-")
        if not _ADDRESS.search(text):
            continue
        parts = [text]
        if i + 1 < len(lines):
            nxt = _PHONE.sub("", clean(lines[i + 1].text)).strip(" ,;-")
            if nxt and _REGION.search(nxt) and not _ADDRESS.search(nxt) and not find_dates(nxt) and len(nxt) <= 60:
                parts.append(nxt)
        d.set("merchantAddress", FV(", ".join(parts), ln.score * 0.9, raw=ln.text, lines=[ln]))
        return


# =========================================================================== optician
_OPTIK = re.compile(r"OPTIK|OPTIC|KACAMATA")
_CAPTIONS = {"name": r"nama|name", "frame": r"frame|bingkai|gagang", "lens": r"lensa|lens", "price": r"harga|total|jumlah",
             "rest": r"sisa|kurang", "office": r"kantor|alamat toko|toko|cabang",
             "address": r"alamat", "phone": r"hp|telepon|telp"}
# lens types as an optician writes them; the hand and the reader both mangle them ('Photocromic', 'Photocnonc')
LENSES = ["Photochromic", "Bluecut", "Blue Ray", "Progressive", "Bifocal", "Kryptok", "Single Vision", "Anti Radiasi",
          "Transitions", "Polarized", "Polycarbonate", "Hi-Index", "Crizal", "Essilor", "Hoya", "Zeiss"]


class OpticalForm:
    """An optician's handwritten receipt: Nama / Frame / Lensa / Harga on dotted lines, a prescription
    table (R, L, CYL, AX) underneath. Glasses are claimed as medical; the frame and lens are what was
    paid for, the name is the patient's."""
    doc_type, category = "optical", "medical"

    def score(self, seg: Segment) -> float:
        c = compact(seg.full_text)
        captions = sum(1 for w in ("FRAME", "LENSA", "RESEP", "CYL") if w in c)
        return 3.0 if (_OPTIK.search(c) and captions >= 1) or captions >= 3 else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="amount", handwritten=True)
        lines = sorted(seg.lines, key=lambda ln: (ln.cy, ln.x0))
        slope = seg_slope(seg)
        caps = {}
        for ln in lines:
            for key, pattern in _CAPTIONS.items():
                if key not in caps and re.match(rf"(?i)^\s*(?:{pattern})\b", clean(ln.text)):
                    caps[key] = ln
        self._shop(d, lines, caps)
        m = next((m for ln in lines for m in [re.search(r"(?i)\bno\s*[:.]?\s*(\d{4,})", clean(ln.text))] if m), None)
        if m:
            d.set("receiptNo", FV(m.group(1), 0.9))
        if "name" in caps:
            got = self._beside(seg, caps["name"], ctx, slope, read=lambda t: _name(t), bound=caps.get("phone"))
            if got:
                d.set("patientName", FV(got[0], min(got[1], .8), got[2], got[4], [caps["name"]]))
        if "address" in caps:
            got = self._beside(seg, caps["address"], ctx, slope, read=_name)
            if got:
                d.set("patientAddress", FV(got[0], min(got[1], .8), got[2], got[4], [caps["address"]]))
        for key, caption in (("patientPhone", caps.get("phone")), ("merchantPhone", caps.get("office"))):
            if caption is not None:
                phone = re.search(r"(?:\+62|0)\d[\d\s-]{7,16}\d", caption.text)
                if phone:
                    d.set(key, FV(re.sub(r"[\s-]", "", phone.group()), min(caption.score, .8), raw=caption.text, lines=[caption]))
        what = []
        if "frame" in caps:
            got = self._beside(seg, caps["frame"], ctx, slope, read=lambda t: t if re.fullmatch(r"[A-Za-z]{0,3}\d{3,6}", t) else None,
                               agree=2)
            what.append(f"frame {got[0]}" if got else "frame")
            if got:
                d.set("frameCode", FV(got[0], min(got[1], .7), got[2], got[4], [caps["frame"]]))
            else:
                d.review.append("Frame code is unclear - check the handwritten code")
        if "lens" in caps:
            got = self._beside(seg, caps["lens"], ctx, slope, read=_lens, agree=2)
            what.append(f"lensa {got[0]}" if got else "lensa")
            if got:
                d.set("lensType", FV(got[0], min(got[1], .7), "ocr_refined", got[4], [caps["lens"]]))
        d.set("serviceDescription", FV("Kacamata: " + " + ".join(what) if what else "Kacamata", 0.7, "ocr_refined"))
        if "price" in caps:
            texts = [(t, sc) for t, sc, _ in self._texts(seg, caps["price"], ctx, slope, caps.get("rest")) if sc >= 0.5]
            sized = sorted({v for t, _ in texts if (v := _price(t)) is not None})
            if len(sized) == 1:
                # the size comes from a reading that kept its thousands dot or all its digits; the readings
                # that lost zeros in the ',-' scribble still agree on the leading digits ('3500O' for 350.000)
                value = sized[0]
                lead = str(value).rstrip("0")
                support = sum(1 for t, _ in texts if _digits(t).rstrip("0") == lead)
                raw = next(t for t, _ in texts if _price(t) == value)
                d.set("amount", FV(value, 0.6 if support >= 3 else 0.45, "ocr_refined", raw, [caps["price"]]))
        if "rest" in caps:
            got = self._beside(seg, caps["rest"], ctx, slope, read=_price, agree=2)
            if got:
                d.review.append(f"Sisa (still to pay) reads Rp {int(got[0]):,} - check what was actually paid")
        self._date(d, lines, caps, ctx, slope)
        self._prescription(d, lines)
        d.review.append("Handwritten optician receipt - verify patient, frame/lens, amount and date against the image")
        return d

    @staticmethod
    def _shop(d: Draft, lines, caps) -> None:
        """'OPTIK' / 'Rizki' / 'eye+': the logo's words above the first caption; the office line's town."""
        first = min((ln.cy for ln in caps.values()), default=1e9)
        left = min(ln.x0 for ln in lines)
        width = max(ln.x1 for ln in lines) - left
        words = [ln for ln in lines if ln.cy < first and ln.x0 < left + 0.45 * width and not re.search(r"\d", ln.text)
                 and len(clean(ln.text).split()) <= 2 and sum(ch.isalpha() for ch in ln.text) >= 3]
        if words:
            d.set("merchantName", FV(" ".join(clean(ln.text) for ln in words).upper(), min(ln.score for ln in words) * 0.9,
                                     lines=words))
        office = caps.get("office")
        if office is not None:
            town = re.sub(r"[\s\-–,]*\(?\+?\d[\d\s\-()]{6,}$", "", clean(office.text).partition(":")[2]).strip(" -,")
            if town and sum(ch.isalpha() for ch in town) >= 3:
                d.set("merchantAddress", FV(town, office.score * 0.9, raw=office.text, lines=[office]))

    @staticmethod
    def _texts(seg: Segment, cap, ctx: Ctx, slope: float, bound=None) -> list[tuple[str, float, str]]:
        """What is written on the dotted line after a caption: the first pass's box there, and re-reads of the band."""
        right = bound.x0 - 0.2 * cap.h if bound is not None and bound.x0 > cap.x1 else seg.bbox[2]
        mates = [o for o in seg.lines if o is not cap and cap.x1 - 0.3 * cap.h <= o.x0 < right
                 and -1.2 * cap.h <= o.cy - cap.cy <= 0.6 * cap.h and o.x0 - cap.x1 <= 6 * cap.h]
        texts = [(o.text, o.score, "ocr") for o in mates]
        texts = sorted(texts, key=lambda t: -t[1])[:2]
        # An inline value must also work when the detector combines caption and handwriting.
        inline = re.sub(r"(?i)^\s*(?:" + "|".join(_CAPTIONS.values()) + r")\b[\s:=.]*", "", cap.text)
        if inline != cap.text and inline.strip():
            texts.append((inline, cap.score, "ocr"))
        if mates and bound is None:
            # Avoid empty paper and the next field on the same dotted line.
            right = min(right, max(o.x1 for o in mates) + cap.h)
        if ctx.refine and right - cap.x1 >= 2 * cap.h:
            zone = row_zone(cap, cap.x1 + 0.2 * cap.h, right, slope, above=1.1, below=0.5)
            texts += [(t, sc, "ocr_refined") for t, sc in ctx.read_zone(zone, h=cap.h, stretch=(0.7, 1.0), handwriting=True)]
        return texts

    @classmethod
    def _beside(cls, seg: Segment, cap, ctx: Ctx, slope: float, *, read, agree: int = 1, bound=None):
        """The value a caption's line holds, by majority of its readings.
        -> (value, confidence, source, readings that agree, raw) or None."""
        texts = cls._texts(seg, cap, ctx, slope, bound)
        votes: dict[object, list[tuple[float, str, str]]] = {}
        for text, score, source in texts:
            value = read(clean(text).strip(" :.-"))
            if value is not None and score >= 0.5:
                votes.setdefault(value, []).append((score, source, text))
        if not votes:
            return None
        value, got = max(votes.items(), key=lambda kv: (len(kv[1]), max(s for s, *_ in kv[1])))
        if len(got) < agree:
            return None
        best = max(got, key=lambda g: g[0])
        source = "ocr" if any(src == "ocr" and s >= 0.9 for s, src, _ in got) else "ocr_refined"
        conf = min(0.95, best[0]) if source == "ocr" else min(0.8, best[0])
        return value, conf, source, len(got), best[2]

    @staticmethod
    def _date(d: Draft, lines, caps, ctx: Ctx, slope: float) -> None:
        """'Veteran, 02/10 2026' up by the receipt number: kept only when two readings give the same date."""
        first = min((ln.cy for ln in caps.values()), default=1e9)
        cands = [ln for ln in lines if ln.cy < first and re.search(r"\d", ln.text) and not re.search(r"(?i)\bno\b|\d{4}\s*\d{4}", ln.text)]
        votes: dict[str, int] = {}
        for ln in cands:
            texts = [ln.text]
            if ctx.refine:
                zone = row_zone(ln, ln.x0, ln.x1, slope, above=0.6, below=0.6)
                texts += [t for t, sc in ctx.read_zone(zone, h=ln.h, stretch=(0.7, 1.0), handwriting=True) if sc >= 0.5]
            for t in texts:
                for found in find_dates(t):
                    votes[found.value.isoformat()] = votes.get(found.value.isoformat(), 0) + 1
        stacked = _stacked_dates(lines, first, ctx)
        for value, raw in stacked:
            votes[value] = votes.get(value, 0) + 2
        supported = [v for v, count in votes.items() if count >= 2]
        if len(supported) == 1 and len(votes) == 1:
            raw = next((raw for value, raw in stacked if value == supported[0]), " | ".join(l.text for l in cands))
            d.set("transactionDate", FV(supported[0], 0.6, "ocr_refined", raw, cands))
        elif votes:
            d.set("transactionDateCandidates", FV(sorted(votes), .4, "ocr_refined"))
            d.review.append("Date readings disagree or lack confirmation - check the handwritten date")

    @staticmethod
    def _prescription(d: Draft, lines) -> None:
        """Transcribe the first prescription row by its printed column headings.

        A dash/empty cell stays absent. Never copy a value from the other eye.
        Decimal restoration is explicitly marked for review, not treated as a reading.
        """
        heads = [l for l in lines if compact(l.text) in {"R", "L", "CYL", "AX", "ADD"}]
        if not {"R", "L", "CYL"} <= {compact(l.text) for l in heads}:
            return
        baseline = float(np.median([l.cy for l in heads]))
        heads = sorted((l for l in heads if abs(l.cy - baseline) <= l.h), key=lambda l: l.cx)
        eye = None
        for i, head in enumerate(heads):
            label = compact(head.text)
            if label in {"R", "L"}:
                eye = "right" if label == "R" else "left"
            key = "addition" if label == "ADD" else (eye or "") + {"R": "Sphere", "L": "Sphere", "CYL": "Cylinder", "AX": "Axis"}.get(label, "")
            if not eye and label != "ADD":
                continue
            x0 = (heads[i-1].cx + head.cx) / 2 if i else head.x0 - head.h
            x1 = (heads[i+1].cx + head.cx) / 2 if i+1 < len(heads) else head.x1 + head.h
            values = [l for l in lines if x0 <= l.cx < x1 and .5 * head.h < l.cy - head.cy < 2 * head.h]
            for ln in values:
                raw = clean(ln.text).replace(",", ".").replace("−", "-").strip()
                if not re.fullmatch(r"[+-]?\d{1,3}(?:\.\d{1,2})?", raw):
                    continue
                value, source = float(raw), "ocr"
                if label == "AX":
                    if value != int(value) or not 0 <= value <= 180:
                        continue
                else:
                    if "." not in raw and abs(value) > 30:
                        value, source = value / 100, "inferred"
                    if not -30 <= value <= 30 or abs(value * 4 - round(value * 4)) > .001:
                        continue
                d.set(key, FV(value, .6 if source == "inferred" else min(.8, ln.score), source, ln.text, [ln]))
                if source == "inferred":
                    d.review.append(f"{key}: decimal inferred from {raw} as {value:+.2f}; confirm against the prescription")
                break


def _stacked_dates(lines, first: float, ctx: Ctx) -> list[tuple[str, str]]:
    """Read day above month beside a printed year, using the detected month position.

    Each component needs two literal numeric readings. No year comes from today's
    date, and OCR letter substitutions alone cannot establish a date.
    """
    if not ctx.refine:
        return []
    result = []
    for month in lines:
        if month.cy >= first or month.w > 2 * month.h or not re.fullmatch(r"[\dIltoO/ .-]{1,4}", month.text):
            continue
        headers = [l for l in lines if l is not month and l.cy < month.cy - .2 * month.h
                   and l.x0 < month.cx < l.x1 and l.w > 3 * month.h
                   and month.cy - l.cy < 2 * month.h and re.search(r"\d", l.text)]
        for header in headers:
            h = month.h
            boxes = [(month.x0 - .6*h, month.y0 - 1.15*h, month.x1 + .3*h, month.y0 - .1*h),
                     (month.x0, month.y0, month.x1 + .2*h, month.y1),
                     (month.x1 + .2*h, month.y0 - .9*h, header.x1, month.y0 + .1*h)]
            parts = []
            for idx, (x0,y0,x1,y1) in enumerate(boxes):
                if x1 <= x0 or y1 <= y0:
                    break
                q = np.array([[x0,y0],[x1,y0],[x1,y1],[x0,y1]], np.float32)
                reads = ctx.read_zone(q, h=h, stretch=(.7,1.,1.3), date_line=True)
                votes = {}
                for text, score in reads:
                    t = re.sub(r"[\s./,]", "", text)
                    if score < .5 or not re.fullmatch(r"\d{4}" if idx == 2 else r"\d{1,2}", t):
                        continue
                    value = int(t)
                    if not (1900 <= value <= 2099 if idx == 2 else 1 <= value <= (31 if idx == 0 else 12)):
                        continue
                    votes[value] = votes.get(value, 0) + 1
                supported = [v for v, count in votes.items() if count >= 2]
                if len(supported) != 1 or len(votes) != 1:
                    break
                parts.append(supported[0])
            if len(parts) == 3:
                day, mon, year = parts
                try:
                    result.append((date(year, mon, day).isoformat(), f"{day:02}/{mon:02}/{year}"))
                except ValueError:
                    pass
    return result


def _lens(text: str) -> str | None:
    """'Photocnonc' -> 'Photochromic': the lens type a reading is closest to, if it is close."""
    word = re.sub(r"[^a-z]", "", text.lower())
    if len(word) < 4:
        return None
    score, lens = max((fuzz.partial_ratio(word, re.sub(r"[^a-z]", "", name.lower())), name) for name in LENSES)
    return lens if score >= 75 and fuzz.ratio(word[:5], re.sub(r"[^a-z]", "", lens.lower())[:5]) >= 60 else None


def _digits(text: str) -> str:
    """The digits of a handwritten price without the 'Rp' in front; look-alike letters count as digits."""
    t = re.sub(r"^\W*[rRlLiI1|][pP0oO][\s.,:]*", "", text)
    return re.sub(r"\D", "", t.translate(str.maketrans("SsOoQDIlB", "550000118")))


def _price(text: str) -> int | None:
    """A handwritten price: 'Rp' in front, ',-' behind - the reader turns both into stray characters
    ('10.350.06P0' for 'Rp. 350.000,-'). Glasses cost from Rp 50,000 to Rp 20,000,000, priced in thousands."""
    digits = _digits(text)
    if not 5 <= len(digits) <= 8:
        return None
    value = int(digits)
    thousands = round(value / 1000) * 1000
    if abs(value - thousands) > 100 or not 50_000 <= thousands <= 20_000_000:
        return None
    return thousands
