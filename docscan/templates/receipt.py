"""General receipt / invoice: any layout that prints its money in labelled rows.

There is no template here. A money row is recognised by what it is called (total,
subtotal, pajak, service, diskon, tunai, kembali, kartu ...) in Indonesian or English
and, because names get garbled and totals are printed in fonts the recogniser
misreads, by how the amounts add up:

    subtotal + tax + service - discount = total          cash - change = total

The printed total is the answer unless the arithmetic says otherwise: sums that land
on it confirm it, rows that were hard to read are read again from the image, and a
total nobody could read is derived only when two independent sums agree. A single
sum is not enough - one unreadable row between subtotal and total would make it wrong.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from ..describe import kind_of
from .medical import medical_fields
from ..layout import Segment
from ..normalize import clean, compact, find_dates, parse_amounts
from ..ocr import Line
from .base import FV, PASS, SKIP, WARN, Cand, Check, Ctx, Draft, row_zone, seg_slope
from .other import _set_datetime, issuer_of

# role -> what the row is called. Longer names win ('total qty' is a count, not the total).
_NAMES = {
    "count": "total qty|total item|total items|total quantity|jumlah item|jml item|jumlah barang|total produk|items|item"
             "|qty|pcs",
    "change": "kembali|kembalian|total kembalian|uang kembali|kembalian tunai|change|change due",
    "subtotal": "subtotal|sub total|sub ttl|nett amt|net amt|net amount|total sebelum pajak|harga sebelum pajak|dpp",
    "discount": "diskon|discount|disc|potongan|potongan harga|total diskon|total disc|total discount|hemat|anda hemat"
                "|voucher",
    "tax": "ppn|pb1|pajak|pajak resto|pajak restoran|pajak daerah|total pajak|tax|tax amt|vat|pjk|pjk resto|govt tax",
    "service": "service|service charge|serv charge|servis|svc|svc chg|svc charge",
    "rounding": "pembulatan|rounding|round",
    "fee": "bea materai|materai|pelunasan bea materai|stamp duty|delivery|ongkir|ongkos kirim|biaya kirim|shipping"
           "|biaya admin|biaya administrasi|admin fee|biaya layanan|biaya tambahan|biaya lain|biaya cc|biaya|surcharge",
    "taxbase": "subject to tax|dasar pengenaan pajak|taxable|taxable amount|tax base|nilai dpp",
    "total": "total|grand total|total bayar|total belanja|total pembayaran|total tagihan|total harga|total transaksi"
             "|total akhir|total due|total amount|total sales|total rp|amount due|balance due|due|net total|bill total"
             "|grand|grand ttl|jumlah|jumlah bayar|jumlah total|tagihan|netto|harga jual|ttl|tl",
    "noncash": "debit|kredit|credit|credit card|debit card|card|kartu|kartu debit|kartu kredit|qris|gopay|ovo|dana"
               "|shopeepay|linkaja|edc|bca|bca card|mandiri|bni|bri|visa|master|mastercard|emoney|e money|flazz"
               "|transfer|non tunai",
    "cash": "tunai|cash|uang bayar|uang tunai|bayar|bayar tunai|dibayar|pay|pay cash|cash payment|payment|pembayaran"
            "|tendered|cash tendered|received|diterima",
}
_PHRASES = sorted(((re.sub(r"\W", "", p).upper(), role) for role, names in _NAMES.items() for p in names.split("|")),
                  key=lambda t: -len(t[0]))
_MONEY_ROLES = ("subtotal", "tax", "service", "discount", "rounding", "fee", "total", "cash", "change", "noncash")
_BIG = ("total", "subtotal", "cash", "noncash")      # never below Rp1,000: a smaller reading lost its digits
_FIELD = {"total": "total", "subtotal": "subtotal", "tax": "tax", "service": "serviceCharge", "discount": "discount",
          "rounding": "rounding", "fee": "otherCharges", "cash": "cashPaid", "change": "change", "noncash": "nonCashPaid"}
_TAX = (1.10, 1.11, 1.12)        # restaurant tax (PB1) and VAT
_AS_LETTERS = str.maketrans("0158", "OISB")
_NOT_A_NAME = re.compile(r"(?i)\b(jl|jln|jalan|telp|tlp|phone|fax|npwp|www|http|kasir|cashier|table|meja|tanggal|date"
                         r"|struk|receipt|bill|invoice|order|no|pos|shift|terima kasih|thank)\b")
_RECEIPT_NO = re.compile(r"(?i)\b(?:invoice|inv|bill|struk|receipt|trx|transaksi|order|nota|faktur)\s*(?:no|nomor|id)?"
                         r"\s*[.:#]*\s*([A-Z0-9][A-Z0-9/\-.]{3,24})")


def role_of(label: str) -> str | None:
    """What a row's name says it is, tolerating OCR noise in the longer names.

    The name has to lead the row, end it ('Dine In Total') or make up most of it:
    'Pay Cash' and 'BCA Card' are payments, 'Cashew Nut Chicken' and 'Bottle' are
    things somebody ordered."""
    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", label.upper()) if not t.isdigit()]
    # digits that stand in for letters inside a word: T01AL
    tokens = [t.translate(_AS_LETTERS) if sum(ch.isalpha() for ch in t) >= 3 else t for t in tokens]
    letters = sum(len(t) for t in tokens)
    if not tokens or letters > 40:
        return None
    best = None
    for i in range(len(tokens)):
        for j in range(i + 1, min(len(tokens), i + 4) + 1):
            gram = "".join(tokens[i:j])
            if i and j < len(tokens) and len(gram) < 0.5 * letters:
                continue
            for phrase, role in _PHRASES:
                if gram == phrase:
                    score = 100.0
                elif len(phrase) >= 5 and abs(len(gram) - len(phrase)) <= 2:
                    score = fuzz.ratio(gram, phrase)
                    if score < (80 if j - i == 1 else 85):      # 'to' + 'tax' is not a misspelt 'total'
                        continue
                else:
                    continue
                if i and j == len(tokens) and len(gram) < 0.5 * letters and role != "total":
                    continue
                key = (len(phrase) * score, -i)          # the longest name that fits: 'total kembalian' over 'total'
                if best is None or key > best[0]:
                    best = (key, role)
    return best[1] if best else None


def money_in(text: str, *, small: bool = False) -> list[tuple[int, str, bool]]:
    """IDR amounts in one OCR line as (value, raw, approx), left to right; '-45' and '34-' are negative.

    Quantities, percentages, dates, phone and card numbers are not money. Without a
    thousands separator only four to seven digits count - or anything, on a row whose
    name says it holds money (`small`: a tax of 727, a change of 0)."""
    s = clean(text)
    if find_dates(s) or re.search(r"\d{1,2}:\d{2}", s) or re.search(r"\d{9,}", re.sub(r"[\s\-]", "", s)):
        return []
    s = re.sub(r"\(?\d+(?:[.,]\d+)?\s*%\)?", " ", s)                    # 'PB1 (10%)', 'TAX 10.00 %'
    s = re.sub(r"(?i)\b(rp|idr)\.?(?=\s*\d)", " ", s)
    s = re.sub(r"(?<=\d)\s*[xX@*]\s*(?=\d)", " x ", s)                 # '1x12,500' is a quantity and a price
    if re.fullmatch(r"\d{1,3}(?: \d{3})+", s):
        s = s.replace(" ", ",")                                           # '154 000' printed with a space
    # digits that belong to a word are not an amount: 'PB1', 'x2', 'BER2' (but 'CASH50.000' is)
    s = re.sub(r"(?<=[A-Za-z])\d{1,3}(?![\d.,])|(?<![\d.,])\d{1,3}(?=[A-Za-z]{2}|[xX]\b)", " ", s)
    out = []
    for a in parse_amounts(s):
        digits = re.sub(r"\D", "", a.raw)
        if a.value is None or (len(digits) > 1 and digits[0] == "0") or a.value > 999_999_999:
            continue                                                        # '000.800': a total read upside down
        grouped = bool(re.search(r"[.,]", a.raw))
        if grouped or 4 <= len(digits) <= 7 or small:
            at = s.find(a.raw)
            minus = s[:at].rstrip().endswith(("-", "(")) or s[at + len(a.raw):].lstrip().startswith("-")
            out.append((-a.value if minus else a.value, a.raw, a.approx))
    return out


# What the fields of an invoice header are called. The value sits after a colon, to the right, or -
# in a table - underneath; bilingual forms print each name twice ('Tanggal Cetak Tagihan/ Statement Date').
_DOC_KEYS = {
    "receiptNo": "no invoice|invoice no|nomor invoice|invoice number|no faktur|nomor faktur|no tagihan|nomor tagihan"
                 "|no kwitansi|no nota|nomor nota|bill no|receipt no|no transaksi|nomor transaksi|no struk|order id"
                 "|order no|no order|trx id|no ref|ref no|no referensi",
    "transactionDate": "tanggal cetak|statement date|invoice date|tanggal invoice|tanggal faktur|tanggal tagihan"
                       "|tanggal transaksi|tgl transaksi|tgl invoice|bill date|tanggal|date|tgl",
    "dueDate": "jatuh tempo|tanggal jatuh tempo|tgl jatuh tempo|due date|payment due",
    "customerId": "kode pelanggan|customer id|id pelanggan|no pelanggan|nomor pelanggan|customer no|customer number",
    "customerName": "kepada|kepada yth|bill to|billed to|sold to|customer|customer name|pelanggan|nama pelanggan|to",
    "npwp": "npwp",
    "printedTotal": "total tagihan|total amount|amount due|jumlah tagihan|total due|total pembayaran",
}
_DOC_KEY_LIST = sorted(((re.sub(r"\W", "", k).upper(), f) for f, ks in _DOC_KEYS.items() for k in ks.split("|")),
                       key=lambda t: -len(t[0]))
_NPWP = re.compile(r"(?<![\d.])\d{2,3}\.\d{3}\.\d{3}\.\d[-.]\d{3}\.\d{3}(?![\d.])")
_DOMAIN = re.compile(r"(?i)(?:www\.|@)([a-z0-9\-]{3,30})\.(?:co\.id|com|id|net|org|biz|co)\b")


def key_of(text: str) -> tuple[str, str] | None:
    """(header field a line names, whatever follows the name on that line) - the most specific name wins."""
    head, _, tail = clean(text).partition(":")
    best = None
    for part in re.split(r"[/|]", head):
        words = re.match(r"[A-Za-z .]*", part).group(0)
        for c, rest in ((compact(part), ""), (compact(words), part[len(words):])):
            hit = next(((k, f) for k, f in _DOC_KEY_LIST if c == k or (len(k) >= 9 and k in c and len(c) <= len(k) + 14)), None)
            if hit and (best is None or len(hit[0]) > len(best[0])):
                best = (*hit, tail.strip() or rest.strip())
    return (best[1], best[2]) if best else None


def _field_value(name: str, text: str):
    t = clean(text).strip(":# ")
    if name in ("transactionDate", "dueDate"):
        found = find_dates(t)
        return found[0].value.isoformat() if found else None
    if name in ("receiptNo", "customerId"):
        return t if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9/\-.]{3,29}", t) and re.search(r"\d", t) else None
    if name == "npwp":
        return t if 15 <= len(re.sub(r"\D", "", t)) <= 16 else None
    if name == "printedTotal":
        found = money_in(t)
        return found[-1][0] if found else None
    letters = sum(ch.isalpha() for ch in t)                                  # customerName
    return t if letters >= 3 and letters >= 0.6 * len(t.replace(" ", "")) and key_of(t) is None else None


def labelled(lines: list[Line], slope: float = 0.0) -> dict[str, tuple[object, Line]]:
    """Header fields found by their name -> (value, line the value was read from)."""
    keys = {id(ln): key_of(ln.text) for ln in lines}
    out: dict[str, tuple[object, Line]] = {}
    for ln in sorted(lines, key=lambda ln: (ln.cy, ln.x0)):
        if not keys[id(ln)] or keys[id(ln)][0] in out:
            continue
        name, tail = keys[id(ln)]
        value = _field_value(name, tail) if tail else None
        if value is not None:
            out[name] = (value, ln)
            continue
        on_row = sorted((o for o in lines if o is not ln and abs(o.cy - (ln.cy + slope * (o.cx - ln.cx))) <= 0.5 * max(o.h, ln.h)
                         and -0.5 * ln.h <= o.x0 - ln.x1 <= 15 * ln.h), key=lambda o: o.x0)[:1]
        under = sorted((o for o in lines if 0 < o.cy - ln.cy <= 4.5 * ln.h
                        and min(o.x1, ln.x1) - max(o.x0, ln.x0) >= 0.3 * min(o.w, ln.w)), key=lambda o: o.cy)[:3]
        for o in on_row + under:
            if keys[id(o)]:
                if keys[id(o)][0] == name:
                    continue            # the same name in the second language
                if any(o is x for x in under):
                    break               # the next field starts here
                continue                # a neighbouring column's name on the row: the value is underneath
            value = _field_value(name, o.text)
            if value is not None:
                out[name] = (value, o)
            break
    return out


@dataclass
class Row:
    idx: int
    lines: list[Line]                      # left to right
    label: str = ""
    role: str | None = None
    guessed: bool = False                  # role taken from the arithmetic, not from the name
    reads: list[Cand] = field(default_factory=list)   # readings of the amount; the first-pass one comes first
    extra: list[int] = field(default_factory=list)    # other numbers on the row (quantity, unit price)
    split: bool = False                    # the amount came back in pieces ('20' + '000')
    aside: bool = False                    # the amount is a header field's value, not a line of the bill

    @property
    def amount(self) -> int | None:
        return int(self.reads[0].value) if self.reads else None


def _label(text: str) -> str:
    """The words of a row: numbers go, names stay - including the digits inside them ('PB1', '10x')."""
    return clean(re.sub(r"[^A-Za-z0-9]+", " ", re.sub(r"(?i)\brp\b\.?|(?<![A-Za-z])\d[\d.,]*%?(?![A-Za-z\d])", " ", text)))


def _group(lines: list[Line], slope: float, med: float) -> list[list[Line]]:
    groups: list[list[Line]] = []
    for ln in sorted(lines, key=lambda ln: ln.cy - slope * ln.cx):
        for g in reversed(groups[-3:]):
            ref = g[0]
            # half the smaller of the two heights: a total printed large must not swallow the row below
            if abs(ln.cy - (ref.cy + slope * (ln.cx - ref.cx))) <= max(0.5 * min(ln.h, ref.h), 0.3 * med):
                g.append(ln)
                break
        else:
            groups.append([ln])
    return groups


def row_slope(seg: Segment) -> float:
    """Skew of the receipt's rows.

    A photo taken at an angle shifts the amount column against the names; read level,
    every name then pairs with the amount of the row below. The right slope is the one
    under which most lines find a partner on their row - starting from what the text
    boxes themselves say, and leaving that only for a clearly better fit."""
    lines = seg.lines
    med = float(np.median([ln.h for ln in lines]))
    lines = [ln for ln in lines if ln.h <= 2.5 * med]
    prior = seg_slope(seg)
    if not prior:
        wide = [ln.slope for ln in lines if ln.w >= 2.5 * max(ln.h, 1)]
        prior = float(np.clip(np.median(wide), -0.18, 0.18)) if wide else 0.0

    def paired(slope: float) -> int:
        return sum(len(g) - 1 for g in _group(lines, slope, med))

    best, best_n = prior, paired(prior)
    for slope in sorted(np.arange(-0.16, 0.161, 0.01), key=lambda v: abs(v - prior)):
        n = paired(float(slope))
        if n > best_n + 1:
            best, best_n = float(slope), n
    return best


def build_rows(seg: Segment, slope: float | None = None) -> list[Row]:
    lines = seg.lines
    if not lines:
        return []
    med = float(np.median([ln.h for ln in lines]))
    slope = row_slope(seg) if slope is None else slope
    lines = [ln for ln in lines if ln.h <= 2.5 * med]                         # drops vertical side print
    # 'Pay Cash 50.000' and 'Change:9,000' printed side by side are two rows, not one
    own = [ln for ln in lines if role_of(_label(ln.text)) in _MONEY_ROLES and money_in(ln.text, small=True)]
    groups = _group([ln for ln in lines if not any(ln is o for o in own)], slope, med) + [[ln] for ln in own]
    groups.sort(key=lambda g: g[0].cy - slope * g[0].cx)
    rows = []
    for i, g in enumerate(groups):
        g.sort(key=lambda ln: ln.x0)
        label = _label(" ".join(ln.text for ln in g))
        row = Row(i, g, label, role_of(label))
        cells = [[ln.text, ln] for ln in g]
        for a, b in zip(cells, cells[1:]):      # a large amount detected as two boxes: '20' + '000'
            if re.fullmatch(r"[\d.,]+", a[0].strip()) and re.fullmatch(r"[.,]?\d{3}", b[0].strip()) \
                    and b[1].x0 - a[1].x1 <= 0.8 * a[1].h:
                a[0], b[0], row.split = a[0].strip() + b[0].strip(), "", True
        found = [(v, raw, approx, ln) for text, ln in cells for v, raw, approx in money_in(text, small=row.role in _MONEY_ROLES)]
        if row.role == "count" and label.upper().startswith(("TOTAL", "JUMLAH")) and any(
                v >= 1000 and re.search(r"[.,]\d{3}", raw) for v, raw, *_ in found):
            row.role = "total"              # 'TOTAL (Qty 1.00) 30,000' counts and totals in one line
        if row.role in _BIG:
            found = [f for f in found if f[0] >= 1000 and not f[2]]
        elif row.role in ("tax", "service"):
            found = [(abs(v), *rest) for v, *rest in found if abs(v) >= 100]      # '5' and '10' are the percentage
        elif row.role in ("discount", "change"):
            found = [(abs(v), *rest) for v, *rest in found]
        if found and row.role != "count":
            v, raw, approx, ln = found[-1]                                 # amounts are flush right
            row.reads = [Cand(v, ln.score * (0.6 if approx else 1.0), "ocr", raw, [ln])]
            row.extra = [x for x, *_ in found[:-1]]
        rows.append(row)
    # some printers put the amount on the line below its name
    out: list[Row] = []
    for row in rows:
        prev = out[-1] if out else None
        if prev is not None and prev.role in _MONEY_ROLES and not prev.reads and row.reads and not row.label:
            prev.lines, prev.reads, prev.split = prev.lines + row.lines, row.reads, row.split
            if prev.role in _BIG and prev.reads[0].value < 1000:
                prev.reads = []
            continue
        row.idx = len(out)
        out.append(row)
    return out


def discover(rows: list[Row]) -> None:
    """Name the money rows whose label is unreadable from how the amounts add up.

    Receipts print the summary in a fixed order - subtotal, tax, TOTAL, cash, change -
    so three rows in that order with a - b = c or a + b = c are those rows. Item prices
    can add up by accident; a total is therefore never smaller than anything above it."""
    money = [r for r in rows if r.reads and not r.aside and r.role in (None, *_MONEY_ROLES)]
    by = lambda role: [r for r in money if r.role == role]       # noqa: E731
    free = lambda r: r.role is None                              # noqa: E731
    tops = lambda t: all(r.amount <= t.amount for r in money if r.idx < t.idx)   # noqa: E731

    def name(row: Row, role: str) -> None:
        if row.role is None:
            row.role, row.guessed = role, True

    if not by("change"):
        for t in by("total") or [r for r in money[len(money) // 3:] if free(r) and tops(r)]:
            below = [r for r in money if r.idx > t.idx]
            hit = next(((c, g) for c in below if free(c) or c.role == "cash" for g in below
                        if g.idx > c.idx and free(g) and g.amount > 0 and c.amount - g.amount == t.amount), None)
            if hit:
                name(t, "total"), name(hit[0], "cash"), name(hit[1], "change")
                break
    for b in by("subtotal")[-1:]:
        if by("total"):
            break
        named = [r for r in money if r.idx > b.idx and r.role in ("tax", "service", "rounding", "discount")]
        gross = b.amount + sum(-abs(r.amount) if r.role == "discount" else r.amount for r in named)
        service = sum(r.amount for r in named if r.role == "service")
        fits = lambda v: abs(v - gross) < 100 or any(abs(v - round((b.amount + service) * k)) <= 2 for k in _TAX)   # noqa: E731
        t = next((r for r in money if r.idx > b.idx and free(r) and fits(r.amount)), None)
        if t is not None:
            name(t, "total")
    if not by("total"):
        for t in [r for r in money[len(money) // 3:] if free(r) and tops(r)]:
            above = [r for r in money if r.idx < t.idx][-3:]
            for k in (3, 2):
                part = above[-k:]
                if len(part) == k and sum(r.amount for r in part) == t.amount and part[0].amount > sum(r.amount for r in part[1:]) \
                        and part[0].role in (None, "subtotal"):
                    name(t, "total"), name(part[0], "subtotal")
                    break
            if by("total"):
                break


_HEADING = re.compile(r"(?i)\b(keterangan|description|deskripsi|uraian|nama barang|item|qty|quantity|kuantitas|harga|price"
                      r"|satuan|unit|jumlah|amount|total|periode|period|pemakaian|usage)\b")


def _item_rows(rows: list[Row], at: int) -> list[Row]:
    """Priced lines above the summary block (the first named money row that carries an amount).

    Where the lines have column headings ('Keterangan | Qty | Harga | Total') they start below
    them: what is above is the header, and a postcode in an address there is not a price."""
    first = min(next((r.idx for r in rows if r.role in _MONEY_ROLES and r.reads), len(rows)), at)
    start = 0
    for r in rows[:first]:
        if not r.reads and len({w.lower() for w in _HEADING.findall(" ".join(ln.text for ln in r.lines))}) >= 2:
            start = r.idx + 1
    return [r for r in rows[start:first] if r.role is None and r.reads and not r.aside and len(r.label) >= 2]


def _kinds(codes) -> set[str]:
    """Independent kinds of evidence among check codes: the tax-rate check and the item
    sum restate the subtotal sum (items add up to the subtotal, not past it), and the
    two cash checks restate each other."""
    same = {"SUBTOTAL_PLUS_TAX_RATE_EQ_TOTAL": "SUBTOTAL_TAX_EQ_TOTAL", "ITEMS_SUM_EQ_TOTAL": "SUBTOTAL_TAX_EQ_TOTAL",
            "CASH_PAYMENT_EQ_TOTAL": "CASH_MINUS_CHANGE_EQ_TOTAL"}
    return {same.get(c, c) for c in codes}


@dataclass
class Proof:
    code: str
    value: int
    picks: dict[int, Cand]                 # row idx -> the reading this proof used
    weight: float
    implies: bool                          # strong enough to supply a total nobody could read
    text: str


class Receipt:
    doc_type, category = "receipt", "other"

    def score(self, seg: Segment) -> float:
        rows = build_rows(seg)
        discover(rows)          # a bare column of amounts that adds up is a receipt whose names were cut off
        named = sum(1 for r in rows if r.reads and r.role in ("total", "subtotal", "cash", "change"))
        priced = sum(1 for r in rows if r.reads and r.label)
        return 1.0 if named or priced >= 3 else 0.0

    def parse(self, seg: Segment, ctx: Ctx) -> Draft:
        d = Draft(self.doc_type, self.category, seg, amount_key="total")
        slope = row_slope(seg)
        rows = build_rows(seg, slope)
        info = labelled(seg.lines, slope)
        for r in rows:          # '3.330.000' under 'Total Amount' in the header table is not a line of the bill
            r.aside = bool(r.reads) and r.role is None and any(r.reads[0].lines[0] is ln for _, ln in info.values())
        printed = info.get("printedTotal")
        discover(rows)
        if ctx.refine:
            # a total printed large comes back mirrored or with digits merged, and a figure that
            # was hard to read once deserves a second look before the arithmetic judges it
            med = float(np.median([ln.h for ln in seg.lines]))
            proven = self._decide(rows, printed)[1]
            for r in rows:
                if r.role not in _MONEY_ROLES:
                    continue
                tall = r.role == "total" and max(ln.h for ln in r.lines) >= 1.3 * med
                weak = not r.reads or r.split or r.reads[0].conf < 0.9
                if tall or weak or (not proven and r.role in ("total", "subtotal", "cash", "change")):
                    r.reads += self._reread(seg, r, ctx, tall, slope)
            discover(rows)
        self._fill(d, rows, printed)
        self._items(d, rows)
        self._header(d, seg, rows, info)
        d.category = kind_of(seg.full_text) or d.category      # what it was for: internet, a meal, a hotel ...
        if d.category == "medical":
            medical_fields(d, seg)                             # patient, doctor, address, what was paid for
        return d

    # ------------------------------------------------------------------ money
    @staticmethod
    def _reread(seg: Segment, row: Row, ctx: Ctx, tall: bool, slope: float) -> list[Cand]:
        ref = max(row.lines, key=lambda ln: ln.h)
        nums = [ln for ln in row.lines if re.search(r"\d", ln.text) and not re.search(r"[A-Za-z]{3}", ln.text)]
        pad = 0.4 * ref.h
        words = [ln for ln in row.lines if not any(ln is n for n in nums)]
        whole = bool(row.reads) and not row.split
        if nums:
            # an amount that came back whole is re-read where it is; one that came back in pieces or
            # as nonsense may continue to the right of what was detected
            right = max(ln.x1 for ln in nums) if whole else max(seg.bbox[2], max(ln.x1 for ln in nums))
            zone = row_zone(ref, min(ln.x0 for ln in nums) - pad, right + pad, slope, above=0.7, below=0.7)
            reads = ctx.read_zone(zone, h=ref.h, stretch=(1.0, 1.5, 2.0) if tall or not whole else (1.0, 1.4))
        elif words and not row.reads:      # nothing detected at all: look right of the name
            zone = row_zone(ref, max(ln.x1 for ln in words) + pad, seg.bbox[2] + pad, slope, above=0.8, below=0.8)
            reads = ctx.read_zone(zone, h=ref.h, stretch=(1.0, 1.5, 2.0), tighten=1.5)
        else:               # name and amount in one box: nothing to cut out
            return []
        out = []
        for txt, sc in reads:
            found = [f for f in money_in(txt, small=True) if not (row.role in _BIG and (f[0] < 1000 or f[2]))]
            if found:
                v, raw, approx = found[-1]
                out.append(Cand(abs(v) if row.role != "rounding" else v, sc * (0.6 if approx else 0.95), "ocr_refined", raw, [ref]))
        return out

    @staticmethod
    def _proofs(rows: list[Row], at: int, printed=None) -> list[Proof]:
        """Every way the other rows say what the total printed on row `at` should be."""
        named = lambda role: [r for r in rows if r.role == role and r.reads]      # noqa: E731
        top = lambda r: sorted(r.reads, key=lambda c: -c.conf)[:3]                # noqa: E731
        out = []
        if printed is not None and not any(printed[1] is ln for r in rows if r.idx == at for ln in r.lines):
            # an invoice states its total twice: in the header table and under the lines
            out.append(Proof("PRINTED_TWICE_EQ_TOTAL", int(printed[0]), {}, 1.0, False, f"Total in the header {int(printed[0]):,}"))
        for c in named("cash"):
            for g in named("change"):
                out += [Proof("CASH_MINUS_CHANGE_EQ_TOTAL", int(x.value - y.value), {c.idx: x, g.idx: y}, 1.0, True,
                              f"Cash {int(x.value):,} - change {int(y.value):,}") for x in top(c) for y in top(g)
                        if x.value - y.value > 0]
            if not named("change"):
                out += [Proof("CASH_PAYMENT_EQ_TOTAL", int(x.value), {c.idx: x}, 0.5, False, f"Cash paid {int(x.value):,}")
                        for x in top(c)]
        for n in named("noncash"):
            out += [Proof("CARD_PAYMENT_EQ_TOTAL", int(x.value), {n.idx: x}, 1.0, False, f"Card payment {int(x.value):,}")
                    for x in top(n)]
        items = _item_rows(rows, at)
        # items that add up to a 'total' with another total printed further down have found the subtotal.
        # A single line counts too: its price and the total are two prints of one number.
        if items and not any(r.role == "total" and r.reads and r.idx > at for r in rows):
            out.append(Proof("ITEMS_SUM_EQ_TOTAL", sum(r.amount for r in items), {}, 0.8, False,
                             f"The {len(items)} item line{'s' if len(items) > 1 else ''}"))
        # subtotal (or an earlier 'TOTAL') plus everything printed between it and the total
        bases = [r for r in rows if r.idx < at and r.reads and r.role in ("subtotal", "total")]
        for b in bases[-2:]:
            between = [r for r in rows if b.idx < r.idx < at and r.reads and not r.aside
                       and r.role in (None, "tax", "service", "rounding", "discount", "fee")]
            signed = lambda r: -abs(r.amount) if r.role == "discount" else r.amount      # noqa: E731
            for x in top(b):
                # a tax printed for information is already in the prices; a row without a readable name
                # may be a charge or may be something else entirely - each sum is tried with and without
                # ... and as a charge or as a discount
                for drop, minus in (((), False), (("tax",), False), ((None,), False), (("tax", None), False),
                                    ((), True), (("tax",), True)):
                    if any(not any(r.role == k for r in between) for k in drop) or (minus and not any(r.role is None for r in between)):
                        continue
                    used = [r for r in between if r.role not in drop]
                    sign = lambda r: -abs(r.amount) if minus and r.role is None else signed(r)      # noqa: E731
                    value = int(x.value) + sum(sign(r) for r in used)
                    out.append(Proof("SUBTOTAL_TAX_EQ_TOTAL", value, {b.idx: x, **{r.idx: r.reads[0] for r in used}}, 1.0,
                                     None not in drop and not any(r.role is None for r in used),
                                     "Subtotal" + "".join(f" {'-' if sign(r) < 0 else '+'} {abs(r.amount):,}" for r in used)))
                service = sum(r.amount for r in between if r.role == "service")
                for k in _TAX:                      # holds even when the tax row itself is unreadable
                    out.append(Proof("SUBTOTAL_PLUS_TAX_RATE_EQ_TOTAL", round((x.value + service) * k), {b.idx: x},
                                     0.7, False, f"Subtotal + {k - 1:.0%} tax"))
        return out

    @classmethod
    def _decide(cls, rows: list[Row], printed=None):
        """-> (reading of the total or None, proofs that confirm it, proofs that could be evaluated, total row)."""
        totals = [r for r in rows if r.role == "total" and r.reads]
        best = None
        for order, t in enumerate(totals or [None]):
            proofs = cls._proofs(rows, t.idx if t else len(rows), printed)
            options: dict[int, Cand | None] = {}
            for c in (t.reads if t else []):
                if c.value not in options or (c.source == "ocr") > (options[c.value].source == "ocr"):
                    options[int(c.value)] = c
            for p in proofs:
                if p.implies:
                    options.setdefault(p.value, None)
            for value, read in options.items():
                # the cashier rounds: a printed total may sit up to Rp99 off the arithmetic - but then one
                # of the two is a round figure. 107,999 against 107,998 is a misread digit, not rounding.
                hits = [p for p in proofs if p.value == value or (read is not None and p.implies and abs(p.value - value) < 100
                                                                  and (value % 50 == 0 or p.value % 50 == 0))]
                codes = {p.code: p for p in sorted(hits, key=lambda p: abs(p.value - value), reverse=True)}
                same = sum(1 for c in (t.reads if t else []) if c.value == value)
                weight = 0.0 if read is None else read.conf * (1.0 if read.source == "ocr" else 0.9) + 0.03 * (same - 1)
                score = weight + sum(p.weight for p in codes.values()) + 0.01 * order
                if read is None and len(_kinds(codes)) < 2:
                    score = 0.3 + 0.01 * order      # one sum against a printed total: the print wins
                if best is None or score > best[0]:
                    best = (score, value, read, list(codes.values()), proofs, t)
        if best is None:
            return None, [], [], None
        _, value, read, hits, proofs, t = best
        if read is None and not any(p.implies for p in hits):
            return None, [], proofs, t
        return (read or Cand(value, 0.0, "derived", "", [])), hits, proofs, t

    @classmethod
    def _fill(cls, d: Draft, rows: list[Row], printed=None) -> None:
        total, hits, proofs, trow = cls._decide(rows, printed)
        chosen: dict[int, Cand] = {}
        for p in hits:
            if total is not None and p.value == total.value:    # a sum that only fits after rounding vouches for no row
                chosen.update(p.picks)
        if total is not None:
            value, codes = int(total.value), {p.code for p in hits}
            strong = sum(p.weight for p in hits) >= 0.7
            kinds = _kinds(codes)
            if total.source == "derived":
                # nobody could read it. Two independent sums that agree are as good as a reading; a
                # single one is wrong whenever a row in it was misread, so it is offered, not reported
                conf = 0.97 if len(kinds) >= 2 else 0.45
                raw = hits[0].text
            elif strong:
                conf, raw = (0.995 if total.source == "ocr" else 0.98), total.raw
            else:
                # Nothing on the receipt confirms this total. Read in the first pass it is reported, but a
                # person has to look: every wrong total seen so far was one of these. Read only on a
                # second look, or contradicted by what was paid by card, it is no more than a suggestion.
                paid = [int(c.value) for r in rows if r.role == "noncash" for c in r.reads[:1]]
                conf = min(total.conf, 0.84) if total.source == "ocr" and not (paid and value not in paid) else min(total.conf, 0.45)
                raw = total.raw
                if conf >= 0.5:
                    d.review.append("Total is not confirmed by any other figure on the receipt - confirm")
            if trow is not None and trow.guessed and len(kinds) < 2:
                # which row is the total was itself worked out from one sum: a suggestion, no more
                conf = min(conf, 0.45)
                d.review.append("Total found by the arithmetic only (its label is unreadable) - confirm")
            d.fields["total"] = FV(value, conf, total.source, raw, list(total.lines))
            guess = conf < 0.5
            for p in hits:      # a sum the total was taken from cannot also confirm it
                d.checks.append(Check(p.code, SKIP if guess else PASS,
                                      f"{p.text} = {value:,}: used to work out the total" if guess
                                      else f"{p.text} = total {value:,}", ["total"]))
            if guess:
                chosen.clear()
            # sums that could be worked out but land elsewhere: a row is unreadable or missing. That
            # sends the receipt to review unless something else already confirmed the total.
            for code in dict.fromkeys(p.code for p in proofs if p.implies and p.code not in codes):
                near = min((p for p in proofs if p.code == code), key=lambda p: abs(p.value - value))
                d.checks.append(Check(code, SKIP if strong else WARN, f"{near.text} = {near.value:,}, total reads {value:,}", ["total"]))
                if not strong:
                    # print and arithmetic disagree and nothing else takes a side: either a row of the
                    # sum is unreadable or the total is misread. That is for a person to settle.
                    d.fields["total"].confidence = min(d.fields["total"].confidence, 0.45)
        # The other money rows are reported only when a sum that came out right used them. A tax or
        # a change that nothing confirms is as likely the neighbouring row, a percentage or a misread.
        by_role: dict[str, list[Cand]] = {}
        for r in rows:
            if r.role in _FIELD and r.role != "total" and r.idx in chosen:
                by_role.setdefault(r.role, []).append(chosen[r.idx])
        for role, got in by_role.items():
            # several tax rows (PB1 and PPN) are one tax to whoever books the receipt
            value = int(sum(abs(c.value) if role == "discount" else c.value for c in got))
            if value or role not in ("fee", "rounding"):        # 'Stamp duty 0', 'Delivery 0' say nothing
                d.fields[_FIELD[role]] = FV(value, 0.995, got[0].source, " + ".join(c.raw for c in got),
                                             [ln for c in got for ln in c.lines])
        # 'Subject to Tax 3.000.000' above 'VAT 330.000': the tax is a fixed share of its base
        base = next((r.reads[0] for r in rows if r.role == "taxbase" and r.reads), None)
        tax = d.fields.get("tax")
        if base is not None and tax is not None:
            rate = next((k for k in _TAX if abs(round(base.value * (k - 1)) - tax.value) <= 1), None)
            if rate:
                d.fields["taxBase"] = FV(int(base.value), 0.995, base.source, base.raw, list(base.lines))
                d.checks.append(Check("TAX_EQ_RATE_OF_BASE", PASS, f"{rate - 1:.0%} of {int(base.value):,} = tax {tax.value:,}",
                                      ["tax", "taxBase"]))

    @staticmethod
    def _items(d: Draft, rows: list[Row]) -> None:
        items = _item_rows(rows, len(rows))
        d.segment.extra["item_names"] = []
        if not items:
            return
        out = []
        for r in items:
            # the description is the cell that is neither a number nor a date; a row read as one box keeps its words
            cell = next((ln for ln in r.lines if re.search(r"[A-Za-z]{2}", ln.text) and not money_in(ln.text)
                         and not find_dates(ln.text)), None)
            name = clean(cell.text) if cell else r.label
            # a description that wraps continues on the next line, flush with the first
            more = rows[r.idx + 1] if cell and r.idx + 1 < len(rows) else None
            if more and more.role is None and not more.reads and len(more.lines) == 1 and key_of(more.lines[0].text) is None \
                    and abs(more.lines[0].x0 - cell.x0) <= cell.h and re.search(r"[A-Za-z]{2}", more.lines[0].text):
                name += " " + clean(more.lines[0].text)
            item = {"name": name, "amount": r.amount}
            d.segment.extra.setdefault("item_names", []).append(name)        # for the description, checked or not
            qty = [v for ln in r.lines for v in map(int, re.findall(r"(?<![\d.,])\d{1,3}(?![\d.,])", ln.text)) if 0 < v < 1000]
            if qty and r.extra and qty[0] * r.extra[-1] == r.amount:
                item["qty"], item["unitPrice"] = qty[0], r.extra[-1]
            out.append(item)
        # the items were read one by one, the (sub)total separately: if they agree, both are right.
        # A list that does not add up is missing a line or has one from elsewhere, and is left out.
        key = "subtotal" if d.val("subtotal") is not None else "total"
        if d.val(key) is None or sum(r.amount for r in items) != d.val(key):
            return
        d.set("items", FV(out, min(r.reads[0].conf for r in items)))
        if not any(c.code == "ITEMS_SUM_EQ_TOTAL" for c in d.checks):
            d.checks.append(Check("ITEMS_SUM_EQ_TOTAL", PASS, f"The {len(items)} item line{'s' if len(items) > 1 else ''} "
                                  f"add{'' if len(items) > 1 else 's'} up to the {key}", ["items", key]))
        if d.fields[key].source in ("ocr", "ocr_refined") and d.fields[key].confidence >= 0.85:
            d.fields[key].confidence = max(d.fields[key].confidence, 0.97)

    @staticmethod
    def _header(d: Draft, seg: Segment, rows: list[Row], info: dict) -> None:
        # the date the document names as its own; a due date or a billing period is not when it was issued
        if "transactionDate" in info:
            ln = info["transactionDate"][1]
            _set_datetime(d, ln.text, [ln])
        else:
            best = None
            for ln in seg.lines:
                found = find_dates(ln.text)
                if len(found) != 1 or any(ln is info[k][1] for k in ("dueDate",) if k in info):
                    continue                    # two dates on a line are a period
                conf = min(found[0].confidence, ln.score)
                if best is None or conf > best[0]:
                    best = (conf, ln)
            if best:
                _set_datetime(d, best[1].text, [best[1]])
        for key in ("dueDate", "receiptNo", "customerName", "customerId", "npwp"):
            if key in info:
                value, ln = info[key]
                d.set(key, FV(value, ln.score * 0.95, raw=ln.text, lines=[ln]))
        # The shop name is the first line of the header. A photo cropped to the items has no header,
        # and its first line is something somebody ordered - so the header has to show itself first
        # (an address, a phone number, a date) before its top line is taken for a name.
        first = next((r.idx for r in rows if r.reads), len(rows))
        head = [clean(" ".join(ln.text for ln in r.lines)) for r in rows[:min(first, 6)]]
        if any(_NOT_A_NAME.search(t) or find_dates(t) for t in head[1:]):
            text = head[0]
            letters = sum(ch.isalpha() for ch in text)
            if letters >= 4 and letters >= 0.6 * len(text.replace(" ", "")) and not _NOT_A_NAME.search(text):
                d.set("merchantName", FV(text.upper(), min(ln.score for ln in rows[0].lines) * 0.8, lines=list(rows[0].lines)))
        # A logo is the worst-read line on the page ('HYPERINEI'); the web or mail address under it
        # spells the same name in plain type. When the two agree, the address is the better reading.
        site = next(((m.group(1).upper(), ln) for ln in seg.lines for m in [_DOMAIN.search(ln.text)] if m), None)
        if site and any(fuzz.ratio(compact(ln.text), site[0]) >= 70 for r in rows[:3] for ln in r.lines):
            d.set("merchantName", FV(site[0], site[1].score * 0.9, raw=site[1].text, lines=[site[1]]))
        if "npwp" not in d.fields:      # the number is unmistakable even when its caption was not read
            hit = next(((m.group(0), ln) for ln in seg.lines for m in [_NPWP.search(ln.text)] if m), None)
            if hit:
                d.set("npwp", FV(hit[0], hit[1].score * 0.9, raw=hit[1].text, lines=[hit[1]]))
        if "receiptNo" not in d.fields:
            for ln in seg.lines:
                m = _RECEIPT_NO.search(clean(ln.text))
                if m and re.search(r"\d", m.group(1)):
                    d.set("receiptNo", FV(m.group(1), ln.score * 0.9, raw=ln.text, lines=[ln]))
                    break
        paid = [r for r in rows if r.role in ("cash", "noncash") and r.reads]
        if paid:
            r = paid[0]
            method = "CASH" if r.role == "cash" else issuer_of(r.label) or clean(r.label).upper()
            d.set("paymentMethod", FV(method, min(ln.score for ln in r.lines) * 0.9, lines=list(r.lines)))
