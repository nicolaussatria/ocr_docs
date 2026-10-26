"""Page layout: label detection, label->value pairing and splitting a page that
holds many glued receipts into one segment per receipt.

Pipeline (all coordinates are in the upright frame):
  1. every OCR line is tested against a label lexicon (fuzzy, OCR-noise tolerant)
  2. each label grabs the value boxes to its right on the same *skew-corrected* row
     (stops at the next label, so values never jump into the neighbouring receipt)
  3. label+values form a unit; units are linked when they stack vertically with
     horizontal overlap -> connected components
  4. components are cut where a receipt header (PERTAMINA / SPBU / station code /
     JASAMARGA / PLAZA / BON KONTAN ...) follows body text, or where a non-repeatable
     field (e.g. 'Pulau/Pompa') occurs twice -> one segment per receipt
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from rapidfuzz import fuzz

from .normalize import clean, compact
from .ocr import Line, local_slope

# canonical key -> aliases (case/space/punctuation-insensitive)
LEXICON: dict[str, list[str]] = {
    "shift": ["Shift"],
    "transNo": ["No. Trans", "No Transaksi", "No. Transaksi"],
    "datetime": ["Waktu", "Tanggal", "Tgl"],
    "pump": ["Pulau/Pompa", "Pulau Pompa", "No. Pompa"],
    "operator": ["Operator", "Kasir"],
    "product": ["Jenis BBM", "Nama Produk", "Produk"],
    "volume": ["Volume"],
    "priceNonSubsidized": ["Harga Non Subsidi"],
    "subsidy": ["Subsidi Pemerintah"],
    "price": ["Harga Jual"],
    "unitPrice": ["Harga/Liter", "Harga per Liter"],
    "totalNonSubsidized": ["Tanpa Subsidi"],
    "paid": ["Dibayar Konsumen"],
    "totalPrice": ["Total Harga"],
    "entryTime": ["In", "Masuk", "Waktu Masuk", "Jam Masuk"],
    "exitTime": ["Out", "Keluar", "Waktu Keluar", "Jam Keluar"],
    "duration": ["Duration", "Durasi", "Lama Parkir"],
    "parkingFee": ["Biaya Parkir", "Tarif Parkir"],
    "total": ["Total"],
    "balance": ["Flazz Balance", "Sisa Saldo", "Saldo", "Sisa"],
    "cardNo": ["Flazz CardNo", "Nomor Kartu", "No Kartu", "CN", "Card No"],
    "paymentMethod": ["Metode Pembayaran", "Pembayaran"],
    "tid": ["TID"],
    "plate": ["Car", "No Pol", "Nopol", "Plat"],
    "serialNo": ["No seri"],
    "gate": ["Gerbang"],
    "receivedFrom": ["Sudah terima dari", "Telah terima dari", "Diterima dari"],
    "amountWords": ["Banyaknya uang", "Terbilang", "Uang sejumlah"],
    "purpose": ["Untuk pembayaran", "Guna membayar"],
    "amountTotal": ["Jumlah"],
    "topupNominal": ["Nominal"],
    "adminFee": ["Biaya Admin", "Biaya Administrasi"],
    "paymentTotal": ["Jumlah Bayar", "Total Bayar"],
    "phoneNumber": ["Nomor HP", "No. HP", "Nomor Ponsel", "Nomor Telepon"],
    "topupSerial": ["SN"],
    "description": ["Keterangan"],
}
REPEATABLE = {"subsidy", "datetime", "cardNo"}
# keys whose value must contain a digit – stops 'JUMLAH' from grabbing a caption
TEXT_KEYS = {"operator", "product", "paymentMethod", "plate", "gate", "receivedFrom", "amountWords", "purpose",
             "description"}
_ALIASES = sorted(
    ((key, a, compact(a)) for key, al in LEXICON.items() for a in al),
    key=lambda t: -len(t[2]),
)

_STATION_CODE = re.compile(r"^\s*(?:SPBU\s*[:.]?\s*)?(\d{7}|\d{2}[.,]\d{3}[.,]\d{2})\s*$")
_PRODUCTS = {"PERTAMAX", "PERTALITE", "PERTAMAXTURBO", "PERTAMAXGREEN", "DEXLITE", "PERTAMINADEX"}
_HEADER_WORDS = [("PERTAMINA", 86), ("JASAMARGA", 78), ("BONKONTAN", 80), ("KWITANSI", 80), ("KUITANSI", 80)]


@dataclass(frozen=True)
class LabelHit:
    key: str
    alias: str
    score: float
    inline: str          # text after the label on the same OCR line


def match_label(text: str) -> LabelHit | None:
    raw = clean(text)
    if not raw or raw[0] in ":;=":
        return None
    head, sep, tail = raw.partition(":")
    head_c = compact(head)
    tokens = re.findall(r"[A-Za-z0-9]+", raw)
    best: LabelHit | None = None
    for key, alias, ac in _ALIASES:
        if len(ac) <= 5:
            # short aliases must be a whole leading token sequence and must not be
            # followed by another word ('Total Penjualan' is not 'Total')
            n = len(re.findall(r"[A-Za-z0-9]+", alias))
            lead = [t.upper() for t in tokens[:n]]
            if compact("".join(lead)) != ac:
                continue
            nxt = tokens[n] if len(tokens) > n else ""
            if nxt and nxt.isalpha() and nxt.upper() not in {"RP"} and not sep:
                continue
            score = 100.0
        else:
            cand = head_c if sep else compact(raw)[: len(ac) + 1]
            score = max(
                fuzz.ratio(ac, cand[: len(ac)]),
                fuzz.ratio(ac, cand[: len(ac) + 1]),
                fuzz.ratio(ac, cand[: max(1, len(ac) - 1)]),
            )
            if score < (84 if len(ac) >= 7 else 88):
                continue
            if sep and len(head_c) > len(ac) + 4:      # 'Harga produk Pertamax saat ini:' etc.
                continue
        if best is None or score > best.score:
            inline = tail if sep else _after_alias(raw, len(ac))
            best = LabelHit(key, alias, score, inline.strip(" :;.,"))
            if score == 100:
                break
    return best


def _after_alias(raw: str, n_alnum: int) -> str:
    seen = 0
    for i, ch in enumerate(raw):
        if ch.isalnum():
            seen += 1
            if seen == n_alnum:
                return raw[i + 1:]
    return ""


def header_kind(text: str) -> str | None:
    """'anchor' for lines that start a receipt, 'header' for address/phone lines."""
    raw = clean(text)
    c = compact(raw)
    if not c:
        return None
    if c == "MCOMMERCE":
        return "anchor"
    if _STATION_CODE.match(raw):
        return "anchor"
    if c.startswith("SPBU") and len(c) >= 6:
        return "anchor"
    if c.startswith("PLAZA") or c.startswith("PARKIR") or c.startswith("SECUREPARKING"):
        return "anchor"
    if len(c) <= 16 and c not in _PRODUCTS:
        for word, cut in _HEADER_WORDS:
            if fuzz.ratio(c, word) >= cut:
                return "anchor"
    if c.startswith(("JL", "J1", "JI")) or c.startswith("TELP") or "INFOTOL" in c or c.startswith("PT"):
        return "header"
    return None


def _headerish(text: str) -> bool:
    k = header_kind(text)
    c = compact(text)
    return k == "anchor" or c.startswith(("JL", "TELP"))


@dataclass(eq=False)
class Unit:
    lines: list[Line]
    label: LabelHit | None = None
    kind: str | None = None          # 'anchor' | 'header' | None

    @property
    def x0(self): return min(ln.x0 for ln in self.lines)
    @property
    def x1(self): return max(ln.x1 for ln in self.lines)
    @property
    def y0(self): return min(ln.y0 for ln in self.lines)
    @property
    def y1(self): return max(ln.y1 for ln in self.lines)
    @property
    def h(self): return float(np.median([ln.h for ln in self.lines]))
    @property
    def key(self): return self.label.key if self.label else None

    @property
    def value_lines(self) -> list[Line]:
        return self.lines[1:] if self.label else []

    @property
    def value_text(self) -> str:
        parts = [self.label.inline] if self.label and self.label.inline else []
        parts += [ln.text for ln in self.value_lines]
        return clean(" ".join(parts))

    @property
    def text(self) -> str:
        return clean(" ".join(ln.text for ln in self.lines))

    @property
    def score(self) -> float:
        return min(ln.score for ln in self.lines)


@dataclass(eq=False)
class Segment:
    units: list[Unit]
    id: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def bbox(self):
        return (min(u.x0 for u in self.units), min(u.y0 for u in self.units),
                max(u.x1 for u in self.units), max(u.y1 for u in self.units))

    def get(self, key: str) -> list[Unit]:
        return [u for u in self.units if u.key == key]

    def first(self, key: str) -> Unit | None:
        g = self.get(key)
        return g[0] if g else None

    @property
    def lines(self) -> list[Line]:
        return [ln for u in self.units for ln in u.lines]

    @property
    def texts(self) -> list[str]:
        return [u.text for u in self.units]

    @property
    def full_text(self) -> str:
        return "\n".join(self.texts)


# --------------------------------------------------------------------------- pairing
def _row_neighbours(lines: list[Line], slopes: np.ndarray):
    """right[i] = (j, gap) nearest line to the right of i on the same row."""
    n = len(lines)
    right: list[tuple[int, float] | None] = [None] * n
    if n == 0:
        return right
    x0 = np.array([ln.x0 for ln in lines]); x1 = np.array([ln.x1 for ln in lines])
    cx = np.array([ln.cx for ln in lines]); cy = np.array([ln.cy for ln in lines])
    hh = np.array([max(ln.h, 1.0) for ln in lines])
    for i in range(n):
        dy = cy - (cy[i] + slopes[i] * (cx - cx[i]))
        tol = 0.45 * np.maximum(hh, hh[i])
        ok = (np.abs(dy) <= tol) & (x0 >= x1[i] - 0.6 * hh[i]) & (cx > cx[i])
        ok[i] = False
        if ok.any():
            idx = np.where(ok)[0]
            gaps = x0[idx] - x1[i]
            k = idx[int(np.argmin(gaps + np.abs(dy[idx]) * 0.5))]
            right[i] = (int(k), float(x0[k] - x1[i]))
    return right


def build_units(lines: list[Line]) -> list[Unit]:
    slopes = local_slope(lines)
    labels = [match_label(ln.text) for ln in lines]
    right = _row_neighbours(lines, slopes)
    # a label-looking line right next to the same label is its value ('Operator  OPERATOR') -
    # unless it carries a value of its own ('Pulau/Pompa : 11'): that is the neighbouring receipt
    for i, hit in enumerate(labels):
        if hit is not None and not hit.inline and right[i] is not None:
            j, gap = right[i]
            if labels[j] is not None and labels[j].key == hit.key and not labels[j].inline and gap <= 12 * lines[i].h:
                labels[j] = None
    used = [False] * len(lines)
    units: list[Unit] = []
    order = sorted(range(len(lines)), key=lambda i: (lines[i].cy, lines[i].x0))
    for i in order:
        hit = labels[i]
        if hit is None or used[i]:
            continue
        used[i] = True
        chain = [lines[i]]
        cur, first = i, True
        # a complete inline value ('No. Trans: 2766899') ends the unit; only an empty or
        # currency-only inline ('Biaya Parkir: Rp') continues to the boxes on the right
        complete = bool(re.search(r"[0-9A-Za-z]", re.sub(r"(?i)\b(rp|idr)\b\.?", "", hit.inline)))
        numeric = hit.key not in TEXT_KEYS
        while not complete and right[cur] is not None:
            j, gap = right[cur]
            # first value may sit far right of its label; continuations only for multi-box
            # text values ('PT. Prambanan' + 'Kencana') and only when nearly touching
            limit = (12.0 if first else 1.2) * max(lines[cur].h, 1.0)
            if used[j] or gap > limit or labels[j] is not None or _headerish(lines[j].text):
                break
            if first and hit.key == "volume" and re.fullmatch(r"[\s:()]*[Ll][\s:()]*", lines[j].text):
                # OCR may split 'Volume : (L) 4.000' into three boxes. The
                # unit marker is a bridge to the number, not the numeric value.
                chain.append(lines[j])
                used[j] = True
                cur = j
                continue
            if numeric and not first:
                break                      # a number never continues into the next column
            if first and numeric and not re.search(r"\d", lines[j].text):
                break
            chain.append(lines[j])
            used[j] = True
            cur, first = j, False
        units.append(Unit(chain, hit))
    for i in order:
        if not used[i]:
            units.append(Unit([lines[i]], None, header_kind(lines[i].text)))
    return units


# --------------------------------------------------------------------------- grouping
def _components(units: list[Unit]) -> list[list[Unit]]:
    n = len(units)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    b = np.array([[u.x0, u.y0, u.x1, u.y1, u.h] for u in units]) if units else np.zeros((0, 5))
    for i in range(n):
        x0, y0, x1, y1, h = b[i]
        ov = np.minimum(b[:, 2], x1) - np.maximum(b[:, 0], x0)
        narrow = np.minimum(b[:, 2] - b[:, 0], x1 - x0)
        vgap = np.maximum(b[:, 1], y0) - np.minimum(b[:, 3], y1)
        hm = (b[:, 4] + h) / 2
        link = (ov >= 0.3 * np.maximum(narrow, 1)) & (vgap <= 2.0 * hm)
        for j in np.where(link)[0]:
            if j > i:
                ra, rb = find(i), find(int(j))
                if ra != rb:
                    parent[rb] = ra
    groups: dict[int, list[Unit]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(units[i])
    return list(groups.values())


def _split(comp: list[Unit]) -> list[list[Unit]]:
    comp = sorted(comp, key=lambda u: (u.y0, u.x0))
    segs: list[list[Unit]] = [[]]
    has_body = False
    keys: dict[str, Unit] = {}
    for u in comp:
        cur = segs[-1]
        n_body = sum(1 for v in cur if v.kind is None)
        if u.kind == "anchor" and has_body and (any(v.label for v in cur) or n_body >= 3):
            segs.append([u]); has_body = False; keys = {}
            continue
        if u.key and u.key in keys and u.key not in REPEATABLE:
            first = keys[u.key]
            # cut at the widest vertical gap between the two occurrences
            pool = [v for v in cur if v.y0 >= first.y0] + [u]
            pool.sort(key=lambda v: v.y0)
            best_gap, cut = -1e9, len(pool) - 1
            for k in range(1, len(pool)):
                g = pool[k].y0 - max(v.y1 for v in pool[:k])
                if g > best_gap:
                    best_gap, cut = g, k
            moved = pool[cut:]
            moved_ids = {id(v) for v in moved}
            segs[-1] = [v for v in cur if id(v) not in moved_ids]
            segs.append(sorted(moved, key=lambda v: (v.y0, v.x0)))
            keys = {v.key: v for v in segs[-1] if v.key}
            has_body = any(v.kind is None for v in segs[-1])
            continue
        cur.append(u)
        if u.key:
            keys.setdefault(u.key, u)
        if u.kind is None:
            has_body = True
    return [s for s in segs if s]


def _bbox_gap(a, b) -> float:
    dx = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    dy = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    return (dx * dx + dy * dy) ** 0.5


def _attach_mobile_footers(segments: list[Segment]) -> None:
    """Keep a bank tax footer with its top-up despite a larger paragraph gap.

    A four-line footer otherwise looks like a separate generic receipt. Limit
    this to footer-only text immediately below an aligned m-Commerce segment.
    """
    for footer in list(segments):
        texts = [compact(u.text) for u in footer.units]
        if any(u.label for u in footer.units) or not any(t.startswith("BIAYATERMASUKPPN") for t in texts):
            continue
        if not all(t.startswith(("BIAYATERMASUKPPN", "PTBANK", "MENARA", "NPWP")) for t in texts):
            continue
        h = float(np.median([u.h for u in footer.units]))
        x0, y0, _, _ = footer.bbox
        candidates = []
        for receipt in segments:
            if not any(compact(u.text) == "MCOMMERCE" for u in receipt.units):
                continue
            if not (receipt.first("topupNominal") or receipt.first("paymentTotal")):
                continue
            rx0, _, _, ry1 = receipt.bbox
            gap = y0 - ry1
            if 0 <= gap <= 6 * h and abs(x0 - rx0) <= 2 * h:
                candidates.append((gap, receipt))
        if candidates:
            _, target = min(candidates, key=lambda p: p[0])
            target.units = sorted(target.units + footer.units, key=lambda u: (u.y0, u.x0))
            segments.remove(footer)


def _attach_amount_columns(segments: list[Segment]) -> list[Segment]:
    """Give a detached column of numbers back to the rows it belongs to.

    Receipts print names on the left and amounts flush right (and quantities flush
    left); with a wide gap between them nothing overlaps vertically and the column
    becomes a 'receipt' of its own. A block that is nearly all numbers joins the block
    beside it whose text lines share its rows. A neighbouring receipt never qualifies:
    it is not all numbers."""
    def numeric(text: str) -> bool:
        c = re.sub(r"[\s.,:;\-]|(?i:rp)", "", text)
        return bool(c) and sum(ch.isdigit() for ch in c) >= 0.6 * len(c)

    out = list(segments)
    for s in sorted(segments, key=lambda s: s.bbox[0]):
        lines = s.lines
        if len(lines) < 2 or sum(numeric(ln.text) for ln in lines) < 0.75 * len(lines):
            continue
        if any(u.label for u in s.units):
            continue        # captioned ('No seri :', 'CN:'): a receipt of mostly numbers, like a toll slip

        h = float(np.median([ln.h for ln in lines]))
        best, best_hits = None, 0
        for c in out:
            if c is s:
                continue
            if c.bbox[0] < s.bbox[0] and s.bbox[0] - c.bbox[2] <= 30 * h:       # amounts, right of the names
                side, near = 1, [m for m in c.lines if m.x1 < s.bbox[0] + h]
            elif c.bbox[2] > s.bbox[2] and c.bbox[0] - s.bbox[2] <= 4 * h:      # quantities, left of them
                side, near = -1, [m for m in c.lines if m.x0 > s.bbox[2] - h]
            else:
                continue
            words = [m for m in near if re.search(r"[A-Za-z]{2}", m.text)]
            long = [m.slope for m in near if m.w >= 4 * max(m.h, 1)]
            # a photo taken at an angle shifts the column against its names: try the skews it could have
            for slope in {0.0, float(np.clip(np.median(long), -0.18, 0.18)) if long else 0.0,
                          *(round(float(v), 2) for v in np.arange(-0.14, 0.141, 0.02))}:
                def same_row(a, b):
                    return abs(a.cy - (b.cy + slope * (a.cx - b.cx))) <= 0.5 * max(a.h, b.h)

                def is_open(m):     # the name has no number of its own on the side the column is on
                    return not any(n is not m and numeric(n.text) and same_row(n, m) and (n.cx - m.cx) * side > 0
                                   for n in c.lines)

                # every number sits on a row of the block, and at least two of those rows are names
                # still waiting for one: a complete receipt next door has none, whatever the skew
                waiting = sum(1 for ln in lines if any(same_row(ln, m) and is_open(m) for m in words))
                hits = sum(1 for ln in lines if any(same_row(ln, m) for m in near)) if waiting >= 2 else 0
                if hits > best_hits:
                    best, best_hits = c, hits
        if best is not None and best_hits >= max(2, 0.6 * len(lines)):
            best.units = sorted(best.units + s.units, key=lambda u: (u.y0, u.x0))
            out.remove(s)
    return out


def _attach_fuel_footers(segments: list[Segment]) -> None:
    """Subsidy explanations and their CASH row belong to the fuel body above."""
    for footer in list(segments):
        text = compact(footer.full_text)
        if not ("SUBSIDI" in text and ("PEMERINTAH" in text or "UNAUDITED" in text or "ESTIMASI" in text)):
            continue
        if any(footer.first(k) for k in ("datetime", "pump", "volume", "paid", "totalPrice")):
            continue
        if any(u.kind == "anchor" for u in footer.units):
            continue
        x0, y0, x1, _ = footer.bbox
        h = float(np.median([l.h for l in footer.lines]))
        candidates = []
        for top in segments:
            if top is footer or not top.first("volume") or not (top.first("paid") or top.first("totalPrice")):
                continue
            tx0, ty0, tx1, ty1 = top.bbox
            # A stray neighbouring fragment may already stretch the component
            # below its payment. Anchor the gap to the actual fuel body instead.
            body_end = max(u.y1 for u in top.units if u.key in {"volume", "paid", "totalPrice"})
            overlap = min(x1, tx1) - max(x0, tx0)
            if ty0 < y0 and -2 * h <= y0 - body_end <= 8 * h and overlap >= .65 * (x1 - x0):
                candidates.append((abs(y0 - body_end), top))
        if candidates:
            _, target = min(candidates, key=lambda item: item[0])
            target.units = sorted(target.units + footer.units, key=lambda u: (u.y0, u.x0))
            segments.remove(footer)


_SUMMARY = re.compile(r"(?i)^\W*(sub\s*total|grand\s*total|total(?!\s*(item|qty|quantity|produk))|jumlah(?!\s*(item|barang))"
                      r"|tunai|cash|kembali|change|pb1|ppn|pajak|tax|service|diskon|discount)\b")


_PAYMENT = re.compile(r"(?i)^\W*(debit|kredit|credit|card|kartu|qris|ovo|gopay|dana|shopeepay|bca|mandiri|bni|bri|edc)\b")


def _join_summary_blocks(segments: list[Segment]) -> None:
    """Put a receipt's totals back under its items when a blank band separates them.

    Only a block that opens with a summary row (Total, Tunai, Kembali ...) moves, and
    only up to a block right above it that has no such row yet: the next receipt on a
    page opens with a shop name, and a complete receipt already has its total. A block
    of bare names (their amounts were swept into the block above) always moves, and so
    does one that holds nothing but totals and payments close under another block:
    'TOTAL / CASH / CHANGE' with no item and no shop name is the end of the receipt
    above it, not a receipt - left alone it would be counted a second time."""
    for low in sorted(segments, key=lambda s: s.bbox[1]):
        if not any(_SUMMARY.match(u.text) for u in low.units[:3]) or any(u.kind == "anchor" for u in low.units):
            continue
        h = float(np.median([u.h for u in low.units]))
        x0, y0, x1, _ = low.bbox
        # names without a single amount are not a receipt: their amounts sit in the block they hang under
        bare = not any(re.search(r"\d[.,]\d{3}|\d{4}", u.text) for u in low.units)
        tail = all(_SUMMARY.match(ln.text) or _PAYMENT.match(ln.text) or not re.search(r"[A-Za-z]{3}", ln.text)
                   for ln in low.lines)
        above = []
        for top in segments:
            tx0, ty0, tx1, ty1 = top.bbox
            overlap = min(x1, tx1) - max(x0, tx0)
            if top is low or ty0 >= y0 or y0 - ty1 > 6 * h or overlap < 0.5 * min(x1 - x0, tx1 - tx0):
                continue
            if bare or (tail and y0 - ty1 <= 3 * h) or not any(_SUMMARY.match(u.text) for u in top.units):
                above.append((y0 - ty1, top))
        if above:
            _, top = min(above, key=lambda t: t[0])
            top.units = sorted(top.units + low.units, key=lambda u: (u.y0, u.x0))
            segments.remove(low)


def segment_page(lines: list[Line]) -> list[Segment]:
    units = build_units(lines)
    segments: list[Segment] = []
    for comp in _components(units):
        for part in _split(comp):
            segments.append(Segment(sorted(part, key=lambda u: (u.y0, u.x0))))
    _attach_mobile_footers(segments)
    segments = _attach_amount_columns(segments)
    _join_summary_blocks(segments)
    _attach_fuel_footers(segments)
    # orphan fragments (a lone 'Rp.' or a handwritten amount floating beside a form)
    # join the closest real segment when they sit within a few text heights of it
    core = [s for s in segments
            if sum(1 for u in s.units if u.label) >= 2 or len(s.units) >= 4 or any(u.kind == "anchor" for u in s.units)]
    if core:
        for s in [s for s in segments if s not in core]:
            h = float(np.median([u.h for u in s.units]))
            dist, tgt = min(((_bbox_gap(s.bbox, c.bbox), c) for c in core), key=lambda t: t[0])
            if dist <= 4 * h:
                tgt.units = sorted(tgt.units + s.units, key=lambda u: (u.y0, u.x0))
                segments.remove(s)
    # reading order: rows of receipts top->bottom, then left->right
    segments.sort(key=lambda s: (round(s.bbox[1] / 150), s.bbox[0]))
    return segments
