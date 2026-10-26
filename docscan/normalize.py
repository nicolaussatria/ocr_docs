"""Text normalisation for Indonesian receipts: amounts, dates, times, volumes,
card numbers, plate numbers and 'terbilang' (amount in words)."""
from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime, time

from rapidfuzz import fuzz, process

# --------------------------------------------------------------------------- basics
_CONFUSABLE = str.maketrans({
    "O": "0", "o": "0", "D": "0", "Q": "0", "U": "0",
    "I": "1", "l": "1", "|": "1", "i": "1", "!": "1", "J": "1", "]": "1", "[": "1",
    "Z": "2", "z": "2", "S": "5", "s": "5", "B": "8", "G": "6", "b": "6", "g": "9", "q": "9",
})


def clean(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")   # full-width '：' -> ':'
    return re.sub(r"\s+", " ", s).strip()


def compact(s: str) -> str:
    """Upper-case alphanumerics only – used for fuzzy label comparison."""
    return re.sub(r"[^A-Z0-9]", "", clean(s).upper())


def strip_value(s: str) -> str:
    return clean(s).lstrip(":;.,=- ").strip()


def fix_digits(token: str) -> str:
    """Replace look-alike letters inside a token that is mostly digits."""
    core = re.sub(r"[.,\s]", "", token)
    if not core:
        return token
    digits = sum(ch.isdigit() for ch in core)
    grouped = re.fullmatch(r"\d[0-9OoDIl|SZB]{0,2}(?:[.,][0-9OoDIl|SZB]{3})+", token.strip()) is not None
    if digits / len(core) < 0.6 and not grouped:
        return token
    return token.translate(_CONFUSABLE)


# --------------------------------------------------------------------------- amounts
@dataclass(frozen=True)
class Amount:
    value: int | None          # None when a digit is unreadable
    raw: str
    pattern: str               # digits with '?' for unreadable characters
    approx: bool = False       # e.g. '35.00' that is probably '35.000'


_AMT_TOKEN = re.compile(r"\d[\d.,]*\d|\d")
# 485[1 / 485(1 -> 485?1. A closing bracket stays a separator: '(021)22329586' is a phone number.
_WILD = re.compile(r"\d+(?:[^\d\s.,:/)\-][\d]+)+")


def parse_amounts(text: str, *, min_value: int = 0) -> list[Amount]:
    """All IDR amounts in `text`, left to right.

    IDR has no cents in practice: '33,000,00' -> 33000, '15,950' and '15.950' -> 15950.
    Tokens such as '35.00' (handwritten '35.000' with a lost zero) are returned with
    approx=True so that a constraint can decide.
    """
    s = clean(text)
    s = re.sub(r"(?i)\bRp\.?\s*", " ", s)
    # repair look-alike letters inside numeric tokens
    s = re.sub(r"[0-9OoDIl|SZB]{2,}[0-9.,OoDIl|SZB]*", lambda m: fix_digits(m.group(0)), s)
    out: list[Amount] = []
    taken = []
    for m in _WILD.finditer(s):
        pat = re.sub(r"[^\d]", "?", m.group(0))
        out.append(Amount(None, m.group(0), pat))
        taken.append(m.span())
    for m in _AMT_TOKEN.finditer(s):
        if any(a <= m.start() < b for a, b in taken):
            continue
        amt = _interpret(m.group(0))
        if amt is not None and (amt.value is None or amt.value >= min_value):
            out.append(amt)
    out.sort(key=lambda a: s.find(a.raw))
    return out


def _interpret(tok: str) -> Amount | None:
    tok = tok.strip(".,")
    if not tok:
        return None
    groups = re.split(r"[.,]", tok)
    if len(groups) == 1:
        return Amount(int(tok), tok, tok)
    # trailing cents ',00' / '.00'
    if len(groups) >= 3 and len(groups[-1]) == 2 and all(len(g) == 3 for g in groups[1:-1]):
        digits = "".join(groups[:-1])
        return Amount(int(digits), tok, digits)
    if all(len(g) == 3 for g in groups[1:]) and 1 <= len(groups[0]) <= 3:
        digits = "".join(groups)
        return Amount(int(digits), tok, digits)
    if len(groups) == 2 and len(groups[1]) == 2:
        if len(groups[0]) >= 4:
            return Amount(int(groups[0]), tok, groups[0])        # '70000.00': cents, nothing is missing
        # '35.00' -> very likely '35.000' on an IDR receipt (decimals are not used)
        if groups[1] == "00":
            return Amount(int(groups[0]) * 1000, tok, groups[0] + "000", approx=True)
        return Amount(int(groups[0]), tok, groups[0], approx=True)
    digits = "".join(groups)
    return Amount(int(digits), tok, digits, approx=True)


def pattern_matches(pattern: str, value: int) -> bool:
    s = str(value)
    return len(s) == len(pattern) and all(p == "?" or p == c for p, c in zip(pattern, s))


def parse_printed_idr(text: str) -> int | None:
    """One printed bank amount, with optional thousands groups and zero cents.

    Unlike handwritten amounts, '35.00' means 35 here. Require the whole value
    to match: partial digits, nonzero cents and malformed grouping need review.
    """
    s = clean(text).lstrip(":= ")
    s = re.sub(r"(?i)^(?:rp|idr)\.?\s*", "", s)
    m = re.fullmatch(r"(\d{1,3}(?P<sep>[.,])\d{3}(?:(?P=sep)\d{3})*|\d+)(?:[.,]00)?", s)
    return int(re.sub(r"[.,]", "", m.group(1))) if m else None


def one_digit_repairs(value: int, ok) -> list[int]:
    """Values that differ from `value` in exactly one digit and satisfy `ok(v)`."""
    s = str(value)
    found = []
    for i, ch in enumerate(s):
        for d in "0123456789":
            if d == ch or (i == 0 and d == "0"):
                continue
            v = int(s[:i] + d + s[i + 1:])
            if ok(v):
                found.append(v)
    return found


# --------------------------------------------------------------------------- re-read amount zones
_GLYPH_DIGIT = str.maketrans({"O": "0", "o": "0", "D": "0", "Q": "0", "I": "1", "l": "1", "i": "1", "|": "1", "!": "1",
                              ")": "1", "(": "1", "]": "1", "[": "1", "t": "1", "T": "1", "S": "5", "B": "8", "Z": "2"})
_EDGE = ".,:;-_'\"`~=*"


def glyph_amount(text: str) -> tuple[int | None, int]:
    """Printed amount read through a marker, right of 'Rp': 'Rpi1000' -> (11000, 1).

    Returns (value, number of look-alike substitutions). None when the 'Rp' prefix is
    damaged ('R1000' may have lost a digit together with the 'p') or when anything
    but digits and look-alike glyphs is left."""
    t = re.sub(r"\s+", "", clean(text)).strip(_EDGE)
    m = re.match(r"(?:R[pPoOe0]|[pP])\.?", t)
    if m:
        t = t[m.end():].strip(_EDGE)
    elif t[:1].isalpha():
        return None, 0
    fixed = t.translate(_GLYPH_DIGIT)
    subs = sum(a != b for a, b in zip(t, fixed))
    if re.fullmatch(r"\d{1,3}(?:[.,]\d{3})+", fixed):
        fixed = re.sub(r"[.,]", "", fixed)
    if not fixed.isdigit() or subs > 2 or 2 * subs > len(fixed):
        return None, 0
    return int(fixed), subs


def hand_amount(text: str) -> Amount | None:
    """Handwritten IDR amount: '600.000,-', 'Rp. 35.000', '6oo.000'.

    Thousands are often scribbled into loops ('35.00', '35.0.'): those come back with
    approx=True, the legible leading part times 1000. So do bare digits ('3500'):
    people write the separator, so a reading without one has lost a character and
    may have lost a digit with it. A number followed by more digits than a trailing
    ',-' can explain ('1.250.00') is not guessed at."""
    t = re.sub(r"(?i)^\W*rp", "", re.sub(r"\s+", "", clean(text))).strip(_EDGE)
    t = t.translate(str.maketrans({"o": "0", "O": "0", "l": "1", "I": "1"}))
    m = re.match(r"(\d{1,3}(?:[.,]\d{3})+)(.{0,2})$", t)
    if m and not re.match(r"[.,]\d", m.group(2)):
        digits = re.sub(r"[.,]", "", m.group(1))
        return Amount(int(digits), m.group(1), digits)
    m = re.match(r"(\d{4,9})\D{0,2}$", t)
    if m:
        return Amount(int(m.group(1)), m.group(1), m.group(1), approx=True)
    m = re.match(r"([1-9]\d{0,2})[.,][0.,]{0,3}\D?$", t)
    if m:
        return Amount(int(m.group(1)) * 1000, t, m.group(1) + "000", approx=True)
    return None


# --------------------------------------------------------------------------- volume / ints
def parse_volume(text: str) -> float | None:
    s = clean(text)
    s = re.sub(r"(?i)\(\s*l\s*\)", " ", s)                # '(L) 18.8'
    s = re.sub(r"[OoD](?=[.,]?\d)", "0", s)
    m = re.search(r"(\d{1,3})\s*[.,]\s*(\d{1,3})", s) or re.search(r"(\d{1,3})(?!\d)", s)
    if not m:
        return None
    try:
        return float(m.group(1) + ("." + m.group(2) if m.lastindex and m.lastindex >= 2 else ""))
    except ValueError:
        return None


def volume_decimals(text: str) -> int:
    m = re.search(r"\d[.,](\d+)", clean(text))
    return len(m.group(1)) if m else 0


def parse_int(text: str) -> int | None:
    m = re.search(r"\d+", fix_digits(strip_value(text)))
    return int(m.group(0)) if m else None


# --------------------------------------------------------------------------- dates
_MONTHS = {
    "JAN": 1, "JANUARI": 1, "JANUARY": 1, "FEB": 2, "FEBRUARI": 2, "FEBRUARY": 2, "MAR": 3, "MARET": 3,
    "MARCH": 3, "APR": 4, "APRIL": 4, "MEI": 5, "MAY": 5, "JUN": 6, "JUNI": 6, "JUNE": 6, "JUL": 7,
    "JULI": 7, "JULY": 7, "AGU": 8, "AGS": 8, "AGT": 8, "AUG": 8, "AGUSTUS": 8, "AUGUST": 8, "SEP": 9,
    "SEPT": 9, "SEPTEMBER": 9, "OKT": 10, "OCT": 10, "OKTOBER": 10, "OCTOBER": 10, "NOV": 11,
    "NOVEMBER": 11, "DES": 12, "DEC": 12, "DESEMBER": 12, "DECEMBER": 12,
}

# A year is 19xx/20xx, or two digits - also when the printer's space before the time was lost:
# '11/02/1819:21' is 11/02/18 19:21, not the year 1819.
_GLUED_TIME = r"(?=\d{1,2}:\d{2})"
_YEAR = rf"((?:19|20)\d{{2}}(?!\d{{3}})|\d{{2}}(?!\d)|\d{{2}}{_GLUED_TIME})"
_NUM_DATE = re.compile(r"(?<!\d)(\d{1,2})\s*([/.\-])\s*(\d{1,2})\s*([/.\-])\s*" + _YEAR)
_ISO_DATE = re.compile(rf"(?<!\d)(20\d{{2}})\s*[/.\-]\s*(\d{{1,2}})\s*[/.\-]\s*(\d{{1,2}})(?:(?!\d)|{_GLUED_TIME})")  # 2026-10-12
_PART_DMY = re.compile(r"(?<!\d)(\d{1,2})\s*/\s*(\d{1,2})\s*/\s*(20\d)(?!\d)")      # 02/10/202  (year cut)
_PART_MY = re.compile(r"(?<![\d/])(\d{1,2})\s*/\s*(20\d{2})(?!\d)")                    # 09/2026    (day cut)
_TXT_DATE = re.compile(r"(?<!\d)(\d{1,2})\s*[\-\s/.]\s*([A-Za-z0-9]{3,9})\s*[\-\s/.]\s*" + _YEAR)   # 02/JAN/2017, 28 Mar 18
_TXT_MDY = re.compile(r"(?<![A-Za-z])([A-Za-z]{3,9})\.?\s+(\d{1,2}),?\s+((?:19|20)\d{2})(?!\d)")              # OCT 3, 2016
_TIME = re.compile(r"(\d{1,2})\s*[:.]\s*(\d{2})(?:\s*[:.]\s*(\d{2}))?(?!\d)")   # '1:36:414' is not a time


@dataclass(frozen=True)
class FoundDate:
    value: date
    raw: str
    span: tuple[int, int]
    confidence: float


def _month(tok: str) -> int | None:
    t = tok.upper().replace("0", "O")
    if t in _MONTHS:
        return _MONTHS[t]
    hit = process.extractOne(t, list(_MONTHS), scorer=fuzz.ratio, score_cutoff=75)
    return _MONTHS[hit[0]] if hit else None


def _matches(rx: re.Pattern, s: str):
    """Every match, overlapping ones included: in 'ORD #07 -REG #19- 10/03/2018' the first try,
    '19- 10/03', must not use up the digits of the real date."""
    pos = 0
    while (m := rx.search(s, pos)) is not None:
        yield m
        pos = m.start() + 1


def find_dates(text: str) -> list[FoundDate]:
    s = clean(text)
    out: list[FoundDate] = []
    for m in _matches(_NUM_DATE, s):
        d, mo, y = int(m.group(1)), int(m.group(3)), int(m.group(5))
        conf = 0.98
        if y < 100:          # 'dd/mm/yy' is often a booth/shift code on toll & parking slips
            if m.group(2) != m.group(4):    # ... and 'TU21- 6/72' is only a code
                continue
            y += 2000
            conf = 0.6
        if mo > 12 and d <= 12:      # month/day swapped
            d, mo, conf = mo, d, 0.7
        try:
            out.append(FoundDate(date(y, mo, d), m.group(0), m.span(), conf))
        except ValueError:
            continue
    for m in _matches(_ISO_DATE, s):
        try:
            out.append(FoundDate(date(int(m.group(1)), int(m.group(2)), int(m.group(3))), m.group(0), m.span(), 0.97))
        except ValueError:
            continue
    for m in _matches(_TXT_DATE, s):
        y = int(m.group(3))
        # a month name makes 'dd MON yy' a date, not a code - a real one: '1 SEPATU 25' is a shoe
        mo = _month(m.group(2)) if y >= 100 else _MONTHS.get(m.group(2).upper().replace("0", "O"))
        if not mo:
            continue
        try:                # ... but its 2-digit year is still a person's to confirm
            out.append(FoundDate(date(y + 2000 if y < 100 else y, mo, int(m.group(1))), m.group(0), m.span(),
                                 0.84 if y < 100 else 0.95))
        except ValueError:
            continue
    for m in _matches(_TXT_MDY, s):
        mo = _MONTHS.get(m.group(1).upper())
        if not mo:
            continue
        try:
            out.append(FoundDate(date(int(m.group(3)), mo, int(m.group(2))), m.group(0), m.span(), 0.95))
        except ValueError:
            continue
    kept: list[FoundDate] = []      # of two readings of the same characters, the more certain one
    for f in sorted(out, key=lambda f: (-f.confidence, f.span[0])):
        if all(f.span[1] <= k.span[0] or f.span[0] >= k.span[1] for k in kept):
            kept.append(f)
    kept.sort(key=lambda f: f.span[0])
    return kept


def find_time(text: str, after: int = 0) -> time | None:
    s = clean(text)[after:]
    for m in _TIME.finditer(s):
        hh, mm, ss = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
        if hh < 24 and mm < 60 and ss < 60 and (m.group(3) or len(m.group(1)) == 2):
            half = re.match(r"\s*([AaPp])\.?[Mm]\b", s[m.end():])        # '5:02:32 PM'
            if half and 1 <= hh <= 12:
                hh = hh % 12 + (12 if half.group(1) in "Pp" else 0)
            return time(hh, mm, ss)
    return None


@dataclass(frozen=True)
class PartialDate:
    day: int | None
    month: int
    year: int | None
    raw: str


def find_partial_date(text: str) -> PartialDate | None:
    """Dates with a cut-off part (overlapping receipts, torn edges)."""
    s = clean(text)
    m = _PART_DMY.search(s)
    if m and 1 <= int(m.group(2)) <= 12 and 1 <= int(m.group(1)) <= 31:
        return PartialDate(int(m.group(1)), int(m.group(2)), None, m.group(0))
    m = _PART_MY.search(s)
    if m and 1 <= int(m.group(1)) <= 12:
        return PartialDate(None, int(m.group(1)), int(m.group(2)), m.group(0))
    return None


def fix_letters(text: str) -> str:
    """'SPBU C1RACAS' -> 'SPBU CIRACAS': digits inside mostly-alphabetic words."""
    def fix(m):
        tok = m.group(0)
        letters = sum(ch.isalpha() for ch in tok)
        if letters >= 3 and letters / len(tok) >= 0.7:
            return tok.translate(str.maketrans({"1": "I", "0": "O", "5": "S", "8": "B"}))
        return tok
    return re.sub(r"[A-Za-z0-9]+", fix, text)


def find_datetime(text: str) -> tuple[date | None, time | None, float]:
    ds = find_dates(text)
    if not ds:
        return None, find_time(text), 0.0
    d = max(ds, key=lambda f: f.confidence)
    s = clean(text)
    t = find_time(s[: d.span[0]] + " " * (d.span[1] - d.span[0]) + s[d.span[1]:])
    return d.value, t, d.confidence


def parse_duration_minutes(text: str) -> int | None:
    s = clean(text).lower().replace("0days", "0 days").replace("odays", "0 days")
    d = re.search(r"(\d+)\s*(?:d|day|days|hari)\b", s)
    h = re.search(r"(\d+)\s*(?:h|hr|hrs|hour|hours|jam)", s)
    m = re.search(r"(\d+)\s*(?:m|min|mins|minute|minutes|menit)\b", s)
    if not (h or m or d):
        return None
    return (int(d.group(1)) * 1440 if d else 0) + (int(h.group(1)) * 60 if h else 0) + (int(m.group(1)) if m else 0)


# --------------------------------------------------------------------------- ids
def find_card_numbers(text: str) -> list[str]:
    """16-digit e-money / bank card numbers; unreadable characters become '?'."""
    s = clean(text)
    out = []
    for m in re.finditer(r"[0-9OoDIl|\[\]{}()]{4}[0-9OoDIl|\[\]{}()\s]{8,20}", s):
        tok = re.sub(r"\s", "", m.group(0))
        tok = re.sub(r"[\[\]{}()|]", "?", tok).translate(_CONFUSABLE)
        tok = re.sub(r"[^0-9]", "?", tok)
        digits = sum(c.isdigit() for c in tok)
        if 14 <= len(tok) <= 19 and digits >= len(tok) - 3:
            out.append(tok)
    return out


def station_code(text: str) -> str | None:
    """Pertamina SPBU code in canonical 'NN.NNN.NN' form ('3413703', '34.13702', '31.411.02')."""
    m = re.search(r"(?<!\d)(\d{2})[.,]?(\d{3})[.,]?(\d{2})(?!\d)", clean(text))
    return f"{m.group(1)}.{m.group(2)}.{m.group(3)}" if m else None


def mask_card(card: str | None) -> str | None:
    if not card:
        return card
    return card[:4] + "*" * max(0, len(card) - 8) + card[-4:] if len(card) > 8 else card


def card_similarity(a: str, b: str) -> float:
    if len(a) != len(b):
        return fuzz.ratio(a, b) / 100
    known = [(x, y) for x, y in zip(a, b) if x != "?" and y != "?"]
    if not known:
        return 0.0
    return sum(x == y for x, y in known) / len(known)


_PLATE = re.compile(r"\b([A-Z]{1,2})\s?(\d{1,4})\s?([A-Z]{1,3})\b")


def find_plate(text: str) -> str | None:
    s = clean(text).upper()
    s = re.sub(r"(?<=[A-Z])0(?=\d)", "O", s)
    m = _PLATE.search(s)
    return f"{m.group(1)} {m.group(2)} {m.group(3)}" if m else None


# --------------------------------------------------------------------------- terbilang
_UNITS = {"nol": 0, "satu": 1, "dua": 2, "tiga": 3, "empat": 4, "lima": 5, "enam": 6, "tujuh": 7,
          "delapan": 8, "sembilan": 9, "sepuluh": 10, "sebelas": 11}
_SCALES = {"belas", "puluh", "ratus", "ribu", "juta", "miliar", "milyar", "triliun"}
_SE = {"seratus": ("satu", "ratus"), "seribu": ("satu", "ribu"), "sejuta": ("satu", "juta")}
_VOCAB = list(_UNITS) + list(_SCALES) + list(_SE) + ["rupiah"]


def terbilang_to_int(text: str, min_word_score: int = 72) -> tuple[int | None, float]:
    """'Enam Ratus Ribu Rupiah' -> 600000. Returns (value, fraction_of_words_recognised)."""
    words = re.findall(r"[a-zA-Z]+", clean(text).lower())
    if not words:
        return None, 0.0
    toks, hits = [], 0
    for w in words:
        if len(w) < 3:
            continue
        m = process.extractOne(w, _VOCAB, scorer=fuzz.ratio, score_cutoff=min_word_score)
        if m:
            hits += 1
            if m[0] in _SE:
                toks.extend(_SE[m[0]])
            elif m[0] != "rupiah":
                toks.append(m[0])
    ratio = hits / max(1, len([w for w in words if len(w) >= 3]))
    if not toks:
        return None, ratio
    return _terbilang_value(toks), ratio


def terbilang_vote(readings: list[str]) -> tuple[int | None, str]:
    """Combine several OCR readings of one handwritten terbilang line.

    Each reading is usually wrong in a different word ('Enau Ratus Bibn Rupiad',
    'Enam Rdus Ribn Ruypnal'), so per word position the vocabulary word that fits
    the readings best wins. Returns (value, words); value is None unless every
    position resolves - a number built from half the words is a wrong number."""
    rows = [[w for w in re.findall(r"[a-zA-Z]+", clean(r).lower()) if len(w) >= 3] for r in readings]
    rows = [r for r in rows if r]
    if len(rows) < 2:
        return None, ""
    n = Counter(len(r) for r in rows).most_common(1)[0][0]
    rows = [r for r in rows if len(r) == n]
    if len(rows) < 2:
        return None, ""
    toks = []
    for i in range(n):
        ranked = sorted(((sum(fuzz.ratio(r[i], v) for r in rows) / len(rows), v) for v in _VOCAB), reverse=True)
        (mean, word), (runner_up, _) = ranked[0], ranked[1]
        if max(fuzz.ratio(r[i], word) for r in rows) < 66 or mean < 50 or mean - runner_up < 8:
            return None, ""
        toks.append(word)
    words = " ".join(toks)
    toks = [t for w in toks if w != "rupiah" for t in _SE.get(w, (w,))]
    return (_terbilang_value(toks) if toks else None), words


def _terbilang_value(toks: list[str]) -> int | None:
    total = section = current = 0
    for t in toks:
        if t in _UNITS:
            current += _UNITS[t]
        elif t == "belas":
            current += 10
        elif t == "puluh":
            section += current * 10
            current = 0
        elif t == "ratus":
            section += (current or 1) * 100
            current = 0
        else:
            mult = {"ribu": 10**3, "juta": 10**6, "miliar": 10**9, "milyar": 10**9, "triliun": 10**12}[t]
            total += (section + current or 1) * mult
            section = current = 0
    total += section + current
    return total or None


def to_iso(d: date | None, t: time | None) -> str | None:
    if d is None:
        return None
    return datetime.combine(d, t).isoformat() if t else d.isoformat()
