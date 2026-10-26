"""Shared building blocks for document templates: field values with provenance,
check results, and a small constraint solver that picks the OCR readings which
agree arithmetically (and derives / repairs the ones that do not)."""
from __future__ import annotations

import itertools
import re
from collections import Counter
from dataclasses import dataclass, field, replace
from typing import Any, Callable

import numpy as np

from ..layout import Segment, Unit
from ..normalize import Amount, clean, compact, fix_letters, one_digit_repairs, parse_amounts, station_code, strip_value
from ..ocr import Line

PASS, FAIL, WARN, SKIP = "pass", "fail", "warn", "skip"


@dataclass
class FV:
    """A field value plus where it came from and how much we trust it."""
    value: Any
    confidence: float
    source: str = "ocr"          # ocr | ocr_refined | derived | corrected | inferred | batch | vlm
    raw: str | None = None
    lines: list[Line] = field(default_factory=list)


@dataclass
class Check:
    code: str
    status: str                  # pass | fail | warn | skip
    message: str
    fields: list[str] = field(default_factory=list)


@dataclass
class Draft:
    doc_type: str
    category: str
    segment: Segment
    fields: dict[str, FV] = field(default_factory=dict)
    checks: list[Check] = field(default_factory=list)
    amount_key: str | None = None
    review: list[str] = field(default_factory=list)
    handwritten: bool = False
    rid: str = ""                 # receipt id, assigned by the pipeline
    page_meta: dict = field(default_factory=dict)

    def set(self, key: str, fv: FV | None):
        if fv is not None and fv.value not in (None, ""):
            self.fields[key] = fv

    def val(self, key: str):
        f = self.fields.get(key)
        return f.value if f else None


@dataclass
class Ctx:
    """What a template may ask the pipeline for while parsing."""
    reread: Callable[[list[Line]], list[tuple[str, float]]]
    refine: bool = True
    # read_zone(quad, h=..., stretch=..., rules=..., tighten=..., handwriting=..., date_line=...) -> image readings
    # region the detector did not turn into a usable line (marker, hatching, handwriting)
    read_zone: Callable[..., list[tuple[str, float]]] = lambda *a, **k: []


# --------------------------------------------------------------------------- helpers
def seg_slope(seg: Segment) -> float:
    """Baseline slope of the receipt, from its long lines (short boxes have unreliable angles)."""
    long = [ln.slope for ln in seg.lines if ln.w >= 4 * max(ln.h, 1)]
    return float(np.clip(np.median(long), -0.18, 0.18)) if long else 0.0


def row_zone(line: Line, x0: float, x1: float, slope: float, *, above: float = 0.6, below: float = 0.6) -> np.ndarray:
    """Quad over [x0, x1] on `line`'s (skewed) row; `above` / `below` are in line heights from the row centre."""
    def y(x: float) -> float:
        return line.cy + slope * (x - line.cx)
    up, dn = above * line.h, below * line.h
    return np.array([[x0, y(x0) - up], [x1, y(x1) - up], [x1, y(x1) + dn], [x0, y(x0) + dn]], dtype=np.float32)


def vote(votes: list[tuple[int, float]]) -> tuple[int, float, float] | None:
    """Weighted majority over repeated readings of one value -> (value, best weight, share of all weight)."""
    if not votes:
        return None
    tally: dict[int, list[float]] = {}
    for v, w in votes:
        tally.setdefault(v, []).append(w)
    total = sum(w for _, w in votes)
    value, ws = max(tally.items(), key=lambda kv: sum(kv[1]))
    return value, max(ws), sum(ws) / total if total else 0.0


def settle(cands: list["Cand"], cap: float = 0.84) -> tuple["Cand", float] | None:
    """Choose among repeated readings of one amount -> (winner, its share of the vote).

    A value that only shows up in re-read zones needs two readings behind it (one
    variant alone is usually noise) and stays below the review threshold: legible
    after clean-up is enough to report, not enough to skip a human look. Something
    independent - a balance chain, the terbilang, the item sum - has to lift it."""
    # '3500' beside '35.00' is the same glyphs with the separator missed: it backs that reading up
    by_digits = {re.sub(r"\D", "", c.raw): c.value for c in cands if not c.weak and c.raw}
    cands = [replace(c, value=by_digits.get(re.sub(r"\D", "", c.raw), c.value)) if c.weak else c for c in cands]
    direct = {int(c.value) for c in cands if c.source == "ocr"}
    seen = Counter(int(c.value) for c in cands)
    firm = {int(c.value) for c in cands if not c.weak}
    cands = [c for c in cands if int(c.value) in firm and (int(c.value) in direct or seen[int(c.value)] >= 2)]
    best = vote([(int(c.value), c.conf) for c in cands])
    if best is None:
        return None
    v, top, share = best
    # scattered readings (marker over the digits) are not a usable amount
    conf = top * (1.0 if share >= 0.8 else 0.85 if share >= 0.6 else 0.5)
    lead = max((c for c in cands if int(c.value) == v), key=lambda c: c.conf)
    if v in direct:
        return Cand(v, conf, "ocr", lead.raw, lead.lines), share
    return Cand(v, min(conf, cap), lead.source, lead.raw, lead.lines), share


def unit_conf(u: Unit) -> float:
    vals = u.value_lines or u.lines
    return min(ln.score for ln in vals)


def text_field(u: Unit | None, *, upper: bool = True) -> FV | None:
    if u is None:
        return None
    v = fix_letters(strip_value(u.value_text))
    if not v:
        return None
    return FV(v.upper() if upper else v, unit_conf(u), raw=u.value_text, lines=list(u.lines))


def amount_cands(u: Unit | None, *, min_value: int = 100, source="ocr") -> list["Cand"]:
    if u is None:
        return []
    out = []
    conf = unit_conf(u)
    for a in parse_amounts(u.value_text):
        if a.value is None or a.value < min_value:
            continue
        out.append(Cand(a.value, conf * (0.6 if a.approx else 1.0), source, a.raw, list(u.lines)))
    return out


def first_amount(u: Unit | None, **kw) -> "Cand | None":
    c = amount_cands(u, **kw)
    return c[0] if c else None


def merchant_header(seg: Segment) -> dict[str, FV]:
    """Station / merchant identity from the header block (lines above the first label)."""
    out: dict[str, FV] = {}
    first_label_y = min((u.y0 for u in seg.units if u.label), default=1e9)
    for u in seg.units:
        if u.label or u.y0 > first_label_y + 2:
            continue
        t = clean(u.text)
        c = compact(t)
        code = station_code(t) if len(c) <= 16 and not c.startswith(("TELP", "JL")) else None
        if code and "merchantId" not in out:
            out["merchantId"] = FV(code, u.score, raw=t, lines=list(u.lines))
            if not c.startswith("SPBU"):
                continue
        if c.startswith("SPBU") and "merchantName" not in out:
            name = fix_letters(re.sub(r"(?i)^spbu\s*[:.]?\s*", "SPBU ", t).strip())
            out["merchantName"] = FV(name.upper(), u.score, raw=t, lines=list(u.lines))
        elif c.startswith(("JL", "J1", "JI")) and "merchantAddress" not in out:
            out["merchantAddress"] = FV(t.upper(), u.score, raw=t, lines=list(u.lines))
    return out


# --------------------------------------------------------------------------- solver
@dataclass
class Cand:
    value: float
    conf: float
    source: str = "ocr"
    raw: str = ""
    lines: list[Line] = field(default_factory=list)
    weak: bool = False           # may back up a value another reading established, never establish one


@dataclass
class Rel:
    """A relation between fields, e.g. hargaNonSubsidi - subsidi == hargaJual.

    `err(values)` returns a normalised error: 0 = perfect agreement, <= 1 = within
    tolerance. Exact relations (integer arithmetic) are worth more than approximate
    ones (volume x price, where the printed volume is rounded)."""
    code: str
    fields: tuple[str, ...]
    err: Callable[[dict], float]
    exact: bool
    message: str
    derive: dict[str, Callable[[dict], Any]] = field(default_factory=dict)

    def ok(self, vals: dict) -> bool:
        try:
            return self.err(vals) <= 1.0
        except (ZeroDivisionError, TypeError, KeyError):
            return False


def solve(cands: dict[str, list[Cand]], rels: list[Rel], max_combos: int = 50_000) -> dict[str, Cand | None]:
    keys = [k for k, v in cands.items() if v]
    options = []
    for k in keys:
        opts = sorted(cands[k], key=lambda c: -c.conf)
        # de-duplicate equal values, keep the most confident reading
        seen, uniq = set(), []
        for c in opts:
            if c.value not in seen:
                seen.add(c.value)
                uniq.append(c)
        options.append(uniq + [None])
    total = 1
    for o in options:
        total *= len(o)
    if total > max_combos:  # prune to the two best readings per field
        options = [o[:2] + [None] for o in options]
    best, best_score = {}, -1e18
    for combo in itertools.product(*options):
        assign = dict(zip(keys, combo))
        vals = {k: c.value for k, c in assign.items() if c is not None}
        score = 0.0
        for c in assign.values():
            if c is not None:
                score += 0.6 + 0.4 * c.conf
        for r in rels:
            if all(f in vals for f in r.fields):
                try:
                    e = r.err(vals)
                except (ZeroDivisionError, TypeError):
                    e = 99.0
                if e <= 1.0:
                    score += 3.0 if r.exact else 1.5 - 0.5 * e
                else:
                    score += -4.0 if r.exact else -1.5
        if score > best_score:
            best, best_score = assign, score
    return best


def apply_solution(draft: Draft, cands: dict[str, list[Cand]], assign: dict[str, Cand | None], rels: list[Rel],
                   cast: dict[str, Callable] | None = None) -> None:
    """Write solved values into the draft, derive missing ones, grade confidence, add checks."""
    cast = cast or {}
    vals: dict[str, Any] = {}
    meta: dict[str, FV] = {}
    for k, c in assign.items():
        if c is not None:
            vals[k] = c.value
            meta[k] = FV(cast.get(k, lambda x: x)(c.value), c.conf, c.source, c.raw, c.lines)
    derived_by: dict[str, str] = {}
    changed = True
    while changed:
        changed = False
        for r in rels:
            if not r.exact:
                continue
            missing = [f for f in r.fields if f not in vals]
            if len(missing) != 1 or missing[0] not in r.derive:
                continue
            f = missing[0]
            try:
                v = r.derive[f](vals)
            except (ZeroDivisionError, TypeError):
                continue
            if v is None or v <= 0:
                continue
            partners = [meta[p].confidence for p in r.fields if p != f]
            ocr_raw = [c for c in cands.get(f, [])]
            src, conf, raw = "derived", min(partners) * 0.97, None
            if ocr_raw:
                raw = ocr_raw[0].raw
                if len(str(int(ocr_raw[0].value))) == len(str(int(v))) and \
                        sum(a != b for a, b in zip(str(int(ocr_raw[0].value)), str(int(v)))) == 1:
                    src = "corrected"
            vals[f] = v
            meta[f] = FV(cast.get(f, lambda x: x)(v), conf, src, raw, [])
            derived_by[f] = r.code
            changed = True
    for r in rels:
        if not all(f in vals for f in r.fields):
            miss = [f for f in r.fields if f not in vals]
            draft.checks.append(Check(r.code, SKIP, f"{r.message}: missing {', '.join(miss)}", list(r.fields)))
            continue
        if any(derived_by.get(f) == r.code for f in r.fields):
            f = next(f for f in r.fields if derived_by.get(f) == r.code)
            draft.checks.append(Check(r.code, SKIP, f"{r.message}: used to derive {f}", list(r.fields)))
            continue
        good = r.ok(vals)
        draft.checks.append(Check(r.code, PASS if good else FAIL, r.message, list(r.fields)))
        if good:
            boost = 0.995 if r.exact else 0.95
            for f in r.fields:
                fv = meta[f]
                if fv.source in ("ocr", "ocr_refined") and (r.exact or fv.confidence >= 0.85):
                    fv.confidence = max(fv.confidence, boost)
                elif fv.source in ("derived", "corrected") and derived_by.get(f) != r.code:
                    # computed from one relation AND confirmed by an independent one
                    fv.confidence = max(fv.confidence, 0.97)
    for k, fv in meta.items():
        draft.fields[k] = fv


def refine_candidates(ctx: Ctx, units: dict[str, Unit | None], keys: list[str], *, min_value=100) -> dict[str, list[Cand]]:
    """Re-recognise the lines behind `keys` with image variants (marker removal,
    CLAHE, upscaling) and return the extra amount readings."""
    out: dict[str, list[Cand]] = {}
    if not ctx.refine:
        return out
    for k in keys:
        u = units.get(k)
        if u is None:
            continue
        lines = u.value_lines or u.lines
        cands = []
        for txt, sc in ctx.reread(lines):
            body = txt
            if u.label and not u.value_lines:  # value was inline: drop the label part
                body = re.sub(r"^[^:]*:", "", txt)
            for a in parse_amounts(body):
                if a.value is not None and a.value >= min_value:
                    cands.append(Cand(a.value, sc * (0.6 if a.approx else 0.95), "ocr_refined", a.raw, list(u.lines)))
        if cands:
            out[k] = cands
    return out


def repair_candidates(cands: list[Cand], ok: Callable[[int], bool]) -> list[Cand]:
    """Single-digit substitutions of an OCR reading that satisfy `ok` (exactly one)."""
    out = []
    for c in cands:
        reps = one_digit_repairs(int(c.value), ok)
        if len(reps) == 1:
            out.append(Cand(reps[0], c.conf * 0.7, "corrected", c.raw, c.lines))
    return out


def failed(draft: Draft) -> set[str]:
    return {f for c in draft.checks if c.status == FAIL for f in c.fields}


def amounts_in(text: str, min_value: int = 1000) -> list[Amount]:
    return [a for a in parse_amounts(text) if a.value is not None and a.value >= min_value]
