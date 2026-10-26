"""Cross-receipt checks: things a single receipt cannot prove on its own.

* duplicates (same station + transaction no., or same fingerprint)
* per-station transaction numbers must grow with time
* e-money card balance chain: balance_before - amount == balance_after links
  toll/parking receipts paid with the same card; it verifies amounts and can
  recover a fare hidden under a highlighter or a balance with an unreadable digit
* partial dates (cut-off year) completed from the batch, date outliers, future dates
"""
from __future__ import annotations

import hashlib
import itertools
import statistics
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta

from .normalize import card_similarity, pattern_matches
from .templates.base import FV, PASS, WARN, FAIL, Check, Draft


def _dt(d: Draft, prefix="transaction") -> datetime | None:
    v = d.val(f"{prefix}At")
    if v and "T" in v:
        return datetime.fromisoformat(v)
    return None


def _date(d: Draft) -> date | None:
    v = d.val("transactionDate")
    return date.fromisoformat(v) if v else None


def fingerprint(d: Draft) -> str | None:
    parts = [d.doc_type, d.val("merchantId"), d.val("transactionNo"), d.val("transactionAt"),
             d.val(d.amount_key) if d.amount_key else None, d.val("cardNumber"), d.val("balanceAfter")]
    if d.doc_type == "mobile_topup":
        parts += [d.val("phoneNumber"), d.val("merchantName")]
    if sum(p is not None for p in parts[1:]) < 2:
        return None
    return hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:16]


def run_batch_checks(drafts: list[Draft], today: date, outlier_days: int = 45) -> list[Check]:
    out: list[Check] = []
    _complete_partial_dates(drafts)
    _date_sanity(drafts, today, outlier_days)
    out += _duplicates(drafts)
    out += _station_sequences(drafts)
    out += _card_chains(drafts)
    return out


# --------------------------------------------------------------------------- dates
def _complete_partial_dates(drafts: list[Draft]) -> None:
    known = [_date(d) for d in drafts if _date(d)]
    if not known:
        return
    median = statistics.median_low(known)
    for d in drafts:
        p = d.fields.get("datePartial")
        if not p or "transactionDate" in d.fields:
            continue
        day, month, year = p.value["day"], p.value["month"], p.value["year"]
        if day is None or year is not None:
            continue
        best = None
        for y in (median.year - 1, median.year, median.year + 1):
            try:
                cand = date(y, month, day)
            except ValueError:
                continue
            if abs((cand - median).days) <= 90 and (best is None or abs((cand - median).days) < abs((best - median).days)):
                best = cand
        if best:
            d.fields["transactionDate"] = FV(best.isoformat(), 0.6, "inferred", p.raw, p.lines)
            t = d.val("transactionTime")
            d.fields["transactionAt"] = FV(f"{best.isoformat()}T{t}" if t else best.isoformat(), 0.6, "inferred")
            d.review.append(f"Year of '{p.raw}' inferred as {best.year} from the other receipts")


def _date_sanity(drafts: list[Draft], today: date, outlier_days: int) -> None:
    dates = [x for x in (_date(d) for d in drafts) if x]
    median = statistics.median_low(dates) if dates else None
    for d in drafts:
        x = _date(d)
        if x is None:
            continue
        if x > today + timedelta(days=1):
            d.checks.append(Check("DATE_NOT_IN_FUTURE", FAIL, f"Date {x} is after processing date {today}",
                                  ["transactionDate"]))
        elif (today - x).days > 365:
            d.checks.append(Check("DATE_RECENT", WARN, f"Date {x} is more than a year old", ["transactionDate"]))
        if median and len(dates) >= 3 and abs((x - median).days) > outlier_days:
            d.checks.append(Check("DATE_WITHIN_BATCH", WARN,
                                  f"Date {x} is {abs((x - median).days)} days from the batch median {median}",
                                  ["transactionDate"]))


# --------------------------------------------------------------------------- duplicates
def _duplicates(drafts: list[Draft]) -> list[Check]:
    out = []
    seen: dict[tuple, Draft] = {}
    for d in drafts:
        keys = []
        if d.val("merchantId") and d.val("transactionNo"):
            keys.append(("station-trx", d.val("merchantId"), d.val("transactionNo")))
        fp = fingerprint(d)
        if fp:
            keys.append(("fp", fp))
        for k in keys:
            if k in seen and seen[k] is not d:
                other = seen[k]
                for x, y in ((d, other), (other, d)):
                    x.checks.append(Check("NOT_DUPLICATE", FAIL, f"Same receipt as {y.rid}", ["transactionNo"]))
                out.append(Check("NOT_DUPLICATE", FAIL, f"{d.rid} duplicates {other.rid}", []))
                break
            seen.setdefault(k, d)
    return out


def _station_sequences(drafts: list[Draft]) -> list[Check]:
    out = []
    by_station: dict[str, list[Draft]] = {}
    for d in drafts:
        if d.category == "fuel" and d.val("merchantId") and d.val("transactionNo") and _dt(d):
            by_station.setdefault(d.val("merchantId"), []).append(d)
    for station, ds in by_station.items():
        if len(ds) < 2:
            continue
        ds.sort(key=_dt)
        ok = True
        for a, b in zip(ds, ds[1:]):
            if int(b.val("transactionNo")) <= int(a.val("transactionNo")):
                ok = False
                for x in (a, b):
                    x.checks.append(Check("STATION_SEQUENCE", WARN,
                                          f"Transaction no. at {station} does not increase with time "
                                          f"({a.rid} {a.val('transactionNo')} -> {b.rid} {b.val('transactionNo')})",
                                          ["transactionNo", "transactionAt"]))
        if ok:
            for x in ds:
                x.checks.append(Check("STATION_SEQUENCE", PASS,
                                      f"Transaction no. increases with time across {len(ds)} receipts from {station}",
                                      ["transactionNo", "transactionAt"]))
        out.append(Check("STATION_SEQUENCE", PASS if ok else WARN, f"{station}: {len(ds)} receipts", []))
    return out


# --------------------------------------------------------------------------- card chain
@dataclass
class _Ev:
    d: Draft
    amount_key: str
    lo: datetime
    hi: datetime

    @property
    def amount(self):
        return self.d.val(self.amount_key)

    @property
    def bal(self):
        return self.d.val("balanceAfter")

    @property
    def pat(self):
        f = self.d.fields.get("balanceAfter")
        return f.raw if f is not None and f.value is None and f.raw and "?" in f.raw else None


def _window(d: Draft) -> tuple[datetime, datetime] | None:
    t = _dt(d)
    if t:
        return t, t
    x = _date(d)
    tt = d.val("transactionTime")
    if x:
        if tt:
            t = datetime.combine(x, time.fromisoformat(tt))
            return t, t
        return datetime.combine(x, time.min), datetime.combine(x, time.max)
    p = d.fields.get("datePartial")
    if p and p.value["month"] and p.value["year"]:
        start = date(p.value["year"], p.value["month"], 1)
        end = (start.replace(day=28) + timedelta(days=4)).replace(day=1) - timedelta(days=1)
        return datetime.combine(start, time.min), datetime.combine(end, time.max)
    return None


def _link_score(prev: _Ev, cur: _Ev) -> float:
    pb, pp, a, cb, cp = prev.bal, prev.pat, cur.amount, cur.bal, cur.pat
    if pb is not None and a is not None and cb is not None:
        return 3.0 if pb - a == cb else 0.0
    if pb is not None and a is not None and cp:
        return 3.0 if pattern_matches(cp, pb - a) else 0.0
    if pp and a is not None and cb is not None:
        return 3.0 if pattern_matches(pp, cb + a) else 0.0
    if pb is not None and cb is not None and a is None:
        diff = pb - cb
        return 1.0 if diff > 0 and diff % 500 == 0 else 0.0
    return 0.0


def _card_chains(drafts: list[Draft]) -> list[Check]:
    out = []
    cards = [d for d in drafts if d.val("cardNumber") and d.category in ("toll", "parking")]
    clean_cards = sorted({d.val("cardNumber") for d in cards if "?" not in d.val("cardNumber")})
    for d in cards:  # snap noisy card numbers onto a clean one from the same batch
        c = d.val("cardNumber")
        if "?" in c or c not in clean_cards:
            best = max(clean_cards, key=lambda k: card_similarity(c, k), default=None)
            if best and best != c and card_similarity(c, best) >= 0.85:
                d.fields["cardNumber"] = FV(best, 0.9, "inferred", c, d.fields["cardNumber"].lines)
    groups: dict[str, list[_Ev]] = {}
    for d in cards:
        w = _window(d)
        if w is None:
            continue
        key = "fare" if d.doc_type == "toll" else "fee"
        groups.setdefault(d.val("cardNumber"), []).append(_Ev(d, key, *w))
    for card, evs in groups.items():
        if len(evs) < 2:
            continue
        order = _best_order(evs)
        completed: dict[int, tuple[tuple[int, int], Draft, str]] = {}   # id(FV) -> (link used, draft, review note)
        changed, rounds = True, 0
        while changed and rounds < 5:
            changed, rounds = False, rounds + 1
            for prev, cur in zip(order, order[1:]):
                changed |= _resolve(prev, cur, completed)
        verified, carry = 0, 0
        for prev, cur in zip(order, order[1:]):
            if prev.bal is not None and cur.amount is not None and cur.bal is not None:
                gap, carry_in = cur.bal - (prev.bal - cur.amount), carry
                carry = 0
                if gap == 0:
                    verified += 1
                    # a value completed from one link and confirmed by a different one is as good as
                    # read; a link that produced one of its own three values proves nothing
                    link = (id(prev.d), id(cur.d))
                    used = [prev.d.fields["balanceAfter"], cur.d.fields[cur.amount_key], cur.d.fields["balanceAfter"]]
                    if all(completed.get(id(f), (None,))[0] != link for f in used):
                        for f in used:
                            if id(f) in completed:
                                _, owner, note = completed[id(f)]
                                f.confidence = max(f.confidence, 0.97)
                                if note in owner.review:
                                    owner.review.remove(note)
                    cur.d.checks.append(Check("CARD_BALANCE_CHAIN", PASS,
                                              f"Balance {prev.bal:,} - {cur.amount:,} = {cur.bal:,} "
                                              f"(previous: {prev.d.rid})", [cur.amount_key, "balanceAfter"]))
                    for k in (cur.amount_key, "balanceAfter"):
                        f = cur.d.fields[k]
                        if f.source in ("ocr", "ocr_refined"):
                            f.confidence = max(f.confidence, 0.98)
                elif gap % 500:
                    # top-ups and the fares of receipts missing from the batch are round amounts; an
                    # odd remainder means a misread digit. The same remainder with the opposite sign
                    # on the next link is the previous balance again, which is already flagged.
                    if gap != -carry_in:
                        carry = gap
                        cur.d.checks.append(Check("CARD_BALANCE_CHAIN", WARN,
                                                  f"Balance {cur.bal:,} is {abs(gap):,} off {prev.bal:,} - {cur.amount:,} "
                                                  f"(previous: {prev.d.rid}); a digit is probably misread",
                                                  [cur.amount_key, "balanceAfter"]))
                        if not any(c.code == "CARD_BALANCE_CHAIN" and c.status in (PASS, WARN) for c in prev.d.checks):
                            prev.d.checks.append(Check("CARD_BALANCE_CHAIN", WARN,
                                                       f"Balance {prev.bal:,} does not lead to the next receipt "
                                                       f"({cur.d.rid}); a digit is probably misread", ["balanceAfter"]))
                elif gap > 0:
                    cur.d.checks.append(Check("CARD_BALANCE_CHAIN", "skip",
                                              f"Top-up between {prev.d.rid} and this receipt; link not verifiable",
                                              [cur.amount_key, "balanceAfter"]))
        out.append(Check("CARD_BALANCE_CHAIN", PASS if verified else "skip",
                         f"Card ****{card[-4:]}: {len(order)} receipts, {verified} balance links verified",
                         []))
    return out


def _best_order(evs: list[_Ev]) -> list[_Ev]:
    base = sorted(evs, key=lambda e: (e.lo, e.hi))
    if len(evs) > 7:
        return base
    best, best_s = base, -1.0
    for perm in itertools.permutations(base):
        if any(perm[j].hi < perm[i].lo for i in range(len(perm)) for j in range(i + 1, len(perm))):
            continue  # violates known time order
        s = sum(_link_score(a, b) for a, b in zip(perm, perm[1:]))
        if s > best_s:
            best, best_s = list(perm), s
    return best


def _resolve(prev: _Ev, cur: _Ev, completed: dict) -> bool:
    pb, a, cb = prev.bal, cur.amount, cur.bal
    note = f"from the card balance chain ({prev.d.rid} -> {cur.d.rid})"
    link = (id(prev.d), id(cur.d))
    if pb is not None and a is not None and cb is None:
        v = pb - a
        if v >= 0 and (cur.pat is None or pattern_matches(cur.pat, v)):
            raw = cur.pat
            old_f = cur.d.fields.get("balanceAfter")
            cur.d.fields["balanceAfter"] = FV(v, 0.95 if raw else 0.75, "inferred", raw, old_f.lines if old_f else [])
            if raw:
                cur.d.review.append(f"Balance read as '{raw}' completed to {v:,} {note}")
            completed[id(cur.d.fields["balanceAfter"])] = (link, cur.d, cur.d.review[-1] if raw else "")
            return True
    if prev.pat and a is not None and cb is not None and pb is None:
        v = cb + a
        if pattern_matches(prev.pat, v):
            pat = prev.pat
            prev.d.fields["balanceAfter"] = FV(v, 0.95, "inferred", pat, prev.d.fields["balanceAfter"].lines)
            prev.d.review.append(f"Balance read as '{pat}' completed to {v:,} {note}")
            completed[id(prev.d.fields["balanceAfter"])] = (link, prev.d, prev.d.review[-1])
            return True
    if pb is not None and cb is not None and a is None:
        v = pb - cb
        if 0 < v <= 2_000_000:
            cur.d.fields[cur.amount_key] = FV(v, 0.85 if v % 500 == 0 else 0.6, "inferred",
                                             f"{pb:,} - {cb:,}", [])
            cur.d.review.append(f"Amount {v:,} inferred {note}; assumes no top-up in between")
            completed[id(cur.d.fields[cur.amount_key])] = (link, cur.d, "")   # an amount sits on this link only
            return True
    return False
