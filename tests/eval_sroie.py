"""Dates and shop names on printed receipts the code was not written for.

ICDAR 2019 SROIE (https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE, CC BY 4.0) holds
987 scanned Malaysian shop, restaurant and petrol receipts, one per image, labelled with the
company name and the date as printed. CORD crops both away, so this is where those two fields are
scored. Totals are not: they are in ringgit with cents, and the scanner reads IDR (`9.00` -> 9,000).
It is not part of this repo and needs `pyarrow`:

    hf download jsdnrs/ICDAR2019-SROIE --repo-type dataset --include "data/*" --local-dir sroie
    python tests/eval_sroie.py sroie/data/train-*.parquet -j 4      # add -v to list every miss

`train` (626) is for looking at misses; keep `test` (361) for a final score of the result.

What matters, in this order: `date.wrong` (a date was reported and is not the printed one - keep it
near 0, and `date.silent`, a wrong date reported with confidence >= 0.85, at 0), `date.right`, then
the merchant counts. A merchant is right when it matches the labelled company closely after
dropping punctuation, or contains it / is contained in it.
"""
from __future__ import annotations

import json
import os
import re
import sys
from datetime import date
from multiprocessing import Pool
from pathlib import Path

TODAY = date(2019, 12, 31)          # just after the newest receipt, as when a claim is filed
MONTHS = {m: i + 1 for i, m in enumerate(("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"))}


def gt_date(text: str | None) -> date | None:
    """The labelled date string -> date. Written independently of docscan.normalize on purpose,
    so a parsing bug there cannot hide by marking its own answers. Day-first, as printed in
    Malaysia (and Indonesia); month-first only when the first number cannot be a month."""
    if not text:
        return None
    t = text.strip().upper()
    m = re.search(r"\b(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})\b", t)
    if m:
        y, mo, d = (int(g) for g in m.groups())
        return _mk(y, mo, d)
    m = re.search(r"\b(\d{1,2})[-/. ](\d{1,2})[-/. ](\d{2,4})\b", t)
    if m:
        a, b, y = (int(g) for g in m.groups())
        y += 2000 if y < 100 else 0
        return _mk(y, b, a) if b <= 12 else _mk(y, a, b)
    m = re.search(r"\b(\d{1,2})[-/ ]?([A-Z]{3})[A-Z]*[-/ ,]*(\d{2,4})\b", t)
    if m and m.group(2) in MONTHS:
        y = int(m.group(3))
        return _mk(y + (2000 if y < 100 else 0), MONTHS[m.group(2)], int(m.group(1)))
    m = re.search(r"\b([A-Z]{3})[A-Z]*[ .]+(\d{1,2}),?[ ]+(\d{4})\b", t)
    if m and m.group(1) in MONTHS:
        return _mk(int(m.group(3)), MONTHS[m.group(1)], int(m.group(2)))
    m = re.fullmatch(r"(\d{8})", t)          # '20180304' or '25032018'
    if m:
        s = m.group(1)
        return _mk(int(s[:4]), int(s[4:6]), int(s[6:])) if s[:2] in ("19", "20") else _mk(int(s[4:]), int(s[2:4]), int(s[:2]))
    return None


def _mk(y: int, mo: int, d: int) -> date | None:
    try:
        return date(y, mo, d)
    except ValueError:
        return None


def squash(text: str | None) -> str:
    return re.sub(r"[^A-Z0-9]", "", (text or "").upper())


def merchant_ok(got: str | None, want: str) -> bool:
    from rapidfuzz import fuzz

    g, w = squash(got), squash(want)
    if len(g) < 3:
        return False
    return fuzz.ratio(g, w) >= 85 or (len(g) >= 5 and (g in w or w in g))


_scanner = None


def _init() -> None:
    global _scanner
    sys.path.insert(0, str(Path(__file__).parents[1]))
    from docscan.pipeline import DocScanner

    _scanner = DocScanner()


def _scan(job: tuple[str, int, bytes, dict]) -> dict:
    name, i, data, ent = job
    res = _scanner.scan([(f"{i}.jpg", data)], today=TODAY)
    # one receipt per image: prefer the segment with a date, then the one with most fields
    main_r = max(res.receipts, key=lambda r: (r.transaction_date is not None, len(r.fields)), default=None)
    df = main_r.fields.get("transactionDate") if main_r else None
    return {
        "key": f"{name}[{i}]", "receipts": len(res.receipts), "elapsed_ms": res.elapsed_ms,
        "doc_type": main_r.doc_type if main_r else None,
        "date": str(main_r.transaction_date) if main_r and main_r.transaction_date else None,
        "date_conf": df.confidence if df else None, "date_raw": df.raw if df else None,
        "other_dates": sorted({str(r.transaction_date) for r in res.receipts if r.transaction_date and r is not main_r}),
        "merchant": main_r.merchant_name if main_r else None,
        "gt_date": ent.get("date"), "gt_company": ent.get("company"),
    }


def main() -> None:
    import pyarrow.parquet as pq

    args = sys.argv[1:]
    verbose = "-v" in args
    jobs = int(args[args.index("-j") + 1]) if "-j" in args else 1
    out = Path(args[args.index("-o") + 1]) if "-o" in args else None
    paths = [a for k, a in enumerate(args) if a.endswith(".parquet") and (k == 0 or args[k - 1] not in ("-j", "-o"))]
    work = []
    for path in paths:
        t = pq.read_table(path, columns=["image", "entities"])
        for i, row in enumerate(t.to_pylist()):
            work.append((Path(path).name.split("-")[0], i, row["image"]["bytes"], row["entities"] or {}))
    if jobs > 1:   # each process holds its own models (~1 GB peak); give each a share of the cores
        os.environ.setdefault("DOCSCAN_INTRA_OP_THREADS", str(max(1, (os.cpu_count() or 2) // jobs)))
        with Pool(jobs, initializer=_init) as pool:
            rows = pool.map(_scan, work, chunksize=4)
    else:
        _init()
        rows = [_scan(w) for w in work]

    s = {"images": len(rows), "one_receipt": 0, "seconds_per_image": 0.0,
         "date": dict(labelled=0, right=0, wrong=0, blank=0, silent=0, right_in_other_segment=0),
         "merchant": dict(labelled=0, right=0, wrong=0, blank=0)}
    for r in rows:
        s["one_receipt"] += r["receipts"] == 1
        s["seconds_per_image"] += r["elapsed_ms"] / 1000 / len(rows)
        want = gt_date(r["gt_date"])
        if want is not None:
            d = s["date"]
            d["labelled"] += 1
            key = "right" if r["date"] == str(want) else "blank" if r["date"] is None else "wrong"
            d[key] += 1
            d["silent"] += key == "wrong" and (r["date_conf"] or 0) >= 0.85
            d["right_in_other_segment"] += key != "right" and str(want) in r["other_dates"]
            r["date_result"] = key
            if verbose and key != "right":
                print(f"  date {key:5s} {r['key']}: want {want} ({r['gt_date']!r}), got {r['date']} "
                      f"conf {r['date_conf']} raw {r['date_raw']!r}, {r['receipts']} receipt(s)")
        if r["gt_company"]:
            m = s["merchant"]
            m["labelled"] += 1
            key = "right" if merchant_ok(r["merchant"], r["gt_company"]) else "blank" if not r["merchant"] else "wrong"
            m[key] += 1
            r["merchant_result"] = key
            if verbose and key != "right":
                print(f"  shop {key:5s} {r['key']}: want {r['gt_company']!r}, got {r['merchant']!r}")
    s["seconds_per_image"] = round(s["seconds_per_image"], 2)
    if out:
        out.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n")
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
