"""Accuracy of the general `receipt` template on receipts it was not written for.

CORD v2 (https://huggingface.co/datasets/naver-clova-ix/cord-v2, CC BY 4.0) holds photos of
Indonesian shop and restaurant receipts, one per image, with the printed total as ground
truth. It is not part of this repo and needs `pyarrow` to read:

    pip install pyarrow huggingface_hub
    hf download naver-clova-ix/cord-v2 --repo-type dataset --include "data/test-*" --local-dir cord
    python tests/eval_cord.py cord/data/test-*.parquet            # add -v to list every miss

What matters, in this order: `wrong` (an amount was reported and is not the receipt's total -
keep it at 0), `right`, and `blank` (left for a person; `suggested` of them carry the right
best guess). CORD crops out dates and shop names, so those two fields cannot be scored here.
"""
from __future__ import annotations

import json
import re
import sys
from datetime import date
from pathlib import Path


def gt_amount(value) -> int | None:
    """'45,500' / '23.000' / 'Rp 45.500,00' / '70000.00' -> rupiah; several totals -> the last (what was paid)."""
    if isinstance(value, list):
        value = value[-1] if value else None
    if value is None or isinstance(value, dict):
        return None
    t = re.sub(r"(?i)rp\.?|idr|\s", "", str(value)).strip(".,-:")
    if re.search(r"\d[.,]\d{3}[.,]\d{1,2}$", t) or re.fullmatch(r"\d{4,}[.,]\d{2}", t):     # cents
        t = re.sub(r"[.,]\d{1,2}$", "", t)
    digits = re.sub(r"\D", "", t)
    return int(digits) if digits else None


def main() -> None:
    import pyarrow.parquet as pq

    sys.path.insert(0, str(Path(__file__).parents[1]))
    from docscan.pipeline import DocScanner

    verbose = "-v" in sys.argv
    scanner = DocScanner()
    stats = dict(images=0, with_total=0, one_receipt=0, right=0, wrong=0, blank=0, suggested=0)
    for path in [a for a in sys.argv[1:] if not a.startswith("-")]:
        for i, row in enumerate(pq.read_table(path).to_pylist()):
            total = json.loads(row["ground_truth"])["gt_parse"].get("total")
            total = total[0] if isinstance(total, list) and total else total
            want = gt_amount(total.get("total_price")) if isinstance(total, dict) else None
            res = scanner.scan([(f"{i}.png", row["image"]["bytes"])], today=date(2026, 10, 12))
            # the image holds one receipt: take the one with an amount, else the one with most fields
            main_r = max(res.receipts, key=lambda r: (r.amount is not None, len(r.fields)), default=None)
            got = main_r.amount if main_r else None
            stats["images"] += 1
            stats["one_receipt"] += len(res.receipts) == 1
            if want is None:
                continue
            stats["with_total"] += 1
            key = "right" if got == want else "blank" if got is None else "wrong"
            stats[key] += 1
            stats["suggested"] += key == "blank" and main_r is not None and main_r.suggested_amount == want
            if verbose and key != "right":
                print(f"  {Path(path).name}[{i}] {key}: want {want:,}, got {got}, "
                      f"suggested {main_r.suggested_amount if main_r else None}, {len(res.receipts)} receipt(s)")
    print(json.dumps(stats, indent=1))


if __name__ == "__main__":
    main()
