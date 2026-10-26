"""Accuracy report against hand-verified ground truth.

    python tests/eval_golden.py            # runs the scanner on tests/golden/*.jpg
    python tests/eval_golden.py out.json   # evaluates an existing response

Metrics that matter for data compilation:
  * silent errors  – a wrong value inside a receipt marked `verified` (must be 0)
  * flagged errors – a wrong value inside a `needs_review` receipt (a human will see it)
  * missing        – expected field not extracted
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path

HERE = Path(__file__).parent / "golden"


def same(a, b) -> bool:
    if a is None or b is None:
        return False
    if isinstance(b, (int, float)) and not isinstance(b, bool):
        try:
            return abs(float(a) - float(b)) < 1e-6
        except (TypeError, ValueError):
            return False
    # text compared ignoring spaces/case: OCR often drops a space ('PLAZASENAYAN'), which is
    # cosmetic for compilation; anything else (a wrong letter/digit) counts as an error
    return "".join(str(a).upper().split()) == "".join(str(b).upper().split())


def evaluate(result: dict, expected: list[dict]) -> dict:
    outs = list(result["receipts"])
    ids = ("transactionNo", "transactionAt", "transactionDate", "amountPaid", "totalPrinted", "fee", "fare",
           "balanceAfter", "amount", "receivedFrom", "gate", "merchantId")

    def score(exp, r):
        if r["docType"] != exp["docType"] or r["source"]["file"] != exp["file"]:
            return -1
        hits = {k for k, v in exp["fields"].items() if same(r["fields"].get(k, {}).get("value"), v)}
        return len(hits) if hits & set(ids) else -1   # must share an identifying value

    # global greedy on the best (expected, output) pairs – avoids order-dependent mismatches
    pairs = sorted(((score(e, r), i, r["id"]) for i, e in enumerate(expected) for r in outs), reverse=True)
    taken_e, used, match = set(), set(), {}
    for sc, i, rid in pairs:
        if sc < 1 or i in taken_e or rid in used:
            continue
        taken_e.add(i); used.add(rid); match[i] = next(r for r in outs if r["id"] == rid)
    rows = [(e, match.get(i)) for i, e in enumerate(expected)]
    stats = {"expected_receipts": len(expected), "matched_receipts": 0, "fields": 0, "correct": 0, "missing": 0,
             "flagged_errors": 0, "silent_errors": 0, "verified_receipts": 0, "verified_all_correct": 0,
             "details": []}
    for exp, got in rows:
        if got is None:
            stats["details"].append({"expected": exp["docType"], "result": "NOT FOUND"})
            stats["fields"] += len(exp["fields"])
            stats["missing"] += len(exp["fields"])
            continue
        stats["matched_receipts"] += 1
        verified = got["status"] == "verified"
        stats["verified_receipts"] += verified
        all_ok = True
        for k, v in exp["fields"].items():
            stats["fields"] += 1
            f = got["fields"].get(k)
            if f is None:
                stats["missing"] += 1
                all_ok = False
                stats["details"].append({"id": got["id"], "field": k, "expected": v, "got": None})
            elif same(f["value"], v):
                stats["correct"] += 1
            else:
                all_ok = False
                key = "silent_errors" if verified else "flagged_errors"
                stats[key] += 1
                stats["details"].append({"id": got["id"], "field": k, "expected": v, "got": f["value"],
                                         "source": f["source"], "status": got["status"], "kind": key})
        stats["verified_all_correct"] += verified and all_ok
    stats["extra_receipts"] = len(outs) - len(used)
    stats["matched_ids"] = sorted(used)
    stats["field_accuracy"] = round(stats["correct"] / max(1, stats["fields"]), 3)
    return stats


def main():
    expected = json.loads((HERE / "expected.json").read_text())
    if len(sys.argv) > 1:
        result = json.loads(Path(sys.argv[1]).read_text())
    else:
        sys.path.insert(0, str(Path(__file__).parents[1]))
        from docscan.pipeline import DocScanner
        files = sorted({e["file"] for e in expected})
        res = DocScanner().scan([(f, (HERE / f).read_bytes()) for f in files], today=date(2026, 10, 12))
        result = json.loads(res.model_dump_json())
        print(f"elapsed {result['elapsedMs']} ms")
    s = evaluate(result, expected)
    for d in s.pop("details"):
        print("  ", d)
    print(json.dumps(s, indent=1))


if __name__ == "__main__":
    main()
