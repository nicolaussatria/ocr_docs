"""CLI:  python -m docscan scan page1.jpg page2.pdf -o result.json [--images] [--raw]"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="docscan")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sc = sub.add_parser("scan", help="scan files and print JSON")
    sc.add_argument("files", nargs="+", type=Path)
    sc.add_argument("-o", "--out", type=Path)
    sc.add_argument("--images", action="store_true", help="embed base64 crops")
    sc.add_argument("--raw", action="store_true", help="include raw OCR text per receipt")
    sc.add_argument("--today", type=date.fromisoformat, help="reference date for date checks (YYYY-MM-DD)")
    sc.add_argument("--no-refine", action="store_true", help="disable re-OCR of failing fields")
    sub.add_parser("schema", help="print the JSON Schema of the response")
    args = ap.parse_args(argv)

    if args.cmd == "schema":
        from .schema import ScanResult
        print(json.dumps(ScanResult.model_json_schema(by_alias=True), indent=2))
        return 0

    from dataclasses import replace

    from .config import Settings
    from .pipeline import DocScanner

    st = Settings.from_env()
    if args.no_refine:
        st = replace(st, refine_enabled=False)
    scanner = DocScanner(st)
    res = scanner.scan([(p.name, p.read_bytes()) for p in args.files], include_images=args.images,
                       include_raw_text=args.raw, today=args.today)
    text = res.model_dump_json(indent=2, exclude_none=True)
    if args.out:
        args.out.write_text(text, encoding="utf-8")
        s = res.summary
        print(f"{s.receipt_count} receipts ({s.verified_count} verified, {s.needs_review_count} need review), "
              f"total Rp {s.total_amount:,} in {res.elapsed_ms} ms -> {args.out}", file=sys.stderr)
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
