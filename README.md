---
title: DocScan
emoji: 🧾
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
short_description: Receipt scans to validated JSON and a claim table
---

# DocScan – receipt scan → validated JSON

Python service that takes photos / scans / PDFs of Indonesian expense receipts (one or **many receipts glued on one page**), finds each receipt, extracts typed fields, **cross-checks them arithmetically**, and returns JSON that a web app, a Flutter app or an ASP.NET backend can consume — plus a small review UI for compiling the data.

Supported layouts today: Pertamina SPBU subsidised (Pertalite "Informasi Harga BBM"), Pertamina retail (Pertamax "Harga/Liter … Total Harga"), SPBU *bon kontan*, Jasa Marga e-Toll slips, parking tickets (Flazz/e-money), *kwitansi*, **m-Commerce mobile top-ups (pulsa)** - and, without a template, **any other receipt or invoice that prints its money in labelled rows** (see below).

## Design principle: wrong data is worse than missing data

OCR alone is not "high accuracy". This service treats every value as a claim that must be supported:

| Layer | What it does | Why it matters |
|---|---|---|
| Orientation | Detects 0/90/180/270° per page | Your scans are rotated 270° |
| Segmentation | Splits a page into receipts (labels → values on skew-corrected rows, header anchors, repeated-field splits) | 9 receipts per A4 page, packed edge to edge |
| Reused backing paper | Detects separate receipt-sheet boundaries with transaction evidence, masks exposed backing text, then runs OCR again before pairing fields | Stops an invoice total on the backing sheet becoming another expense or attaching to a receipt |
| Template parsing | Fuzzy, OCR-noise-tolerant labels (`Dibavar Konsumen`, `No. Irans`, `Pemer intah`) | OCR noise in labels is common |
| **Constraint solver** | e.g. `Harga Non Subsidi − Subsidi = Harga Jual`, `Tanpa Subsidi − Subsidi = Dibayar`, `Volume × Harga ≈ Total` | A misread digit breaks the arithmetic → dropped, re-derived, or flagged |
| Constraint-guided re-OCR | Only failing fields are re-read with marker removal / CLAHE / upscaling | Accuracy where needed, no cost elsewhere |
| **Amount-zone re-read** | Where the detector returns nothing usable for an amount, the zone is cut from the image by layout (right of `GOL-1 e-Toll <issuer>`, of `Jumlah Rp.`, of the `Rp.` column of a bon kontan), cleaned (marker shading flattened, hatching removed, double-height toll font widened) and re-recognised in several variants that vote | Toll fares under a highlighter, the hatched amount box of a kwitansi, handwriting on a dotted leader. A value that only appears this way needs two agreeing readings and stays `needs_review` until an independent check confirms it |
| One-digit repair | `349.000` → `319,000` when volume × price proves it (marked `corrected`) | Struck-through printed totals |
| Batch checks | Duplicates, station transaction-number sequence, **e-money balance chain** (`saldo₁ − tarif₂ = saldo₂`), partial-date completion, date outliers | Confirms a toll fare read from under a highlighter, completes a balance with an unreadable digit, and flags a link that is off by an odd remainder (misread digit) |
| Handwriting detection | Handwritten amount over a printed one → `claimedAmount` + review flag | Both sample Pertamax receipts had this |
| Status | `verified` only if key fields exist, are read directly or cross-verified, and no check fails | Everything else is `needs_review` with reasons |

Every field carries `confidence`, `source` (`ocr`, `ocr_refined`, `derived`, `corrected`, `inferred`, `vlm`), the raw OCR text and a bbox in the original image.

## Any other receipt or invoice

A receipt that matches none of the layouts above is not a dead end. `docType: "receipt"` (`templates/receipt.py`) has no template: it finds the money rows by what they are called - total, subtotal, pajak/PB1/PPN, service, diskon, pembulatan, tunai, kembali, kartu/debit/QRIS ..., Indonesian or English, tolerant of OCR noise (`T01AL`, `GR0 TTAL`) - and by how the amounts add up.

| Rule | Why |
|---|---|
| The printed total is the answer by default | It is the one number the receipt states outright |
| Sums confirm it: `subtotal + tax + service - discount = total`, `cash - change = total`, card payment = total, items add up, total = subtotal + 10/11/12 % | A confirmed total gets confidence 0.99; the rows a closed sum used (`subtotal`, `tax`, `serviceCharge`, `discount`, `rounding`, `cashPaid`, `change`, `nonCashPaid`, `items`) are reported, rows nothing confirms are left out |
| A total printed large, or one that fails the sums, is re-read from the image | Big fonts come back mirrored (`000.800` for `308,000`) or in pieces (`20` + `000`) |
| A total nobody could read is derived only when two independent sums agree | One sum is wrong whenever a row in it is unreadable or belongs elsewhere |
| One sum only, or print and arithmetic disagree with nothing to break the tie | `amount` stays `null`, the best guess goes to `suggestedAmount`, the receipt needs review |
| A printed total that nothing else on the receipt confirms | Reported, but capped at confidence 0.84 with a review note: nearly every wrong total seen in testing was one of these |
| A cashier's rounding (up to Rp99) is accepted only when one of the two figures is round | `107,999` against a sum of `107,998` is a misread digit, not rounding |

Also extracted when present: `transactionDate`/`Time`, `merchantName` (only when a header - address, phone, date - is visible; on a photo cropped to the items the first line is a product), `receiptNo`, `paymentMethod`. Layout support for these receipts: a detached amount (or quantity) column is rejoined with its names, a totals block is rejoined with its items, rows are paired under the photo's skew, and amounts printed on the line below their name are picked up.

**Invoices** are the same document type with a header on top. What that adds:

| Layer | What it does | Example |
|---|---|---|
| One document per page | The layout pass returns letterhead, header table, lines and stray column headings as separate pieces. When exactly one piece carries the money and another states no amount of its own, they are joined (`templates.assemble`). Several receipts on a page are never joined: each has amounts the others lack | A full-page invoice comes back as one receipt instead of one receipt and two `unknown` fragments |
| Header fields by name | `receiptNo`, `transactionDate`, `dueDate`, `customerName`, `customerId`, `npwp` are found by their caption - Indonesian, English or both (`Tanggal Cetak Tagihan/ Statement Date`) - with the value after the colon, to the right, or underneath in a table | The statement date becomes the document date; a due date or a billing period never does |
| Total printed twice | The total in the header table and the total under the lines are two independent prints: when they agree, that is a check (`PRINTED_TWICE_EQ_TOTAL`) | `Total Amount 3.330.000` twice |
| Tax base | `Subject to Tax` / DPP is not added into the sum; the tax must be 10/11/12 % of it (`TAX_EQ_RATE_OF_BASE`), which confirms `tax` and `taxBase` | 11 % of 3,000,000 = 330,000 |
| Vendor name | A logo is the worst-read line on the page; when the web or mail address spells the same name, the address is used | logo read as `HYPERINEI`, `www.hypernet.co.id` -> `HYPERNET` |

**What it is, in words.** Every receipt comes back with a `description` - one Indonesian line saying what was bought and where (`BBM Pertalite 3,2 L - SPBU Ciracas`, `Parkir Plaza Senayan`, `Pulsa Telkomsel 0813...`, `Internet Hypernet: FION INTERNET DEDICATED ...`) - built in `docscan/describe.py` from fields the receipt gave with some confidence. For a receipt without a template the words on it also set its `category` (`internet`, `meal`, `lodging`, `transport`, `vehicle_service`, `medical`, `office`, `utilities`, `pulsa`, else `other`). Both are a starting point for a person, not checked values: a restaurant's "service charge" is deliberately not a car service, and a description stays generic (`Struk belanja`) when nothing readable says more.

Further fields from invoices: `otherCharges` (stamp duty, delivery, admin fee - when not zero) and item descriptions that wrap onto a second line.

Scope: IDR amounts (no cents), documents whose money is printed in rows. Like any other receipt it is `verified` only with a trusted total **and** a date.

## Mobile top-ups (pulsa)

Printed BCA m-Commerce / Telkomsel receipts return `docType: "mobile_topup"` and
`category: "pulsa"`. Repeated `m-Commerce:` headers split multiple transactions
on one sheet; a separated bank tax footer stays attached to its transaction.
The parser extracts `nominal`, `adminFee`, `amountPaid`, `phoneNumber`,
`transactionNo` (the printed `SN`), `transactionStatus`, product, merchant,
purpose, and transaction date/time. Phone and serial numbers remain strings
so leading zeros are preserved. They are not treated as bank-card numbers.

`amount` is the **paid total including the admin fee**. The check
`NOMINAL_PLUS_ADMIN_EQ_PAID` requires nominal + admin fee = paid total.
Printed amounts accept both `200,000.00` and `200.000,00`; malformed amounts
and nonzero fractional rupiah require review. Missing or weak amounts may be
re-read with image variants. A missing total can be derived from the two
components, but stays `needs_review`. Missing phone/status/amount evidence and
arithmetic mismatches also require review. Explicitly failed or pending
transactions retain the printed values in `fields`, but have a null `amount`
and contribute nothing to totals or automatic claim amounts.

The review UI puts this category into the **Pulsa** column even when raw OCR
text is omitted. The example photo contains Rp200,000 + Rp2,000 and
Rp75,000 + Rp2,000: **Rp279,000 paid in total**. Its manually transcribed data
is in `samples/mobile-topups.transcription.json`; it is not an OCR benchmark.

Tests cover transcribed OCR boxes and a generated page with uneven lighting
and mild perspective distortion through the real OCR engine. These synthetic
tests do not measure accuracy on the original camera photo, whose image file
was not available in the workspace. Other operators sharing the same printed
layout are recognised by name; their real receipts have not been evaluated.

## Measured results (read the caveat)

On the two scanned pages provided (17 receipts, 167 hand-verified fields), CPU only:

| Variant | Receipts found | Field accuracy | Wrong values in `verified` receipts |
|---|---|---|---|
| As scanned | 17 / 17 | 97.6 % (163/167) | **0** |
| Upside down | 17 / 17 | 97.6 % (163/167) | **0** |
| Rotated +3° / −5° | 15 / 17 | 89.8 % / 90.4 % ¹ | **0** |
| Downscaled to 60 % | 15 / 17 | 87.3 % ¹ | **0** |
| Blur + JPEG q40 | 16 / 17 | 89.8 % ¹ | **0** |
| Perspective warp | 15 / 17 | 89.2 % ¹ | **0** |

¹ Measured before the amount-zone re-read was added (166 fields) and not repeated with the same distortions. With a comparable set of distortions the current code scored 2–5 fields higher than the previous one on every variant and added no wrong value to a `verified` receipt.

All five amounts that sit under a grey marker or are handwritten (three toll fares, the kwitansi, the bon kontan) are now reported. The Halim fare is `verified` because the card balance chain confirms it; the other four stay `needs_review`. What is still missed is what a human also struggles with: handwritten dates, text hidden under an overlapping receipt, and a balance with a damaged digit that no chain link pins down. Text fields are compared ignoring spaces (OCR sometimes returns `PLAZASENAYAN`).

**Caveat [High confidence]:** these numbers are in-sample — the parsers were tuned on these same two pages, so they overstate accuracy on new layouts. Before relying on the `verified` flag in production, label 50–100 receipts that were not used for tuning and run `tests/eval_golden.py` on them. The metric to watch is `silent_errors`; it must stay at 0.

Speed (2 vCPU): single receipt photo ≈ 2.7 s, page with 8–9 receipts ≈ 7–10 s. Peak memory ≈ 1 GB per worker for
one scan (measured 2026-10-21 with 2 OCR threads, compressed memory included); see *Many people at once*.

### Receipts the code was not written for

Measured on [CORD v2](https://huggingface.co/datasets/naver-clova-ix/cord-v2) (CC BY 4.0): photos of Indonesian shop and restaurant receipts, one per image, with the total as ground truth. "Before" is the same code without the general template and its layout rules.

| Set | Images with a total | Total right, before -> after | Total wrong, before -> after | Left blank for review, after |
|---|---|---|---|---|
| **200 receipts never looked at** (last shard of the train split, scored once) | 193 | 116 -> **171** (89 %) | 16 -> **2** | 20 (8 with the right `suggestedAmount`) |
| The other 800 images, all used at some point while developing (validation, test and three train shards) | 779 | 482 -> 714 (92 %) | 41 -> 2 ² | 63 |
| Six receipts glued on one page, 8 pages | 47 | 34 -> 41 | 1 -> 0 | 44 of 48 receipts found |

² One of the two is a typo in the dataset's own answer (`57,0000`); the scanner returns the printed 57,000.

How to read this **[High confidence]**: only the first row is an unbiased estimate - roughly nine totals in ten right, one in ten left for a person, one in a hundred wrong. The second row is optimistic: rules were adjusted after looking at failures in those sets, and each time a new unseen shard turned up errors the previous numbers had not predicted (one such shard showed 6 wrong totals before the rules behind them were fixed). The two wrong totals of the first row were deliberately not examined. A total that no other figure on the receipt confirms is reported with confidence 0.84 and a review note, never as `verified`. CORD images are cropped to the items and totals, so dates and shop names could not be scored at all - those two fields are untested on real receipts. Reproduce with `python tests/eval_cord.py <split.parquet>`.

### Dates and shop names

Measured on [ICDAR 2019 SROIE](https://huggingface.co/datasets/jsdnrs/ICDAR2019-SROIE) (CC BY 4.0): 987 scanned
Malaysian shop, restaurant and petrol receipts, one per image, labelled with the printed date and the company name.
Totals are not scored - they are ringgit with cents, and the scanner reads rupiah (`9.00` -> 9,000). "Before" is the
code before the date fixes of 2026-10-21 (a year glued to the time, `11/02/1819:21`, read as 1819; order and register
numbers using up the digits of the date beside them; `TU21- 6/72` read as the year 2072; `02/JAN/2017`, `28 Mar 18`,
`OCT 3, 2016` and `2018-01-0522:55:25` not recognised).

| Set | Date right, before -> after | Date wrong | ... of them with confidence >= 0.85 | Left blank |
|---|---|---|---|---|
| **361 receipts never looked at** (`test`, scored once before and once after) | 324 -> **344** (95 %) | 14 -> **7** | 8 -> **5** | 23 -> 10 |
| 626 receipts whose misses were studied (`train`) | 574 -> 608 (97 %) | 22 -> 3 | 8 -> 2 | 30 -> 15 |

On `test`, 20 receipts improved and none went from right to wrong; one that was blank got a wrong date at confidence
0.9 from the new `dd MON yy` reading - after that score, a 2-digit year with a month name was set to 0.84 (review),
as every other 2-digit year already was. The remaining wrong dates are OCR misreads (`2018` read as `2013`) and a
receipt printing two dates. Shop names are weak on this set: 1 receipt in 5 gets the labelled company, because these
headers put a registration number or a numbered address straight under the name, and a third of the images come back
as several pieces (with ringgit amounts no piece carries a total, so the pieces are not joined). Reproduce with
`python tests/eval_sroie.py <split.parquet> -j 4`.

### Bad photos

The same receipts with photo damage applied - dark, shadow, faded, blur, motion blur, low resolution, heavy JPEG,
noise, tilt, perspective, yellow light, and a mix - before and after the contrast stretch and the second pass
(`SECOND_PASS`):

| Set | Before | After |
|---|---|---|
| 40 CORD phone photos x 9 kinds of damage: totals right | 312 / 333 | 323 / 333 |
| The two golden pages (17 receipts) x 13 kinds of damage: amounts right | 175 / 221 | 181 / 221 |

Wrong amounts did not increase (CORD 2 and 2, golden 10 and 10) and none is in a `verified` receipt, before or after.
The gains are dark and faded photos (a too-dark photo used to come back as "no receipt detected"), shadows and the
mixed phone-photo damage. Strong motion blur and very low resolution on a dense multi-receipt page are not helped:
the text is found, but the letters are no longer in the picture - retake it. Handwriting is not helped either (see
*Known limitations*). Cost: a page that read poorly gets one more OCR pass (the 8-receipt sample page 5.4 -> 7.8 s);
a page that reads well costs nothing extra.

`python tests/eval_bad_photos.py` repeats this with a fixed, harsher damage set (stronger blur and motion blur) on
the two golden pages: 2001 of 2171 values right, 207 of 221 amounts. It also shows what the table above does not: 13
wrong values inside `verified` receipts, 9 of them under heavy motion blur, where one misread digit repeats through
figures that still add up (`15892 - 5892` read as `15092 - 5092`). One is a claim date (28 read as 20); none is an
amount since a value read off a cleaned-up crop needs a passing check to be trusted (it was a 240,000 for 319,000).

## Run

```bash
pip install -r requirements.txt                       # Python 3.10+; models ship inside the rapidocr wheel
python -m docscan scan tests/golden/page1.jpg tests/golden/page2.jpg -o result.json
uvicorn docscan.api:app --host 0.0.0.0 --port 8000     # then open http://localhost:8000/ for the review UI

docker build -t docscan . && docker run -p 8000:8000 -e DOCSCAN_API_KEY=change-me docscan
```

Tests: `pytest -q` (`-m "not slow"` skips the ones that load the OCR models).
Accuracy report: `python tests/eval_golden.py`; for receipts without a template, `python tests/eval_cord.py`; dates and
shop names, `python tests/eval_sroie.py`; bad photos, `python tests/eval_bad_photos.py` (see their docstrings).

## Many people at once

The OCR engine reads one image at a time per worker, so people scanning together share it: the total time grows with
the number of pages, whatever the setup. The review page sends one document at a time per person. A worker starts at
most `MAX_CONCURRENT_SCANS` (2) scans; the rest wait as uploads, first come first served, instead of every scan
decoding its pages and all of them finishing at the end. Measured 2026-10-21 (laptop CPU, 2 OCR threads per worker,
memory including what macOS compressed - a Linux server holds it all in RAM):

| 10 people upload at the same moment | Workers | First result | Median | Last | Peak memory |
|---|---|---|---|---|---|
| a phone photo or scanned page each | 1, before the queue | 18 s | 58 s | 66 s | 1.9 GB |
| a phone photo or scanned page each | 1 | 9 s | 45 s | 68 s | 1.5 GB |
| a phone photo or scanned page each | 2 | 12 s | 37 s | 59 s | 2.0 GB |
| a 4-page PDF each (34 receipts per PDF) | 1, before the queue | 440 s | 440 s | 441 s | 2.5 GB |
| a 4-page PDF each | 1 | 83 s | 276 s | 464 s | 1.7 GB |
| a 4-page PDF each | 2 | 101 s | 215 s | 373 s | 3.0 GB |

Sizing: 2 vCPU / 4 GB RAM with 1 worker serves a team of about 10 that scans a few receipts each, with waits of a
minute or so when everyone scans at once. More cores cut the waits: one worker per 2 cores, `DOCSCAN_INTRA_OP_THREADS`
= cores ÷ workers, and about 1.5 GB of RAM per worker. Two workers gave 1.2-1.25x here, not 2x, because uvicorn hands
connections to workers without regard to how busy each is. Behind a reverse proxy, allow long requests
(`proxy_read_timeout 600s` in nginx): a queued scan answers only when its turn is done.

## Review workspace

Open `/` for the responsive receipt workspace:

The full-width upload area keeps the workspace compact, with summaries and receipts directly below it. **Review next** opens the first receipt that still needs attention.

1. **Add documents** — choose JPG, PNG, WebP, or PDF files, or drop them on the page. The queue shows which files are ready, scanning, finished, or need a retry.
2. **Scan receipts** — select **Scan documents**. Each document is processed separately; more files can be added while scanning.
3. **Review & correct** — filter by **To review**, search merchants/dates/notes, and edit the table directly. **View receipt** opens the original image beside editable fields, with image zoom and previous/next navigation. **Scan checked** is the scanner's result; **Reviewed by you** means a person checked it. Editing a reviewed receipt or uploading a new duplicate resets its review flag. Original extracted fields and checks remain available in an expandable section.
4. **Export your claim** — **Export Excel** writes the claim forms as one `.xlsx` workbook - *Claim Tunj Operasional* (one table per claim, stacked) and, for medical receipts, *Claim Medical*, then *Attachment*: the picture of every receipt under its form and row number (a row's No links to its picture, the picture's heading back to the row; one receipt is never split across two printed pages); **CSV** gives the same table as plain text for systems that import CSV. Each document gets a separate claim unless **Combine into one claim** is chosen. **JSON data** preserves the original scan and adds corrections under `userEdits`. Exports include unreviewed receipts; the workspace displays the outstanding review count.

The claim total includes corrections, handwritten claims, and suggested amounts, as before. The **To review** count includes uncertain scans, duplicates, and unconfirmed edits. API keys are optional and available under **Connection settings** in the top bar. Files and edits belong to the current tab; export before reloading or leaving. The build version is in the footer.

The UI remains a self-contained HTML file served by FastAPI, with no frontend build step or external asset requests. Browser regression checks use mocked API responses and cover editing, validation, filters, retries, downloads, and desktop/mobile layouts:

```bash
# With Playwright available to Node (a development-only dependency):
node tests/test_ui.cjs
# Optional: save desktop/mobile screenshots.
DOCSCAN_UI_SCREENSHOTS=/tmp/docscan-previews node tests/test_ui.cjs
# With the local server running, exercise the real OCR upload and review date:
DOCSCAN_TEST_DOCUMENT=/path/to/original.jpg node tests/test_ui_live.cjs
# Omit DOCSCAN_TEST_DOCUMENT to use tests/golden/page2.jpg.
```

The live test expects the sample kwitansi's signing date, `2026-10-02`, in both the API and the review form. It uses a fresh browser context and no mocked scan response. The high-resolution date crops in `tests/golden/handwriting/` cover the original upload's signature-crossing error (`13/10` instead of `2/10`); the smaller full-page copy is also tested. Large signing-date crops are normalized before ink cleanup, without local shading amplification. Existing browser results must be scanned again after a parser update; they are not recomputed when the server reloads.

## API

`POST /v1/export/xlsx` — JSON `{"tables": [{"name", "rows": [{"date", "liters", "bbm", "tol", "parkir", "pulsa", "servis", "lain", "note", "image"}]}], "medical": [{"date", "amount", "service", "patient", "note", "image"}]}` (what the review page sends, corrections included) -> one `.xlsx` workbook: *Claim Tunj Operasional* and, when there are medical rows, *Claim Medical* (Nilai Claim = 90 % of `amount`); when rows carry `image` (the base64 JPEG/PNG `image` of the scan response), an *Attachment* sheet with those pictures. Pictures are limited to 3 MB each and 150 MB per export; behind nginx, `client_max_body_size` must allow the request (a page of 8 receipts is ~0.5 MB). Same `X-API-Key` rule as scanning.

`POST /v1/scan` — multipart field `files` (repeatable). Query: `includeImages` (base64 crop per receipt), `includeRawText`, `today` (reference date for date checks). Header `X-API-Key` when `DOCSCAN_API_KEY` is set. Errors: 401 key, 413 too large, 422 unreadable file.

```jsonc
{
  "schemaVersion": "1.0",
  "summary": { "receiptCount": 17, "verifiedCount": 11, "needsReviewCount": 6, "totalAmount": 1660450, "verifiedAmount": 319950, ... },
  "receipts": [{
    "id": "r5", "docType": "fuel_subsidized", "category": "fuel",
    "status": "verified", "confidence": 0.98,
    "amount": 28000, "claimedAmount": null, "transactionAt": "2026-09-23T18:32:25",
    "merchantName": "SPBU KRAMAT JATI NO.109", "merchantId": "34.135.05",
    "fingerprint": "…",                       // store it; reject the same receipt uploaded twice
    "bbox": [x0, y0, x1, y1],                  // original image pixels
    "fields": { "amountPaid": { "value": 28000, "confidence": 0.995, "source": "ocr", "raw": "28,000", "bbox": [...] }, ... },
    "checks": [{ "code": "TOTAL_MINUS_SUBSIDY_EQ_PAID", "status": "pass", ... }],
    "reviewReasons": []
  }],
  "batchChecks": [...], "pages": [...], "warnings": []
}
```

Full contract: `docs/scan-result.schema.json` (also `GET /v1/schema`). Sample: `samples/result.sample.json`.

Keys are camelCase on purpose: they bind directly to Dart field names and to ASP.NET `System.Text.Json` web defaults, avoiding the camelCase/PascalCase null-binding problem.

Amount semantics: `amount` = evidence-backed transaction amount (printed / arithmetic-verified). `claimedAmount` = handwritten override. When the amount is not reliably readable, `amount` is `null` so it never leaks into totals; the best guess comes back as `suggestedAmount` (and stays in `fields`) so a client can pre-fill it for a person to confirm.

## Integrating

**Flutter** — `clients/flutter/scan_result.dart` (model, no codegen) and `docscan_client.dart` (Dio multipart). Autofill from `amount` / `transactionDate`; show `reviewReasons` and never auto-submit `needs_review` receipts. (Not compiled here — no Dart SDK in the build environment.)

**ASP.NET** — recommended topology: Flutter → ASP.NET (auth, storage, duplicate check on `fingerprint`, audit of user edits) → DocScan on a private port with `DOCSCAN_API_KEY`. Persist the original image with the extracted JSON so reviewers can always compare.

**Web** — the review UI at `/` takes any number of documents (*Choose documents* as often as needed, or drop files on the page), lists receipts with crops, lets a reviewer correct what the scanner got wrong - **type** (a dropdown of expense kinds: BBM, Tol, Parkir, Pulsa, Servis kendaraan, and the kinds that go under Lain-lain), **date**, **merchant**, **Keterangan** and **amount** - tick "reviewed", and exports CSV/JSON. Edited boxes are outlined; setting a value back to what was read undoes the edit. A corrected type or merchant is carried into Keterangan until you type your own; the export puts each receipt in the column of its (edited) type and the row of its (edited) date, and *JSON data* keeps the scan as read with your changes under `userEdits`. Each file is one document = one claim: it is sent to `/v1/scan` on its own, so its dates and card balances are checked against its own receipts, a file that fails only fails itself (*Retry* / *Remove*), and documents added later are scanned without redoing the earlier ones. A receipt that shows up in two documents is flagged in both. With two or more documents, the export panel asks how they go into the CSV: *One claim per document* (a table per file, each under its file name) or *Combine into one claim* (all receipts in one table, sorted by date, one total - for pages or photos of the same claim); each option shows what it would produce, and the browser remembers the choice. The amount box is pre-filled with what goes on the claim: the handwritten claim if there is one, else the amount as read, else the best guess (marked). *Export Excel* and *CSV* write the operational claim form (FRM-S3.3-002 Rev.01) and nothing else - one table per document, each under its file name, and every document also has its own *Export* button (Excel). The Excel file is laid out like the paper form - *BBM* over KM / Liter / Rp, bold boxed headings, real dates (`dd/mm/yyyy`) and numbers with thousand separators, printable on one landscape page width. Documents claimed separately are stacked on the one *Claim Tunj Operasional* sheet, each table under its document's name. **Medical receipts** (hospital, clinic, doctor, pharmacy - or a kwitansi for treatment) are not part of it: they go to a second sheet, *Claim Medical Karyawan* - `No | Tanggal | Nilai Kwitansi | Nilai Claim | Deskripsi Pelayanan/Nama Obat | Nama Pasien di Kwitansi | Keterangan` - where Nilai Claim is 90 % of the receipt (rounded to the rupiah) and Keterangan is hospital or clinic + address + doctor. The scanner reads the patient (`patientName`), the doctor (`doctorName`, also from a `dr.` title), the address (`merchantAddress`) and the services or medicines (`serviceDescription`); patient and service can be corrected in a receipt's detail view. An optician's handwritten receipt (Nama / Frame / Lensa / Harga, with the R/L prescription table) is read as *Kacamata: frame + lensa ...*, the lens type matched against known types (Photochromic, Bluecut, Progressive ...), the price kept only when a reading shows its size and others agree on its digits. Measured on a printed sample clinic receipt and one real optician receipt so far. Both formats: one row per receipt, in date order (three receipts on one day are three rows), with `BBM (KM, Liter, Rp) | Tol | Parkir | Pulsa | Servis Kendaraan | Lain-lain | Total | Keterangan` and a Total row. KM stays empty; **Keterangan** is filled with each receipt's `description` and can be edited per receipt in the table before exporting; liters keep their decimals as printed (`3.20`, `2.80`), dates are `yyyy-mm-dd` so a spreadsheet cannot swap day and month, and receipts without a readable date come last. Per-receipt detail is in *JSON data*. The page is served `no-store` and carries a version (in the footer, also `ui` in `/healthz`): a tab left open across a deployment would otherwise keep building the export the old way from the new server's answers. Such a tab reloads itself before scanning when it has nothing to lose, and shows a *Reload* banner when it has.

## Configuration (env `DOCSCAN_*`)

| Variable | Default | Notes |
|---|---|---|
| `API_KEY` | – | Require `X-API-Key` |
| `CORS_ORIGINS` | – | Comma-separated origins for browser clients |
| `MAX_SIDE_LEN` | 2000 | Detector input size; raise for very dense pages (slower) |
| `REFINE_ENABLED` | true | Re-OCR of fields that fail checks |
| `SECOND_PASS` | true | A page whose amount or date is missing or doubtful is read once more from a copy with the light evened out (shadow, dim corner, coloured lamp); the better of the two readings is kept. Costs one more OCR pass on such pages only |
| `REVIEW_THRESHOLD` | 0.85 | Below → `needs_review` |
| `DATE_OUTLIER_DAYS` | 45 | Warn when far from the batch median date |
| `MASK_CARD_NUMBERS` | true | `0145********6480` in output |
| `MAX_UPLOAD_MB` / `MAX_FILES` / `MAX_PDF_PAGES` | 20 / 20 / 10 | Input limits |
| `MAX_CONCURRENT_SCANS` | 2 | Scans decoded at once per worker; later uploads wait their turn (see *Many people at once*) |
| `INTRA_OP_THREADS` | -1 (all cores) | OCR threads per worker; set to cores ÷ workers when running more than one worker |
| `VLM_MODE` | off | `needs_review` sends only unresolved receipts to a vision LLM |
| `VLM_BASE_URL` / `VLM_API_KEY` / `VLM_MODEL` | – | Any OpenAI-compatible `/chat/completions` endpoint (e.g. an LLM router) |
| `VLM_MAX_RECEIPTS` | 10 | Cost guard per request |

The vision fallback is the realistic route for handwriting (kwitansi amounts, bon kontan, a date written as a fraction - day over month - which every OCR model here reads as '%'). It also fills the medical fields (patient, doctor, address, service). Any OpenAI-compatible endpoint works, e.g. Anthropic (`https://api.anthropic.com/v1`, a Claude model) or Hugging Face Inference Providers (`https://router.huggingface.co/v1`, a vision model). It is off unless configured: receipt pictures then leave the machine. Its merge rules keep it from adding errors: it only fills missing/low-confidence fields, never overrides a value that passed a check, and anything it fills stays `needs_review`. The HTTP/merge logic is unit-tested with a mocked endpoint; extraction quality depends on the model you point it at and has not been measured here.

## Capture guidelines (biggest accuracy lever)

1. Do not highlight amounts with a marker — on a greyscale scan it buries the digits (all three toll fares in the samples). They are recovered where possible, but such a receipt needs a human look unless a balance chain confirms it.
2. Do not overlap receipts; a covered date cannot be recovered.
3. Write corrections next to, not over, printed totals.
4. Phone photos: flat, filled frame, no flash glare. A document-scanner capture mode in the app (edge detection + perspective flattening) helps more than any server-side tweak.

## Extending

Add a class in `docscan/templates/` with `doc_type`, `category`, `score(segment)` and `parse(segment, ctx)`, register it in `templates/__init__.py`, add its labels to `layout.LEXICON`, define its arithmetic `Rel`s, and add labelled samples to `tests/golden/`. A layout needs its own class only when the general `receipt` template is not enough (fields beyond the money rows, checks specific to the document); for a row name it does not know yet, add the word to `_NAMES` in `templates/receipt.py`.

## Known limitations

- Optical forms: `tests/golden/optical-handwritten.jpg` exercises the original uploaded photo with real OCR. It extracts the date 02/10/2026, Rp 350,000, patient name/address/phone, shop details, lens type and prescription sphere columns. Stacked day/month dates require repeated numeric evidence for all three date components and a valid calendar date. Empty prescription cells remain absent; a restored decimal is marked `inferred` and requires confirmation. The sample's frame code remains unclear and is explicitly flagged. Optical details are visible in the review dialog and included in JSON; claim exports retain the date, amount, patient and service fields. Other handwriting/layouts are not guaranteed to be fully readable.

- Reused backing paper: filtering uses multiple recognisable receipt sheets. When edges disappear, side-by-side SPBU receipts can also be isolated using separate station headings, dates, volumes, pump labels and payment rows. Subsidy footers stay with their fuel receipts. Ambiguous columns and independently readable documents outside the selected regions prevent masking; single documents and unsupported layouts retain ordinary parsing. `pages[].backgroundLinesIgnored` and a response warning report filtering. Original uploads are unchanged. The actual three-receipt photo is covered by `tests/golden/receipts-on-used-paper.jpg`: dates 28/09, 29/09 and 02/10/2026; volumes 4.2, 4 and 4 L; amounts Rp 42,000, Rp 40,000 and Rp 40,000. Its backing invoice and detached footers must not become extra receipts. Generated upright/sideways cases also cover Rp 400,000 over a backing total of Rp 1,432,432.40. Scan again after updating to replace earlier results.
- Invoices: built and tested on a single layout (one internet provider's invoice, which stays fully correct under rotation, scaling, blur and perspective). Other invoice layouts use the same general rules - captions, sums, one document per page - but are unmeasured; an invoice with several tax bases or a multi-page line table will need review.
- General receipts: about one total in ten is left for a person (label and amount both unreadable, or the sums disagree); foreign currencies with cents, receipts whose total is not in a row of its own, and multi-column invoices are not handled. Dates and shop names on such receipts are extracted but unmeasured.
- Handwriting: amounts and dates always stay `needs_review`; terbilang is only used when every word resolves. Focused crops and local ink thresholds recover the sample kwitansi's **2 October 2026 / Rp 600,000**, and the bon kontan's **3.5 L / Rp 35,000**. These results are covered by the real OCR regression tests; other handwriting remains unmeasured.
- Handwritten dates: accept complete DMY dates, including a printed year prefix (`4.07 20..26`). Calendar validation rejects impossible dates; missing years are never filled from today's date. Conflicting rereads leave `transactionDate` empty and return `transactionDateCandidates` plus a review reason. The sample bon kontan's date still needs confirmation. An accepted signing date stays at confidence ≤ 0.6; a billing period in the kwitansi purpose is not used as the signing date.
- Bon kontan volume and price: read `volumeLiters` from the handwritten quantity crop, retaining the decimal separator, raw OCR and bounding box. This works without a reference price, including Pertamax. `pricePerLiter` is then **derived** from amount / liters; this calculation does not increase confidence in its own inputs. If volume remains unreadable, the existing Pertalite sample reference of Rp 10,000/L can provide an explicitly flagged estimate at confidence ≤ 0.65. It is not a guarantee of the price on another date or at another location. Written quantities take precedence over this reference; a mismatch requests review. Vouchers with multiple priced fuel rows do not receive a single inferred volume or unit price.
- Photos of a claim page: e-toll slips are recognised through the usual misreadings (`e-To11`, `GO1-1`, `No se-i`, a cut-off card number after `CN:`); a slip whose date is covered by the next one keeps its fare and asks for the date.
- Receipts glued at mixed orientations on one page: the page is rotated to the majority orientation; a sideways minority receipt is not parsed.
- Station/merchant names keep OCR spacing quirks (`SANGKURIANGCIMAHE`).
- Lock-step processing per worker (one OCR at a time); scale with more workers or instances, each peaking at ~1.2–1.6 GB on a large multi-receipt page.
- PP-OCRv6 *small* models are bundled; the *tiny* variant may be faster on small VPSes but was not downloadable in the build environment, so it is untested.
