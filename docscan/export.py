"""The claim forms as one Excel workbook: operational allowance and medical.

What goes on the forms - one row per receipt, with the reviewer's corrections - is decided by the
review page, which holds those corrections; it sends the rows here and this module only lays them
out the way the paper forms look.

* Claim Tunj Operasional (FRM-S3.3-002 Rev.01): 'BBM' over KM / Liter / Rp, one column per kind of
  expense, a Total row. Documents claimed separately are stacked on the one sheet, each table under
  the document's name.
* Claim Medical: what the receipt says was paid (Nilai Kwitansi), the 90 % that is claimed (Nilai
  Claim), what it was for, the patient, and - under Keterangan - hospital or clinic, address, doctor.
* Attachment: the picture of every receipt on the forms, in the order and with the numbers of their
  rows, so whoever checks the claim sees each receipt beside what was claimed for it. A row's No links
  to its picture and the picture's heading back to the row. Only made when the rows carry pictures.

Dates are real dates and amounts real numbers, so a spreadsheet sorts and adds them without
mixing up day and month, and each sheet prints on one landscape page width.
"""
from __future__ import annotations

import base64
import binascii
import datetime as dt
import io
import math
import re
from dataclasses import dataclass

from openpyxl import Workbook
from openpyxl.drawing.image import Image as XlImage
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.worksheet.hyperlink import Hyperlink
from openpyxl.worksheet.pagebreak import Break
from PIL import Image as PILImage
from pydantic import BaseModel, Field, field_validator, model_validator

MONEY = ["bbm", "tol", "parkir", "pulsa", "servis", "lain"]
CLAIM_TITLE, CLAIM_SHEET = "Claim Tunjangan Operasional Karyawan", "Claim Tunj Operasional"
MEDICAL_TITLE, MEDICAL_SHEET = "Claim Medical Karyawan", "Claim Medical"
ATTACH_TITLE, ATTACH_SHEET = "Attachment - Struk dan Kwitansi", "Attachment"
KINDS = {"bbm": "BBM", "tol": "Tol", "parkir": "Parkir", "pulsa": "Pulsa", "servis": "Servis Kendaraan", "lain": "Lain-lain"}
IMAGE_MAX_B64 = 3_000_000           # one receipt picture (the scan sends <= 900 px JPEGs, ~0.1 MB)
IMAGES_MAX_B64 = 150_000_000        # all of them in one export
# column, heading, sub-heading (the row under it), width
LAYOUT = [("A", "No", None, 5), ("B", "Tanggal", None, 12), ("C", "BBM", "KM", 8), ("D", None, "Liter", 9),
          ("E", None, "Rp", 13), ("F", "Tol", None, 12), ("G", "Parkir", None, 12), ("H", "Pulsa", None, 12),
          ("I", "Servis Kendaraan", None, 13), ("J", "Lain-lain", None, 13), ("K", "Total", None, 14),
          ("L", "Keterangan", None, 60)]
MEDICAL_LAYOUT = [("A", "No", 5), ("B", "Tanggal", 12), ("C", "Nilai Kwitansi", 15), ("D", "Nilai Claim", 15),
                  ("E", "Deskripsi Pelayanan/Nama Obat", 38), ("F", "Nama Pasien di Kwitansi", 26), ("G", "Keterangan", 60)]
AMOUNT, LITERS, DAY = "#,##0", "0.00", "dd/mm/yyyy"
XLSX_TYPE = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def medical_claim(amount: int) -> int:
    """The medical claim is 90 % of what the receipt says was paid, in whole rupiah, rounded half up."""
    return (amount * 9 + 5) // 10


def _picture_b64(value: str | None) -> str | None:
    """A receipt picture as the scan returned it: base64 JPEG (or PNG). Anything else is refused."""
    if not value:
        return None
    try:
        head = base64.b64decode(value[:64], validate=True)
        base64.b64decode(value, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise ValueError("image is not base64") from exc
    if not head.startswith((b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n")):
        raise ValueError("image must be a JPEG or PNG")
    return value


class ClaimRow(BaseModel):
    date: dt.date | None = None                 # the field name is the JSON key
    liters: float | None = Field(None, ge=0, le=10_000)
    bbm: int = 0
    tol: int = 0
    parkir: int = 0
    pulsa: int = 0
    servis: int = 0
    lain: int = 0
    note: str = Field("", max_length=1000)
    image: str | None = Field(None, max_length=IMAGE_MAX_B64)   # the receipt's picture, for the Attachment sheet

    @field_validator("image")
    @classmethod
    def check_image(cls, value: str | None) -> str | None:
        return _picture_b64(value)


class ClaimTable(BaseModel):
    name: str = Field("Claim", max_length=200)
    rows: list[ClaimRow] = Field(default_factory=list, max_length=5000)


class MedicalRow(BaseModel):
    date: dt.date | None = None
    amount: int = Field(0, ge=0)                 # Nilai Kwitansi: what the receipt says was paid
    service: str = Field("", max_length=1000)    # Deskripsi Pelayanan/Nama Obat
    patient: str = Field("", max_length=200)     # Nama Pasien di Kwitansi
    note: str = Field("", max_length=1000)       # Keterangan: hospital or clinic, address, doctor
    image: str | None = Field(None, max_length=IMAGE_MAX_B64)

    @field_validator("image")
    @classmethod
    def check_image(cls, value: str | None) -> str | None:
        return _picture_b64(value)


class ClaimExport(BaseModel):
    tables: list[ClaimTable] = Field(default_factory=list, max_length=50)    # operational claims, one table each
    medical: list[MedicalRow] = Field(default_factory=list, max_length=5000)

    @model_validator(mode="after")
    def _something(self):
        if not self.tables and not self.medical:
            raise ValueError("nothing to export")
        rows = [r for t in self.tables for r in t.rows] + list(self.medical)
        if sum(len(r.image or "") for r in rows) > IMAGES_MAX_B64:
            raise ValueError("too many or too large receipt pictures for one export")
        return self


class _Style:
    thin = Side(style="thin", color="9AA7BB")
    box = Border(left=thin, right=thin, top=thin, bottom=thin)
    head_fill = PatternFill("solid", fgColor="E9F0FC")
    bold, label = Font(bold=True), Font(bold=True, italic=True, size=12)
    centre = Alignment(horizontal="center", vertical="center", wrap_text=True)


def _title(ws, text: str, last_col: str) -> None:
    ws.merge_cells(f"A1:{last_col}1")
    ws["A1"] = text
    ws["A1"].font = Font(bold=True, size=16)
    ws["A1"].alignment = Alignment(horizontal="left", vertical="center")
    ws.row_dimensions[1].height = 28
    ws.row_dimensions[2].height = 10


def _print_setup(ws, title_rows: str | None) -> None:
    ws.page_setup.orientation = "landscape"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True
    if title_rows:
        ws.print_title_rows = title_rows


def _document(name: str) -> str:
    return re.sub(r"\.[A-Za-z0-9]{2,4}$", "", name).strip() or name


def _claim_table(ws, top: int, table: ClaimTable) -> int:
    """One claim's table with its two heading rows starting at row `top`; returns the Total row."""
    s = _Style
    for col, head, sub, width in LAYOUT:
        ws.column_dimensions[col].width = width
        ws[f"{col}{top}"].value, ws[f"{col}{top + 1}"].value = head, sub
        for r in (top, top + 1):
            cell = ws[f"{col}{r}"]
            cell.font, cell.alignment, cell.fill, cell.border = s.bold, s.centre, s.head_fill, s.box
    for col, head, sub, _ in LAYOUT:          # merged only once written: a merged cell takes no value
        if head is not None and sub is None:
            ws.merge_cells(f"{col}{top}:{col}{top + 1}")
    ws.merge_cells(f"C{top}:E{top}")
    last = top + 1
    for i, row in enumerate(table.rows, start=1):
        last = top + 1 + i
        money = [getattr(row, k) for k in MONEY]
        values = [i, row.date, None, row.liters or None, *[v or None for v in money], sum(money) or None, row.note]
        for (col, *_), value in zip(LAYOUT, values):
            ws[f"{col}{last}"].value = value
    total = last + 1
    ws[f"B{total}"] = "Total"
    ws[f"D{total}"] = round(sum(r.liters or 0 for r in table.rows), 2) or None
    for col, key in zip("EFGHIJ", MONEY):
        ws[f"{col}{total}"] = sum(getattr(r, key) for r in table.rows) or None
    ws[f"K{total}"] = sum(getattr(r, k) for r in table.rows for k in MONEY) or None
    for r in range(top + 2, total + 1):
        for col, *_ in LAYOUT:
            cell = ws[f"{col}{r}"]
            cell.border = s.box
            cell.number_format = DAY if col == "B" else LITERS if col == "D" else AMOUNT if col in "EFGHIJK" else "General"
            if r == total:
                cell.font = s.bold
    return total


def _operational_sheet(wb, tables: list[ClaimTable]) -> list[int]:
    """The sheet; returns the row of each table's first receipt."""
    ws = wb.create_sheet(CLAIM_SHEET)
    _title(ws, CLAIM_TITLE, "L")
    if len(tables) == 1:
        _claim_table(ws, 3, tables[0])
        ws.freeze_panes = "A5"
        _print_setup(ws, "1:4")
        return [5]
    # several claims on one sheet, each under its document's name and two empty rows apart
    top, firsts = 2, []
    for table in tables:
        ws[f"A{top}"] = _document(table.name)
        ws[f"A{top}"].font = _Style.label
        firsts.append(top + 3)
        top = _claim_table(ws, top + 1, table) + 3
    _print_setup(ws, "1:1")
    return firsts


def _medical_sheet(wb, rows: list[MedicalRow]) -> None:
    s = _Style
    ws = wb.create_sheet(MEDICAL_SHEET)
    _title(ws, MEDICAL_TITLE, "G")
    for col, head, width in MEDICAL_LAYOUT:
        ws.column_dimensions[col].width = width
        cell = ws[f"{col}3"]
        cell.value, cell.font, cell.alignment, cell.fill, cell.border = head, s.bold, s.centre, s.head_fill, s.box
    ws.row_dimensions[3].height = 32
    last = 3
    for i, row in enumerate(rows, start=1):
        last = 3 + i
        values = [i, row.date, row.amount or None, medical_claim(row.amount) or None, row.service, row.patient, row.note]
        for (col, *_), value in zip(MEDICAL_LAYOUT, values):
            ws[f"{col}{last}"].value = value
    total = last + 1
    ws[f"B{total}"] = "Total"
    ws[f"D{total}"] = sum(medical_claim(r.amount) for r in rows) or None
    for r in range(4, total + 1):
        for col, *_ in MEDICAL_LAYOUT:
            cell = ws[f"{col}{r}"]
            cell.border = s.box
            cell.number_format = DAY if col == "B" else AMOUNT if col in "CD" else "General"
            if r == total:
                cell.font = s.bold
    ws.freeze_panes = "A4"
    _print_setup(ws, "1:3")


@dataclass
class _Attachment:
    sheet: str                   # the claim form the receipt is on ...
    row: int                     # ... and its row there
    heading: str                 # 'Claim Tunj Operasional - page-1 - No 3'
    detail: str                  # date, what was claimed, Keterangan
    image: str | None


# Receipts are tall: a picture is shown at most this big (px), and kept whole on one printed page
PICTURE_W, PICTURE_H, ROW_PX, PAGE_PX = 480, 640, 20, 940


def _rupiah(v: int) -> str:
    return "Rp " + f"{v:,}".replace(",", ".")


def _day(d: dt.date | None) -> str:
    return d.strftime("%d/%m/%Y") if d else "tanggal tidak terbaca"


def _picture(b64: str) -> XlImage | None:
    data = base64.b64decode(b64)
    try:
        with PILImage.open(io.BytesIO(data)) as im:
            (w, h), kind = im.size, im.format
    except Exception:            # noqa: BLE001 - a broken picture leaves its receipt without one, not the export
        return None
    if kind not in ("JPEG", "PNG") or not w or not h or w * h > 40_000_000:
        return None
    pic = XlImage(io.BytesIO(data))
    scale = min(1.0, PICTURE_W / w, PICTURE_H / h)
    pic.width, pic.height = max(1, round(w * scale)), max(1, round(h * scale))
    return pic


def _link(cell, sheet: str, row: int) -> None:
    cell.hyperlink = Hyperlink(ref=cell.coordinate, location=f"'{sheet}'!A{row}")
    cell.font = Font(color="1F5FBF", underline="single", bold=cell.font.bold)


def _attachment_sheet(wb, items: list[_Attachment]) -> None:
    ws = wb.create_sheet(ATTACH_SHEET)
    _title(ws, ATTACH_TITLE, "H")
    row, used = 3, 38 + 13                         # the title rows take the top of the first page
    for item in items:
        pic = _picture(item.image) if item.image else None
        picture_rows = math.ceil(pic.height / ROW_PX) if pic else 1
        height = (2 + picture_rows + 1) * ROW_PX
        if used + height > PAGE_PX and used > 0:   # start the receipt on a new page rather than cut it in two
            ws.row_breaks.append(Break(id=row - 1))
            used = 0
        ws[f"A{row}"] = item.heading
        ws[f"A{row}"].font = _Style.bold
        _link(ws[f"A{row}"], item.sheet, item.row)                     # back to its row on the form
        _link(wb[item.sheet][f"A{item.row}"], ATTACH_SHEET, row)        # and the row's No to its picture
        ws[f"A{row + 1}"] = item.detail
        if pic:
            ws.add_image(pic, f"A{row + 2}")
        else:
            ws[f"A{row + 2}"] = "(gambar struk tidak tersedia)"
            ws[f"A{row + 2}"].font = Font(italic=True, color="686D83")
        row += 2 + picture_rows + 1
        used += height
    ws.page_setup.orientation = "portrait"
    ws.page_setup.fitToWidth, ws.page_setup.fitToHeight = 1, 0
    ws.sheet_properties.pageSetUpPr.fitToPage = True


def build_xlsx(claim: ClaimExport) -> bytes:
    wb = Workbook()
    wb.remove(wb.active)
    tables = [t for t in claim.tables if t.rows] or ([] if claim.medical else claim.tables)
    items: list[_Attachment] = []
    if tables:
        for table, first in zip(tables, _operational_sheet(wb, tables)):
            name = f" - {_document(table.name)}" if len(tables) > 1 else ""
            for i, r in enumerate(table.rows):
                claimed = [f"{KINDS[k]} {_rupiah(getattr(r, k))}" for k in MONEY if getattr(r, k)]
                if r.liters:
                    claimed.append(f"{r.liters:.2f} L".replace(".", ","))
                items.append(_Attachment(CLAIM_SHEET, first + i, f"{CLAIM_SHEET}{name} - No {i + 1}",
                                         " | ".join(x for x in [_day(r.date), *claimed, r.note] if x), r.image))
    if claim.medical:
        _medical_sheet(wb, claim.medical)
        for i, r in enumerate(claim.medical):
            detail = [_day(r.date), f"Nilai Kwitansi {_rupiah(r.amount)}", f"Nilai Claim {_rupiah(medical_claim(r.amount))}",
                      r.patient, r.note]
            items.append(_Attachment(MEDICAL_SHEET, 4 + i, f"{MEDICAL_SHEET} - No {i + 1}",
                                     " | ".join(x for x in detail if x), r.image))
    if any(item.image for item in items):
        _attachment_sheet(wb, items)
    out = io.BytesIO()
    wb.save(out)
    return out.getvalue()
