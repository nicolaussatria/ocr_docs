"""HTTP API.

  POST /v1/scan     multipart 'files' (JPEG/PNG/WebP/PDF, many receipts per page OK)
  GET  /v1/schema   JSON Schema of the response
  GET  /healthz     liveness + engine info
  POST /v1/export/xlsx  claim rows (JSON) -> the claim forms (operational, medical) as an Excel workbook
  GET  /            review UI (upload, inspect, edit, export Excel / CSV)

Run:  uvicorn docscan.api:app --host 0.0.0.0 --port 8000 --workers 1
"""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import logging
import os
from contextlib import asynccontextmanager
from datetime import date
from pathlib import Path

import numpy as np
from fastapi import Depends, FastAPI, File, Header, HTTPException, Query, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response

from .config import Settings
from .export import XLSX_TYPE, ClaimExport, build_xlsx
from .imaging import InputError
from .pipeline import DocScanner
from .schema import ScanResult

log = logging.getLogger("docscan")
STATIC = Path(__file__).parent / "static"
_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.from_env()
    scanner = DocScanner(settings)
    # warm-up: loads ONNX sessions so the first real request is not slow
    blank = np.full((64, 256, 3), 255, np.uint8)
    try:
        scanner.ocr.read(blank)
    except Exception:  # pragma: no cover
        log.warning("warm-up failed", exc_info=True)
    _state["scanner"] = scanner
    yield
    _state.clear()


app = FastAPI(title="DocScan – receipt extraction", version="1.0.0", lifespan=lifespan)
_origins = [o.strip() for o in os.getenv("DOCSCAN_CORS_ORIGINS", "").split(",") if o.strip()]
if _origins:
    app.add_middleware(CORSMiddleware, allow_origins=_origins, allow_methods=["GET", "POST"], allow_headers=["*"])


def require_key(x_api_key: str | None = Header(default=None)):
    expected = os.getenv("DOCSCAN_API_KEY", "")
    if expected and not (x_api_key and hmac.compare_digest(x_api_key, expected)):
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


def ui_build() -> str:
    """Fingerprint of the review page. The page carries it and compares it with /healthz, so a tab
    left open across a deployment notices that it is an old copy talking to a new server."""
    return hashlib.sha1((STATIC / "review.html").read_bytes()).hexdigest()[:8]


@app.get("/healthz")
def healthz():
    s: DocScanner | None = _state.get("scanner")
    return {"status": "ok" if s else "starting", "engine": s.ocr.version if s else None, "ui": ui_build()}


@app.get("/v1/schema")
def schema():
    return ScanResult.model_json_schema(by_alias=True)


@app.post("/v1/scan", response_model=ScanResult, response_model_exclude_none=True, dependencies=[Depends(require_key)])
async def scan(
    files: list[UploadFile] = File(..., description="One or more images / PDFs"),
    include_images: bool = Query(False, alias="includeImages", description="Embed a base64 JPEG crop per receipt"),
    include_raw_text: bool = Query(False, alias="includeRawText"),
    today: date | None = Query(None, description="Reference date for date checks (default: server date, UTC)"),
):
    scanner: DocScanner = _state["scanner"]
    limit = scanner.settings.max_upload_mb * 1024 * 1024
    payload = []
    for f in files:
        data = await f.read(limit + 1)
        if len(data) > limit:
            raise HTTPException(413, f"{f.filename}: larger than {scanner.settings.max_upload_mb} MB")
        payload.append((f.filename or "upload", data))
    # first come, first served: a waiting scan holds its upload, not its decoded pages, nor a thread
    if "slots" not in _state:
        _state["slots"] = asyncio.Semaphore(max(1, scanner.settings.max_concurrent_scans))
    async with _state["slots"]:
        try:
            return await run_in_threadpool(scanner.scan, payload, include_images=include_images,
                                           include_raw_text=include_raw_text, today=today)
        except InputError as exc:
            raise HTTPException(422, str(exc)) from exc


@app.post("/v1/export/xlsx", dependencies=[Depends(require_key)])
def export_xlsx(body: ClaimExport):
    """The review page's claim rows as one .xlsx workbook: Claim Tunj Operasional and, when there are
    medical receipts, Claim Medical."""
    return Response(build_xlsx(body), media_type=XLSX_TYPE,
                    headers={"Content-Disposition": 'attachment; filename="claim.xlsx"'})


@app.get("/", include_in_schema=False)
def ui():
    html = (STATIC / "review.html").read_text(encoding="utf-8").replace("__UI_BUILD__", ui_build())
    # never from a cache: the export is built by this page, and an old copy builds it the old way
    return HTMLResponse(html, headers={"Cache-Control": "no-store, max-age=0"})


@app.exception_handler(Exception)
async def unhandled(_, exc: Exception):  # pragma: no cover
    log.exception("unhandled error")
    return JSONResponse(status_code=500, content={"detail": "internal error"})
