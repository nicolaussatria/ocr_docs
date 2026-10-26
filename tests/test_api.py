from pathlib import Path

import pytest
from fastapi.testclient import TestClient

GOLDEN = Path(__file__).parent / "golden"


@pytest.mark.slow
def test_scan_endpoint(monkeypatch):
    monkeypatch.setenv("DOCSCAN_API_KEY", "secret")
    from docscan.api import app

    with TestClient(app) as c:
        health = c.get("/healthz").json()
        assert health["status"] == "ok"
        page = c.get("/")
        # the page is never cached and knows its own version, so a stale tab can tell
        assert page.status_code == 200 and "no-store" in page.headers["cache-control"]
        assert f'const BUILD = "{health["ui"]}"' in page.text and "__UI_BUILD__" not in page.text
        f = {"files": ("page2.jpg", (GOLDEN / "page2.jpg").read_bytes(), "image/jpeg")}
        assert c.post("/v1/scan", files=f).status_code == 401
        r = c.post("/v1/scan?includeImages=true&today=2026-10-12", files=f, headers={"X-API-Key": "secret"})
        assert r.status_code == 200
        body = r.json()
        assert body["schemaVersion"] == "1.0" and body["summary"]["receiptCount"] == 8
        rec = body["receipts"][0]
        assert {"docType", "status", "confidence", "fields", "checks", "reviewReasons", "bbox", "image"} <= rec.keys()
        assert all("*" in r["fields"]["cardNumber"]["value"] for r in body["receipts"] if "cardNumber" in r["fields"])
        bad = c.post("/v1/scan", files={"files": ("x.jpg", b"not an image", "image/jpeg")},
                     headers={"X-API-Key": "secret"})
        assert bad.status_code == 422
