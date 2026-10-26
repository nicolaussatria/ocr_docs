FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 \
    OMP_NUM_THREADS=2 \
    DOCSCAN_MAX_UPLOAD_MB=20

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
 && python -c "from rapidocr import RapidOCR; RapidOCR(params={'Global.log_level':'error'})"   # fail the build if models are missing

COPY docscan ./docscan
# uid 1000 is what Hugging Face Spaces runs the container as; any non-root uid works elsewhere
RUN useradd -m -u 1000 app && chown -R app /app
USER app

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s CMD python -c "import urllib.request,sys; sys.exit(0 if b'ok' in urllib.request.urlopen('http://127.0.0.1:8000/healthz').read() else 1)"
# One worker per ~2 vCPU (with DOCSCAN_INTRA_OP_THREADS = cores / workers). Each worker holds its own models and
# peaks at ~1-1.6 GB; README "Many people at once" has measured numbers for 10 simultaneous users.
CMD ["uvicorn", "docscan.api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
