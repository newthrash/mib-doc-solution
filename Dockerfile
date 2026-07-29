FROM python:3.12-slim

# Offline runtime: no network, read-only root, /tmp tmpfs, 4 vCPU / 8 GiB.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY run.sh /app/run.sh
COPY mib /app/mib
COPY policy /app/policy
RUN chmod +x /app/run.sh

ENV PYTHONUNBUFFERED=1 \
    OMP_THREAD_LIMIT=1 \
    TMPDIR=/tmp

ENTRYPOINT ["/app/run.sh"]
