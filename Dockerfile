FROM python:3.12-slim

# Offline runtime: no network, read-only root, /tmp tmpfs, 4 vCPU / 8 GiB.
RUN apt-get update && apt-get install -y --no-install-recommends \
      tesseract-ocr tesseract-ocr-eng \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt
# rapidocr-onnxruntime declares GUI opencv-python, whose import needs libGL -
# absent from slim images. cv2 import then fails and, because per-page error
# isolation swallows it, every page silently degrades to native text only.
# Force the headless build back on top so cv2 works without X libraries.
RUN pip install --no-cache-dir -r /app/requirements.txt  && pip uninstall -y opencv-python  && pip install --no-cache-dir --force-reinstall opencv-python-headless==4.11.0.86

COPY run.sh /app/run.sh
COPY mib /app/mib
COPY policy /app/policy
RUN chmod +x /app/run.sh

ENV PYTHONUNBUFFERED=1 \
    OMP_THREAD_LIMIT=1 \
    TMPDIR=/tmp

ENTRYPOINT ["/app/run.sh"]
