# FastAPI backend with the fine-tuned BERT router. Build from the repository root:
#   docker build -f backend/Dockerfile -t adaptive-rag-backend .
#
# The router is fine-tuned and the Chroma index is built during the build, so the image starts without downloading
# or training anything, and no model weights need to live in git.
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app/src \
    APP_ROOT=/app \
    HF_HOME=/app/.cache/huggingface \
    TOKENIZERS_PARALLELISM=false

WORKDIR /app

# CPU-only PyTorch: the CUDA build would add several GB, and BERT routes a query in milliseconds on a CPU
RUN pip install --index-url https://download.pytorch.org/whl/cpu torch==2.14.1
COPY backend/requirements.txt backend/requirements.txt
RUN pip install -r backend/requirements.txt

# Train the router and build the index first, so editing the API code doesn't redo them
COPY src/adaptive_jev_rag/__init__.py src/adaptive_jev_rag/config.py src/adaptive_jev_rag/train_router.py \
     src/adaptive_jev_rag/knowledge_base.py src/adaptive_jev_rag/
COPY eval/router_dataset.json eval/router_dataset.json
COPY pdf/ pdf/
RUN python -m adaptive_jev_rag.train_router && python -m adaptive_jev_rag.knowledge_base

COPY src/ src/

# everything the app needs is now in the image; never call out to the Hugging Face Hub at runtime
ENV HF_HUB_OFFLINE=1

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "adaptive_jev_rag.api:app", "--host", "0.0.0.0", "--port", "8000"]
