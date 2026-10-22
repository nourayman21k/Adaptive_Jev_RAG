# Streamlit frontend. Build from the repository root so the benchmark results can be copied in:
#   docker build -f frontend/Dockerfile -t adaptive-rag-frontend .
FROM python:3.13-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    BACKEND_URL=http://backend:8000 \
    RESULTS_DIR=/app/results

WORKDIR /app
COPY frontend/requirements.txt .
RUN pip install -r requirements.txt

COPY results/ results/
COPY frontend/ .

EXPOSE 8501
HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')"
CMD ["streamlit", "run", "streamlit_app.py", "--server.address=0.0.0.0", "--server.port=8501"]
