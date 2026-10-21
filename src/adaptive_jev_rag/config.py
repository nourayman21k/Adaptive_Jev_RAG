"""Settings shared by the router training script, the knowledge base, the pipeline and the API.

Every path and model name can be overridden with an environment variable, which is how the Docker image points them
at its baked-in copies.
"""
import os
from pathlib import Path

PROJECT_ROOT = Path(os.getenv("APP_ROOT", Path(__file__).resolve().parents[2]))

ROUTES = ["direct_answer", "web_search", "rag_retrieval"]
# below this router confidence the query goes to rag_retrieval, the most capable route
THRESHOLD = float(os.getenv("ROUTER_THRESHOLD", "0.7"))

BERT_BASE = os.getenv("BERT_BASE", "bert-base-uncased")
ROUTER_DIR = Path(os.getenv("ROUTER_DIR", PROJECT_ROOT / "models" / "bert-router"))
ROUTER_DATASET = Path(os.getenv("ROUTER_DATASET", PROJECT_ROOT / "eval" / "router_dataset.json"))
ROUTER_MAX_LENGTH = 64

PDF_DIR = Path(os.getenv("PDF_DIR", PROJECT_ROOT / "pdf"))
CHROMA_DIR = Path(os.getenv("CHROMA_DIR", PROJECT_ROOT / "chroma_db"))
COLLECTION = "papers"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-mpnet-base-v2")
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 150
RETRIEVER_K = 6

LLM_MODEL = os.getenv("LLM_MODEL", "openai/gpt-oss-120b")
LLM_MAX_RETRIES = 8  # rides out Groq free-tier per-minute rate limits
WEB_RESULTS = 5
