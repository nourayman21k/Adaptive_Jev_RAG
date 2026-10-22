"""FastAPI backend: the fine-tuned BERT router and the adaptive RAG pipeline behind three endpoints.

    GET  /health   what is loaded (router, index size, LLM)
    POST /route    BERT routing only: no LLM call, no API key needed
    POST /ask      the full pipeline: route, then answer

Run locally with:  uv run uvicorn adaptive_jev_rag.api:app --app-dir src --port 8000
"""
import json
import time
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from groq import RateLimitError
from pydantic import BaseModel, Field

from . import knowledge_base
from .config import LLM_MODEL, ROUTER_DIR, ROUTES, THRESHOLD
from .pipeline import AdaptivePipeline, LLMUnavailable, make_llm
from .router import BertRouter

services: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_dotenv()
    start = time.perf_counter()
    router = BertRouter()
    router.route("warm up")  # the first forward pass is slow (CUDA/kernel setup); pay it before the first user does
    store = knowledge_base.load(knowledge_base.load_embeddings())
    services["pipeline"] = AdaptivePipeline(router, store, make_llm())
    services["chunks"] = store._collection.count()
    metrics_file = ROUTER_DIR / "metrics.json"
    services["router_metrics"] = json.loads(metrics_file.read_text(encoding="utf-8")) if metrics_file.exists() else None
    services["startup_s"] = time.perf_counter() - start
    yield
    services.clear()


app = FastAPI(title="Adaptive RAG API", version="1.0", lifespan=lifespan)


class Query(BaseModel):
    query: str = Field(min_length=1, max_length=2000)


@app.get("/health")
def health():
    pipeline: AdaptivePipeline = services["pipeline"]
    return {
        "status": "ok",
        "router": "bert-base-uncased (fine-tuned)",
        "router_device": pipeline.router.device,
        "router_metrics": services["router_metrics"],
        "routes": ROUTES,
        "threshold": THRESHOLD,
        "indexed_chunks": services["chunks"],
        "llm": LLM_MODEL if pipeline.llm is not None else None,
        "startup_s": services["startup_s"],
    }


@app.post("/route")
def route(q: Query):
    start = time.perf_counter()
    decision = services["pipeline"].router.route(q.query)
    return {
        "query": q.query,
        "route": decision.route,
        "confidence": decision.confidence,
        "probabilities": decision.probabilities,
        "final_route": decision.final_route,
        "input_tokens": decision.input_tokens,
        "latency_ms": (time.perf_counter() - start) * 1000,
    }


@app.post("/ask")
def ask(q: Query):
    try:
        return services["pipeline"].ask(q.query)
    except LLMUnavailable as e:
        raise HTTPException(status_code=503, detail=str(e))
    except RateLimitError as e:
        raise HTTPException(status_code=429, detail=f"Groq rate limit reached: {e}")
