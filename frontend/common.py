"""Backend client, route styling and benchmark-result loading shared by every page."""
import json
import os
from pathlib import Path

import pandas as pd
import requests
import streamlit as st

BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000").rstrip("/")
RESULTS_DIR = Path(os.getenv("RESULTS_DIR", Path(__file__).resolve().parent.parent / "results"))

ROUTE_STYLE = {
    "direct_answer": {"label": "Direct answer", "icon": ":material/bolt:", "color": "green",
                      "about": "answered from the model's own knowledge"},
    "web_search": {"label": "Web search", "icon": ":material/travel_explore:", "color": "blue",
                   "about": "needs fresh information from the web"},
    "rag_retrieval": {"label": "Paper RAG", "icon": ":material/library_books:", "color": "violet",
                      "about": "answered from the 10 research papers"},
}


class BackendError(Exception):
    def __init__(self, status: int | None, detail: str):
        super().__init__(detail)
        self.status = status
        self.detail = detail


def _post(path: str, query: str, timeout: float) -> dict:
    try:
        resp = requests.post(f"{BACKEND_URL}{path}", json={"query": query}, timeout=timeout)
    except requests.RequestException as e:
        raise BackendError(None, f"Can't reach the backend at {BACKEND_URL}: {e}") from e
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("detail", resp.text)
        except ValueError:
            detail = resp.text
        raise BackendError(resp.status_code, str(detail))
    return resp.json()


@st.cache_data(ttl=15, show_spinner=False)
def health() -> dict | None:
    try:
        resp = requests.get(f"{BACKEND_URL}/health", timeout=5)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException:
        return None


def route(query: str) -> dict:
    return _post("/route", query, timeout=30)


def ask(query: str) -> dict:
    return _post("/ask", query, timeout=600)  # a RAG answer on the Groq free tier can take a minute


def route_badge(route_name: str):
    style = ROUTE_STYLE[route_name]
    st.badge(style["label"], icon=style["icon"], color=style["color"])


def probability_bars(probabilities: dict[str, float]):
    for name, p in sorted(probabilities.items(), key=lambda kv: -kv[1]):
        style = ROUTE_STYLE[name]
        st.progress(p, text=f"{style['icon']} {style['label']} · {p:.1%}")


@st.cache_data(show_spinner=False)
def load_result(name: str):
    path = RESULTS_DIR / name
    if not path.exists():
        return None
    if path.suffix == ".csv":
        return pd.read_csv(path)
    return json.loads(path.read_text(encoding="utf-8"))
