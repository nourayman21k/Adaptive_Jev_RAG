"""The fine-tuned BERT router: one query in, a route and its confidence out."""
from dataclasses import dataclass
from pathlib import Path

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import ROUTER_DIR, ROUTER_MAX_LENGTH, THRESHOLD


@dataclass
class RouteDecision:
    route: str
    confidence: float
    probabilities: dict[str, float]
    input_tokens: int

    @property
    def final_route(self) -> str:
        """The route the pipeline actually takes: low-confidence decisions fall back to rag_retrieval."""
        return self.route if self.confidence >= THRESHOLD else "rag_retrieval"


class BertRouter:
    def __init__(self, model_dir: Path = ROUTER_DIR, device: str | None = None):
        if not (Path(model_dir) / "config.json").exists():
            raise FileNotFoundError(
                f"No trained router in {model_dir}. Train it first: python -m adaptive_jev_rag.train_router")
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir).to(self.device).eval()
        self.routes = [self.model.config.id2label[i] for i in range(self.model.config.num_labels)]

    @torch.inference_mode()
    def route(self, query: str) -> RouteDecision:
        enc = self.tokenizer(query, truncation=True, max_length=ROUTER_MAX_LENGTH, return_tensors="pt").to(self.device)
        probs = self.model(**enc).logits.softmax(-1)[0].tolist()
        i = max(range(len(probs)), key=probs.__getitem__)
        return RouteDecision(
            route=self.routes[i],
            confidence=probs[i],
            probabilities=dict(zip(self.routes, probs)),
            input_tokens=enc["input_ids"].shape[1],
        )
