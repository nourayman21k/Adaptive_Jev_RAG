"""Fine-tune the BERT router on eval/router_dataset.json and save it for the API.

Same recipe as adaptive_bert.ipynb: bert-base-uncased, 8 epochs, batch size 16, AdamW (lr 5e-5, weight decay 0.01),
seed 42. The held-out test split is scored after training and written to metrics.json next to the model.

    python -m adaptive_jev_rag.train_router [--output DIR] [--epochs N]
"""
import argparse
import json
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from .config import BERT_BASE, ROUTER_DATASET, ROUTER_DIR, ROUTER_MAX_LENGTH, ROUTES


def train(output: Path, epochs: int = 8, seed: int = 42) -> dict:
    data = json.loads(ROUTER_DATASET.read_text(encoding="utf-8"))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)

    tokenizer = AutoTokenizer.from_pretrained(BERT_BASE)
    model = AutoModelForSequenceClassification.from_pretrained(
        BERT_BASE, num_labels=len(ROUTES), id2label=dict(enumerate(ROUTES)), label2id={r: i for i, r in enumerate(ROUTES)}
    ).to(device)

    def collate(batch):
        enc = tokenizer([b["query"] for b in batch], padding=True, truncation=True,
                        max_length=ROUTER_MAX_LENGTH, return_tensors="pt")
        enc["labels"] = torch.tensor([ROUTES.index(b["label"]) for b in batch])
        return enc

    loader = DataLoader(data["train"], batch_size=16, shuffle=True, collate_fn=collate,
                        generator=torch.Generator().manual_seed(seed))
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)

    start = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        total = 0.0
        for batch in loader:
            loss = model(**{k: v.to(device) for k, v in batch.items()}).loss
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += loss.item()
        print(f"epoch {epoch + 1}: loss {total / len(loader):.4f}", flush=True)
    train_s = time.perf_counter() - start

    model.eval()
    correct = 0
    with torch.inference_mode():
        for row in data["test"]:
            enc = tokenizer(row["query"], truncation=True, max_length=ROUTER_MAX_LENGTH, return_tensors="pt").to(device)
            correct += ROUTES[int(model(**enc).logits.argmax(-1))] == row["label"]

    output.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output)
    tokenizer.save_pretrained(output)
    metrics = {
        "base_model": BERT_BASE,
        "device": device,
        "epochs": epochs,
        "train_examples": len(data["train"]),
        "train_s": train_s,
        "test_examples": len(data["test"]),
        "test_accuracy": correct / len(data["test"]),
    }
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output", type=Path, default=ROUTER_DIR)
    ap.add_argument("--epochs", type=int, default=8)
    args = ap.parse_args()
    m = train(args.output, args.epochs)
    print(f"saved to {args.output}: test accuracy {m['test_accuracy']:.3f} on {m['test_examples']} queries, "
          f"trained in {m['train_s']:.1f}s on {m['device']}")
