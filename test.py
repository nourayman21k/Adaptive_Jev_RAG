import torch
from so1 import Decider, Choice

MODEL = "Qwen/Qwen3.5-4B"
THRESHOLD = 0.75

ROUTES = ["direct_answer", "web_search", "rag_retrieval"]

ROUTE_DESCRIPTIONS = """You are routing a user query to one of three routes based on how complex it is:
- direct_answer: the query is simple (greetings, basic facts, definitions, simple math). The LLM answers it directly from its own knowledge.
- web_search: the query is a little complex. It needs extra or up-to-date information from the internet (news, prices, recent events, current versions).
- rag_retrieval: the query is very complex. It needs deep, detailed analysis using information retrieved from the internal document knowledge base."""

TEST_QUERIES = [
    ("Hi, how are you?", "direct_answer"),
    ("What is the capital of France?", "direct_answer"),
    ("What is 15 times 4?", "direct_answer"),
    ("What is the price of Bitcoin today?", "web_search"),
    ("Who won yesterday's Champions League match?", "web_search"),
    ("What are the latest features in the newest Python release?", "web_search"),
    ("Based on our internal documents, compare the verification process of project A and project B and summarize the main risks.", "rag_retrieval"),
    ("Analyze all the clinic's treatment records and explain which procedures had the most complications and why.", "rag_retrieval"),
    ("Using our uploaded reports, explain step by step how our system architecture handles failures across all modules.", "rag_retrieval"),
]


def load_decider():
    if torch.cuda.is_available():
        return Decider.from_pretrained(MODEL, backend="hf", dtype=torch.bfloat16)
    print("No GPU found - using the small CPU test model (results will be weak).")
    return Decider.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", backend="hf")


def route_query(decider, query):
    decision = decider.decide(
        state=f"{ROUTE_DESCRIPTIONS}\n\nUser query: {query}",
        questions=[Choice("Which route should handle this query?", ROUTES)],
        mode="separate",
    )[0]
    return decision.choice, decision.confidence


def answer_directly(query):
    return "[LLM] answering directly"


def web_search(query):
    return "[WEB SEARCH] searching the internet"


def rag_retrieval(query):
    return "[RAG] retrieving from the knowledge base"


HANDLERS = {
    "direct_answer": answer_directly,
    "web_search": web_search,
    "rag_retrieval": rag_retrieval,
}


def main():
    decider = load_decider()
    correct = 0

    for query, expected in TEST_QUERIES:
        route, confidence = route_query(decider, query)
        is_correct = route == expected
        correct += is_correct
        low = "  (LOW CONFIDENCE)" if confidence < THRESHOLD else ""

        print(f"\nQuery:      {query}")
        print(f"Route:      {route}  ({confidence:.2f}){low}")
        print(f"Expected:   {expected}  -> {'OK' if is_correct else 'WRONG'}")
        print(f"Action:     {HANDLERS[route](query)}")

    print(f"\nAccuracy: {correct}/{len(TEST_QUERIES)}")


if __name__ == "__main__":
    main()