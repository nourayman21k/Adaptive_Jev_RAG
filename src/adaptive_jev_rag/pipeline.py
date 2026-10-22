"""The adaptive RAG pipeline from adaptive_bert.ipynb, packaged for the API.

The fine-tuned BERT router picks a route; low-confidence decisions fall back to rag_retrieval. Then:
- direct_answer: one LLM call with SYSTEM_PROMPT
- web_search:    DuckDuckGo, then the generator
- rag_retrieval: the corrective RAG graph (retrieve -> grade -> generate, or rewrite -> web search -> generate)
"""
import os
import time
from typing import List, Literal

from ddgs import DDGS
from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from .config import LLM_MAX_RETRIES, LLM_MODEL, RETRIEVER_K, WEB_RESULTS
from .router import BertRouter

GRADER_PROMPT = SystemMessage(content="""You are a relevance grader in an Adaptive RAG pipeline.
A router has already judged the user's QUESTION complex enough to need the internal knowledge base of research papers.
You will receive that QUESTION and one retrieved DOCUMENT chunk from the knowledge base.

Decide whether the document contains information that helps answer the question.
Your grades decide the next step: relevant chunks are passed to the answer generator,
and if too few chunks are relevant, the pipeline adapts (rewrites the query or falls back to web search).

Grade "Yes" if the chunk:
- directly answers the question, or part of it, or
- contains facts, definitions, methods, or results that are clearly about the question's topic.

Grade "No" if the chunk:
- is only loosely related (shares a keyword but not the meaning),
- is a reference list, author list, acknowledgements, table of contents, or other boilerplate,
- is about a different concept than the one asked about.

Be strict: the goal is to filter out irrelevant chunks, not to answer the question.
Judge only the given chunk; do not use outside knowledge to fill gaps.""")

REWRITE_PROMPT = SystemMessage(content="""You are a query rewriter in an Adaptive RAG pipeline.
A router sent the user's QUESTION to the internal knowledge base of research papers, but too few retrieved chunks were relevant.
Your rewritten query will be used for the next attempt: either retrieving again from the knowledge base or searching the web.

Rewrite the question into a single, concise search query that is most likely to return relevant, factual results.

Guidelines:
- Keep the original intent and all key entities, names, acronyms, and technical terms.
- Expand ambiguous acronyms when it helps (e.g. "RAG" -> "retrieval-augmented generation").
- Use the vocabulary a research paper on the topic would use (method names, task names, standard terminology).
- Remove filler words, greetings, and conversational phrasing.
- Prefer keywords and specific terms over full sentences; about 5-15 words.
- Do not answer the question, add facts, or introduce assumptions not present in it.
- Output only the rewritten query, with no explanation or quotes.""")

GENERATION_PROMPT = SystemMessage(content="""You are the answer generator in an Adaptive RAG pipeline about retrieval-augmented generation and LLM research.
A router judged the user's QUESTION too complex to answer from model knowledge alone, so it was sent to retrieval.
You will receive the QUESTION and a set of CONTEXT passages. Each passage is either a chunk from a research paper in the internal knowledge base or a web search result, and each one is labeled with its source.

Answer the question using only the information in the context.

Guidelines:
- Base every claim on the context; do not add facts from your own knowledge.
- If the context only partly answers the question, answer that part and clearly say what is missing.
- If the context does not contain the answer, say "I could not find this in the provided sources." and do not guess.
- Cite the source of each key claim in square brackets, e.g. [2005.11401v4.pdf, p. 3] or [https://example.com].
- If sources disagree, point out the disagreement instead of picking one silently.
- Since the question was routed as complex, give a well-structured answer: open with a one-sentence answer (no heading), then the supporting analysis (comparisons, steps, trade-offs) as the question requires. Use short paragraphs or bullet points when that is easier to read.
- Explain technical terms briefly when they are central to the answer.""")

SYSTEM_PROMPT = SystemMessage(content="""You are a helpful, knowledgeable assistant in an Adaptive RAG system specialised in retrieval-augmented generation and LLM research.
A router has judged the user's message simple enough to answer directly from your own knowledge, without retrieving documents or searching the web.

Guidelines:
- Answer clearly and concisely: give the direct answer first, then a short explanation only if it helps.
- For greetings and small talk, reply briefly and naturally, and offer help.
- For definitions and concepts, give an accurate, plain-language explanation; add a short example when it makes the idea clearer.
- For math or simple reasoning, show the key steps and the final result.
- Reply in the same language the user writes in.
- Be honest about uncertainty: if you are not sure, say so instead of guessing.
- Do not invent facts, numbers, paper titles, citations, or links.
- If the question actually needs up-to-date information (news, prices, recent releases) or detailed analysis of specific papers, say that you cannot answer it reliably from memory and that it needs a web search or the document knowledge base.""")


class GradeDocument(BaseModel):
    """Relevance grade for one chunk retrieved from the research-paper knowledge base in the Adaptive RAG pipeline."""
    relevant: Literal["Yes", "No"] = Field(
        description="'Yes' if the chunk contains facts, definitions, methods, or results that help answer the question; "
                    "'No' if it is only loosely related, about a different concept, or boilerplate (references, author list, acknowledgements)."
    )


class RewrittenQuery(BaseModel):
    """Web-search-optimized version of the user's question."""
    query: str = Field(description="The rewritten web search query")


class RAGState(BaseModel):
    question: str = ""
    documents: List[Document] = Field(default_factory=list)
    feedback: str = ""
    generation: str = ""


class LLMUnavailable(RuntimeError):
    """Raised when no Groq API key is configured, so only routing is possible."""


def make_llm() -> ChatGroq | None:
    if not os.getenv("GROQ_API_KEY"):
        return None
    return ChatGroq(model=LLM_MODEL, max_retries=LLM_MAX_RETRIES)


def web_search(query: str) -> list[Document]:
    try:
        results = DDGS().text(query, max_results=WEB_RESULTS)
    except Exception as e:  # a failed search should degrade the answer, not crash the request
        print(f"Web search failed: {e}")
        return []
    return [Document(page_content=r["body"], metadata={"source": r["href"], "title": r["title"]}) for r in results]


def format_docs(docs: List[Document]) -> str:
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        if not source.startswith("http"):  # web results keep their full URL so they can be cited
            source = os.path.basename(source)
        page = doc.metadata.get("page")
        label = f"{source}, p. {page + 1}" if page is not None else source
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n---\n\n".join(parts)


def source_info(doc: Document) -> dict:
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page")
    is_web = source.startswith("http")
    return {
        "source": source if is_web else os.path.basename(source),
        "kind": "web" if is_web else "paper",
        "page": page + 1 if page is not None else None,
        "title": doc.metadata.get("title"),
        "snippet": " ".join(doc.page_content.split())[:300],
    }


class AdaptivePipeline:
    def __init__(self, router: BertRouter, vector_store, llm: ChatGroq | None):
        self.router = router
        self.retriever = vector_store.as_retriever(search_kwargs={"k": RETRIEVER_K})
        self.llm = llm
        if llm is not None:
            # gpt-oss occasionally skips the tool call, so structured calls retry
            self.grader = llm.with_structured_output(GradeDocument).with_retry(stop_after_attempt=3)
            self.rewriter = llm.with_structured_output(RewrittenQuery).with_retry(stop_after_attempt=3)
            self.rag_graph = self._build_rag_graph()

    # ---------------------------------------------------------------- corrective RAG graph
    def _retrieve(self, state: RAGState) -> dict:
        return {"documents": self.retriever.invoke(state.question)}

    def _grade(self, state: RAGState) -> dict:
        grades = self.grader.batch(
            [[GRADER_PROMPT, HumanMessage(content=f"QUESTION:\n{state.question}\n\nDOCUMENT:\n{doc.page_content}")]
             for doc in state.documents],
            config={"max_concurrency": 3},  # stays under Groq's free-tier tokens-per-minute limit
        )
        relevant = [doc for doc, grade in zip(state.documents, grades) if grade.relevant == "Yes"]
        return {"documents": relevant, "feedback": "Yes" if relevant else "No"}

    def _rewrite(self, state: RAGState) -> dict:
        result = self.rewriter.invoke([REWRITE_PROMPT, HumanMessage(content=f"QUESTION:\n{state.question}")])
        return {"question": result.query}

    def _web_search(self, state: RAGState) -> dict:
        return {"documents": state.documents + web_search(state.question)}

    def _generate(self, state: RAGState, config: RunnableConfig | None = None) -> dict:
        response = self.llm.invoke(
            [GENERATION_PROMPT, HumanMessage(content=f"QUESTION:\n{state.question}\n\nCONTEXT:\n{format_docs(state.documents)}")],
            config=config,
        )
        return {"generation": response.content}

    @staticmethod
    def _after_grading(state: RAGState) -> Literal["transform_query", "generate"]:
        return "generate" if state.feedback == "Yes" else "transform_query"

    def _build_rag_graph(self):
        graph = StateGraph(RAGState)
        graph.add_node("retrieve", self._retrieve)
        graph.add_node("grade_documents", self._grade)
        graph.add_node("transform_query", self._rewrite)
        graph.add_node("web_search", self._web_search)
        graph.add_node("generate", self._generate)
        graph.set_entry_point("retrieve")
        graph.add_edge("retrieve", "grade_documents")
        graph.add_conditional_edges("grade_documents", self._after_grading)
        graph.add_edge("transform_query", "web_search")
        graph.add_edge("web_search", "generate")
        graph.add_edge("generate", END)
        return graph.compile()

    # ---------------------------------------------------------------- entry point
    def ask(self, query: str) -> dict:
        start = time.perf_counter()
        decision = self.router.route(query)
        router_ms = (time.perf_counter() - start) * 1000
        route = decision.final_route
        if self.llm is None:
            raise LLMUnavailable("GROQ_API_KEY is not set, so the pipeline can route queries but not answer them.")

        usage = UsageMetadataCallbackHandler()
        config: RunnableConfig = {"callbacks": [usage]}
        rewritten = None
        if route == "direct_answer":
            response = self.llm.invoke([SYSTEM_PROMPT, HumanMessage(content=query)], config=config)
            answer, docs, path = response.content, [], ["direct_answer"]
        elif route == "web_search":
            docs = web_search(query)
            answer = self._generate(RAGState(question=query, documents=docs), config)["generation"]
            path = ["web_search", "generate"]
        else:
            answer, docs, path = "", [], []
            for update in self.rag_graph.stream({"question": query}, stream_mode="updates", config=config):
                for node, values in update.items():
                    path.append(node)
                    values = values or {}
                    docs = values.get("documents", docs)
                    answer = values.get("generation", answer)
                    if node == "transform_query":
                        rewritten = values.get("question")

        tokens = {k: sum(u.get(k, 0) for u in usage.usage_metadata.values())
                  for k in ("input_tokens", "output_tokens", "total_tokens")}
        return {
            "query": query,
            "route": decision.route,
            "confidence": decision.confidence,
            "probabilities": decision.probabilities,
            "final_route": route,
            "fell_back": route != decision.route,
            "answer": answer,
            "path": path,
            "rewritten_query": rewritten,
            "sources": [source_info(d) for d in docs],
            "router_ms": router_ms,
            "router_input_tokens": decision.input_tokens,
            "total_s": time.perf_counter() - start,
            "llm_tokens": tokens,
        }
