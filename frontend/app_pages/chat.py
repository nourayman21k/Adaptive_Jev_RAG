import streamlit as st

from common import ROUTE_STYLE, BackendError, ask, probability_bars, route, route_badge

SUGGESTIONS = {
    ":green[:material/bolt:] Say hello": "Hi! What can you help me with?",
    ":blue[:material/travel_explore:] Today's tech news": "What are the top technology news stories today?",
    ":violet[:material/library_books:] Self-RAG reflection tokens": "What are reflection tokens in Self-RAG and what are they used for?",
    ":violet[:material/library_books:] RAG-Sequence vs RAG-Token": "What is the difference between the RAG-Sequence and RAG-Token models?",
}
AVATARS = {"user": ":material/person:", "assistant": ":material/hub:"}


def step_lines(result: dict) -> list[str]:
    """One line per pipeline node, in the order they ran."""
    kept = 0 if "transform_query" in result["path"] else len(result["sources"])
    lines = []
    for node in result["path"]:
        if node == "direct_answer":
            lines.append(":material/bolt: Answered from the model's own knowledge")
        elif node == "retrieve":
            lines.append(":material/search: Retrieved 6 chunks from the papers")
        elif node == "grade_documents":
            lines.append(f":material/fact_check: Graded every chunk, kept **{kept}** as relevant")
        elif node == "transform_query":
            lines.append(f":material/edit_note: None were relevant, so rewrote the query: *{result['rewritten_query']}*")
        elif node == "web_search":
            lines.append(":material/travel_explore: Searched the web")
        elif node == "generate":
            lines.append(f":material/edit: Wrote the answer from {len(result['sources'])} sources")
    return lines


def summary_label(result: dict) -> str:
    style = ROUTE_STYLE[result["final_route"]]
    return f"{style['label']} · {result['total_s']:.1f}s · {result['llm_tokens']['total_tokens']:,} tokens"


def render_footer(result: dict):
    with st.container(horizontal=True, gap="small"):
        route_badge(result["final_route"])
        st.badge(f"{result['confidence']:.0%} confident", icon=":material/percent:", color="gray")
        if result["fell_back"]:
            st.badge(f"Low confidence: {ROUTE_STYLE[result['route']]['label']} → Paper RAG",
                     icon=":material/shield:", color="orange")
        st.badge(f"{result['total_s']:.1f}s", icon=":material/timer:", color="gray")
        st.badge(f"{result['llm_tokens']['total_tokens']:,} LLM tokens", icon=":material/token:", color="gray")
        st.badge(f"router {result['router_ms']:.0f} ms", icon=":material/alt_route:", color="gray")
    if result["sources"]:
        with st.expander(f"Sources ({len(result['sources'])})", icon=":material/menu_book:"):
            for src in result["sources"]:
                if src["kind"] == "web":
                    st.markdown(f":material/public: [{src['title'] or src['source']}]({src['source']})")
                else:
                    st.markdown(f":material/description: **{src['source']}** · page {src['page']}")
                st.caption(src["snippet"] + "…")


def render_saved(message: dict):
    result = message.get("result")
    if result:
        with st.expander(summary_label(result), type="compact"):
            with st.expander(f"Routed by BERT to {ROUTE_STYLE[result['route']]['label']}", type="step"):
                probability_bars(result["probabilities"])
            with st.expander("Pipeline", type="step"):
                for line in step_lines(result):
                    st.markdown(line)
    if message.get("error"):
        st.warning(message["error"], title=message.get("error_title"), icon=":material/error:")
    else:
        st.markdown(message["content"])
    if result:
        render_footer(result)


def answer(prompt: str):
    with st.chat_message("assistant", avatar=AVATARS["assistant"]):
        message = {"role": "assistant", "content": "", "result": None}
        with st.status(":shimmer[Routing your question]", type="compact") as status:
            with st.status("Routing with the fine-tuned BERT classifier", type="step") as routing:
                try:
                    decision = route(prompt)
                except BackendError as e:
                    routing.update(label="Couldn't reach the router", state="error")
                    status.update(label="Backend unavailable", state="error")
                    message.update(error=e.detail, error_title="Backend unavailable")
                    st.session_state.messages.append(message)
                    st.warning(e.detail, title="Backend unavailable", icon=":material/cloud_off:")
                    return
                probability_bars(decision["probabilities"])
                target = ROUTE_STYLE[decision["final_route"]]
                routing.update(label=f"Routed to {ROUTE_STYLE[decision['route']]['label']} "
                                     f"({decision['confidence']:.0%} confident, {decision['latency_ms']:.0f} ms)",
                               state="complete")

            status.update(label=f":shimmer[{target['label']}: working]")
            with st.status(f"{target['label']}: {target['about']}", type="step") as pipeline:
                try:
                    result = ask(prompt)
                except BackendError as e:
                    title = {429: "Groq rate limit reached", 503: "No LLM configured"}.get(e.status, "Request failed")
                    pipeline.update(label=title, state="error")
                    status.update(label=title, state="error")
                    message.update(error=e.detail, error_title=title)
                    st.session_state.messages.append(message)
                    st.warning(e.detail, title=title, icon=":material/error:")
                    return
                for line in step_lines(result):
                    st.markdown(line)
                pipeline.update(label=f"{target['label']}: done in {result['total_s']:.1f}s", state="complete")
            status.update(label=summary_label(result), state="complete")

        st.markdown(result["answer"])
        render_footer(result)
        message.update(content=result["answer"], result=result)
        st.session_state.messages.append(message)


for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar=AVATARS[msg["role"]]):
        if msg["role"] == "user":
            st.markdown(msg["content"])
        else:
            render_saved(msg)

prompt = st.chat_input("Ask about RAG research, today's news, or anything else", submit_mode="disable")

if not st.session_state.messages:
    intro = st.empty()
    with intro.container():
        st.markdown("Ask anything. The router decides whether the question needs **the model alone**, "
                    "**a web search**, or **the 10 RAG papers** in the knowledge base.")
        selected = st.pills("Try asking", list(SUGGESTIONS), label_visibility="collapsed")
    if selected:
        prompt = SUGGESTIONS[selected]
    if prompt:
        intro.empty()

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=AVATARS["user"]):
        st.markdown(prompt)
    answer(prompt)
