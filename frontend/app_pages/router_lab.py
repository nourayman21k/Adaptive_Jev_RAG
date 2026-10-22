import pandas as pd
import streamlit as st

from common import ROUTE_STYLE, BackendError, probability_bars, route, route_badge

EXAMPLES = """Good morning!
What is 12 times 7?
What is the price of gold today?
Who won last night's football match?
How does RAPTOR build its tree of summaries?
What is HyDE?
Compare Self-RAG and CRAG."""

st.caption("Try the fine-tuned BERT router on its own. No LLM is called, so this works without an API key.")

query = st.text_input("Question", placeholder="Type a question to see where it would be routed",
                      key="lab_query")
if query:
    try:
        decision = route(query)
    except BackendError as e:
        st.warning(e.detail, title="Backend unavailable", icon=":material/cloud_off:")
    else:
        with st.container(border=True):
            with st.container(horizontal=True, vertical_alignment="center"):
                route_badge(decision["final_route"])
                if decision["final_route"] != decision["route"]:
                    st.badge(f"Fell back from {ROUTE_STYLE[decision['route']]['label']}: confidence under 70%",
                             icon=":material/shield:", color="orange")
            route_col, conf_col, latency_col, tokens_col = st.columns(4)
            route_col.metric("Route", ROUTE_STYLE[decision["route"]]["label"], border=True)
            conf_col.metric("Confidence", f"{decision['confidence']:.1%}", border=True)
            latency_col.metric("Router latency", f"{decision['latency_ms']:.1f} ms", border=True)
            tokens_col.metric("Tokens read", decision["input_tokens"], border=True)
            probability_bars(decision["probabilities"])

st.subheader("Route many at once", icon=":material/playlist_play:")
with st.form("batch"):
    text = st.text_area("One question per line", value=EXAMPLES, height=200)
    submitted = st.form_submit_button("Route all", icon=":material/alt_route:", type="primary")

if submitted:
    rows = []
    try:
        for line in [l.strip() for l in text.splitlines() if l.strip()]:
            d = route(line)
            rows.append({
                "question": line,
                "route": f"{ROUTE_STYLE[d['final_route']]['label']}",
                "confidence": d["confidence"],
                "direct": d["probabilities"]["direct_answer"],
                "web": d["probabilities"]["web_search"],
                "rag": d["probabilities"]["rag_retrieval"],
                "latency_ms": d["latency_ms"],
            })
    except BackendError as e:
        st.warning(e.detail, title="Backend unavailable", icon=":material/cloud_off:")
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(
            df,
            hide_index=True,
            column_config={
                "question": st.column_config.TextColumn("Question", width="large"),
                "route": st.column_config.TextColumn("Route"),
                "confidence": st.column_config.ProgressColumn("Confidence", format="percent", min_value=0, max_value=1),
                "direct": st.column_config.ProgressColumn("Direct", format="percent", min_value=0, max_value=1),
                "web": st.column_config.ProgressColumn("Web", format="percent", min_value=0, max_value=1),
                "rag": st.column_config.ProgressColumn("Paper RAG", format="percent", min_value=0, max_value=1),
                "latency_ms": st.column_config.NumberColumn("Latency", format="%.1f ms"),
            },
            alt="Route, confidence and per-route probabilities for each question",
        )
        st.caption(f"{len(df)} questions routed · mean router latency {df['latency_ms'].mean():.1f} ms")
