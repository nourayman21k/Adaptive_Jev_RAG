import streamlit as st

from common import BACKEND_URL, health

st.set_page_config(page_title="Adaptive RAG", page_icon=":material/hub:", layout="wide")

if "messages" not in st.session_state:
    st.session_state.messages = []

page = st.navigation(
    [
        st.Page("app_pages/chat.py", title="Chat", icon=":material/forum:"),
        st.Page("app_pages/router_lab.py", title="Router lab", icon=":material/alt_route:"),
        st.Page("app_pages/benchmarks.py", title="Benchmarks", icon=":material/monitoring:"),
    ],
    position="top",
)

with st.sidebar:
    st.header("Adaptive RAG", icon=":material/hub:")
    st.caption("A fine-tuned BERT router sends each question to the cheapest route that can answer it.")
    status = health()
    if status:
        st.badge("Backend online", icon=":material/check_circle:", color="green")
        metrics = status.get("router_metrics") or {}
        if metrics.get("test_accuracy") is not None:
            st.caption(f":material/alt_route: Router test accuracy **{metrics['test_accuracy']:.1%}** "
                       f"on {metrics['test_examples']} held-out queries")
        st.caption(f":material/database: **{status['indexed_chunks']:,}** chunks indexed from 10 papers")
        if status.get("llm"):
            st.caption(f":material/neurology: Answers by `{status['llm']}`")
        else:
            st.badge("No GROQ_API_KEY: routing only", icon=":material/key_off:", color="orange")
        st.caption(f":material/tune: Below {status['threshold']:.0%} confidence, questions go to Paper RAG")
    else:
        st.badge("Backend offline", icon=":material/cloud_off:", color="red")
        st.caption(f"Can't reach `{BACKEND_URL}`")
    if st.button("New chat", icon=":material/add_comment:", width="stretch"):
        st.session_state.messages = []
        st.rerun()

st.title(page.title, icon=page.icon)
page.run()
