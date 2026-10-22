"""Headless smoke tests for the Streamlit frontend. They need the backend running (BACKEND_URL, default
http://localhost:8000); the chat test also makes one Groq call.

    cd frontend && uv run pytest tests -q
"""
import pytest
import requests
from streamlit.testing.v1 import AppTest

from common import BACKEND_URL

APP = "../streamlit_app.py"

try:
    requests.get(f"{BACKEND_URL}/health", timeout=5).raise_for_status()
    BACKEND_UP = True
except requests.RequestException:
    BACKEND_UP = False

needs_backend = pytest.mark.skipif(not BACKEND_UP, reason=f"backend not reachable at {BACKEND_URL}")


def test_benchmarks_page_renders():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.switch_page("app_pages/benchmarks.py").run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert "Faithfulness" in labels and "Context recall" in labels


@needs_backend
def test_sidebar_shows_backend_online():
    at = AppTest.from_file(APP, default_timeout=30).run()
    assert not at.exception
    # st.badge is rendered as a markdown badge directive
    assert any("Backend online" in m.value for m in at.sidebar.markdown)


@needs_backend
def test_router_lab_routes_a_question():
    at = AppTest.from_file(APP, default_timeout=30).run()
    at.switch_page("app_pages/router_lab.py").run()
    at.text_input(key="lab_query").set_value("What is the price of gold today?").run()
    assert not at.exception
    assert at.metric[0].value == "Web search"
    at.form_submit_button[0].click().run()
    assert not at.exception
    assert len(at.dataframe[0].value) == 7


@needs_backend
def test_chat_answers_a_greeting():
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.chat_input[0].set_value("Hi! What can you help me with?").run()
    assert not at.exception
    messages = at.session_state["messages"]
    assert [m["role"] for m in messages] == ["user", "assistant"]
    reply = messages[1]
    assert reply.get("error") is None, reply.get("error")
    assert reply["result"]["final_route"] == "direct_answer"
    assert reply["content"]
