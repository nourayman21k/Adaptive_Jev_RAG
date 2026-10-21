def main() -> None:
    """`uv run adaptive-jev-rag` starts the API on port 8000."""
    import uvicorn

    uvicorn.run("adaptive_jev_rag.api:app", host="0.0.0.0", port=8000)
