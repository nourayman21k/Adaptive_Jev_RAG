"""The Chroma index over the papers in pdf/, persisted to disk so it is built once, not on every start.

    python -m adaptive_jev_rag.knowledge_base    # (re)build the index
"""
import time

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import CHROMA_DIR, CHUNK_OVERLAP, CHUNK_SIZE, COLLECTION, EMBEDDING_MODEL, PDF_DIR


def load_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL, model_kwargs={"device": "cpu"})


def build(embeddings: HuggingFaceEmbeddings) -> Chroma:
    pages = []
    for pdf in sorted(PDF_DIR.glob("*.pdf")):
        pages.extend(PyPDFLoader(str(pdf)).load())
    if not pages:
        raise FileNotFoundError(f"No PDFs found in {PDF_DIR}")
    chunks = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP).split_documents(pages)
    # start from an empty collection so a rebuild doesn't store every chunk twice
    Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(CHROMA_DIR)).delete_collection()
    return Chroma.from_documents(chunks, embeddings, collection_name=COLLECTION, persist_directory=str(CHROMA_DIR))


def load(embeddings: HuggingFaceEmbeddings) -> Chroma:
    """Open the persisted index, building it first if it is missing or empty."""
    store = Chroma(collection_name=COLLECTION, embedding_function=embeddings, persist_directory=str(CHROMA_DIR))
    if store._collection.count() == 0:
        store = build(embeddings)
    return store


if __name__ == "__main__":
    start = time.perf_counter()
    store = build(load_embeddings())
    print(f"indexed {store._collection.count()} chunks from {PDF_DIR} into {CHROMA_DIR} "
          f"in {time.perf_counter() - start:.1f}s")
