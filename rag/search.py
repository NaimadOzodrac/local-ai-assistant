from typing import Any

import chromadb

from rag.embeddings import embed_text
from config import TOP_K

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("documents")


def search_documents(query: str, n_results: int = TOP_K) -> dict[str, Any]:
    """Search for documents most similar to the query using vector similarity."""
    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results