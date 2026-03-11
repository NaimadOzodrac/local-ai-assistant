from typing import Any

import chromadb

from rag.embeddings import embed_text
from rag.exceptions import CollectionEmptyError, SearchError
from config import TOP_K

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("documents")


def search_documents(query: str, n_results: int = TOP_K) -> dict[str, Any]:
    """Search for documents most similar to the query using vector similarity."""
    if not query or not query.strip():
        raise ValueError("Query cannot be empty")

    if n_results <= 0:
        raise ValueError("n_results must be positive")

    try:
        count = collection.count()
        if count == 0:
            raise CollectionEmptyError("No documents indexed. Run index_documents first.")
    except CollectionEmptyError:
        raise
    except Exception as e:
        raise SearchError(f"Failed to access collection: {e}") from e

    try:
        query_embedding = embed_text(query)

        results = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results
        )

        return results
    except CollectionEmptyError:
        raise
    except Exception as e:
        raise SearchError(f"Search failed: {e}") from e