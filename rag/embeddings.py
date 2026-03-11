from typing import Optional

import requests

from config import OLLAMA_EMBEDDING_URL, EMBED_MODEL


def embed_text(text: str, i: Optional[int] = None, total: Optional[int] = None) -> list[float]:
    """Generate embeddings for the given text using Ollama."""
    if i is not None:
        print(f"Embedding {i}/{total}")

    response = requests.post(
        OLLAMA_EMBEDDING_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )

    return response.json()["embedding"]