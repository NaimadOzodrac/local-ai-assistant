import json
from typing import Optional

import requests

from config import OLLAMA_EMBEDDING_URL, EMBED_MODEL, EMBEDDING_TIMEOUT
from rag.exceptions import EmbeddingError
from rag.logger import get_logger

logger = get_logger(__name__)


def embed_text(text: str, i: Optional[int] = None, total: Optional[int] = None) -> list[float]:
    """Generate embeddings for the given text using Ollama."""
    if i is not None:
        logger.info(f"Embedding {i}/{total}")

    try:
        response = requests.post(
            OLLAMA_EMBEDDING_URL,
            json={
                "model": EMBED_MODEL,
                "prompt": text
            },
            timeout=EMBEDDING_TIMEOUT
        )
        response.raise_for_status()
    except requests.RequestException as e:
        raise EmbeddingError(f"Failed to connect to embedding service: {e}") from e

    try:
        data = response.json()
        return data["embedding"]
    except (json.JSONDecodeError, KeyError) as e:
        raise EmbeddingError(f"Invalid response from embedding service: {e}") from e