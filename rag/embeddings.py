import json
import time
from typing import Optional

import requests

from config import OLLAMA_EMBEDDING_URL, EMBED_MODEL, EMBEDDING_TIMEOUT, MAX_RETRIES, RETRY_BASE_WAIT, RETRY_BACKOFF_FACTOR
from rag.exceptions import EmbeddingError
from rag.logger import get_logger

logger = get_logger(__name__)


def _make_embedding_request(text: str) -> list[float]:
    """Make a request to the embedding service with retry logic."""
    last_exception = None
    
    for attempt in range(MAX_RETRIES + 1):
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
            data = response.json()
            return data["embedding"]
        except (requests.RequestException, json.JSONDecodeError, KeyError) as e:
            last_exception = e
            
            if attempt == MAX_RETRIES:
                logger.error(f"Embedding request failed after {MAX_RETRIES} retries")
                raise EmbeddingError(f"Failed to get embedding after {MAX_RETRIES} retries: {e}") from e
            
            wait_time = RETRY_BASE_WAIT * (RETRY_BACKOFF_FACTOR ** attempt)
            logger.warning(
                f"Embedding attempt {attempt + 1}/{MAX_RETRIES + 1} failed: {e}. "
                f"Retrying in {wait_time:.1f}s..."
            )
            time.sleep(wait_time)
    
    raise EmbeddingError(f"Failed to get embedding: {last_exception}") from last_exception


def embed_text(text: str, i: Optional[int] = None, total: Optional[int] = None) -> list[float]:
    """Generate embeddings for the given text using Ollama."""
    if i is not None:
        logger.info(f"Embedding {i}/{total}")

    return _make_embedding_request(text)