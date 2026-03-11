import json
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import httpx

from config import OLLAMA_EMBEDDING_URL, EMBED_MODEL, EMBEDDING_TIMEOUT, MAX_RETRIES, RETRY_BASE_WAIT, RETRY_BACKOFF_FACTOR, INDEXING_MAX_WORKERS
from rag.cache import get_cached_embedding, cache_embedding
from rag.exceptions import EmbeddingError
from rag.logger import get_logger

logger = get_logger(__name__)


def _make_embedding_request(text: str) -> list[float]:
    """Make a request to the embedding service with retry logic."""
    last_exception = None
    
    for attempt in range(MAX_RETRIES + 1):
        try:
            with httpx.Client(timeout=EMBEDDING_TIMEOUT) as client:
                response = client.post(
                    OLLAMA_EMBEDDING_URL,
                    json={
                        "model": EMBED_MODEL,
                        "prompt": text
                    }
                )
                response.raise_for_status()
                data = response.json()
                return data["embedding"]
        except (httpx.RequestError, json.JSONDecodeError, KeyError) as e:
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


def embed_text(text: str, i: Optional[int] = None, total: Optional[int] = None, use_cache: bool = True) -> list[float]:
    """Generate embeddings for the given text using Ollama."""
    if i is not None:
        logger.info(f"Embedding {i}/{total}")

    if use_cache:
        cached = get_cached_embedding(text)
        if cached is not None:
            return cached

    embedding = _make_embedding_request(text)

    if use_cache:
        cache_embedding(text, embedding)

    return embedding


def embed_texts_batch(texts: list[str], max_workers: int | None = None) -> list[list[float]]:
    """
    Generate embeddings for multiple texts in parallel using ThreadPoolExecutor.
    
    Args:
        texts: List of texts to embed
        max_workers: Maximum number of parallel workers. Defaults to INDEXING_MAX_WORKERS from config.
    
    Returns:
        List of embeddings in the same order as input texts.
    """
    if max_workers is None:
        max_workers = INDEXING_MAX_WORKERS
    
    if max_workers <= 1:
        return [embed_text(text, use_cache=True) for text in texts]
    
    logger.info(f"Processing {len(texts)} embeddings with {max_workers} workers")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        embeddings = list(executor.map(embed_text, texts))
    
    return embeddings