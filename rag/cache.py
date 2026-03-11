"""Embedding cache for storing and retrieving pre-computed embeddings."""
import hashlib
import json
from pathlib import Path
from typing import Optional

from rag.logger import get_logger

logger = get_logger(__name__)

CACHE_FILE = Path("./embeddings_cache.json")


def _get_cache() -> dict:
    """Load the embedding cache from disk."""
    if CACHE_FILE.exists():
        try:
            return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, IOError) as e:
            logger.warning(f"Failed to load cache: {e}")
            return {}
    return {}


def _save_cache(cache: dict) -> None:
    """Save the embedding cache to disk."""
    try:
        CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    except IOError as e:
        logger.warning(f"Failed to save cache: {e}")


def _get_text_hash(text: str) -> str:
    """Generate a hash key for the given text."""
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def get_cached_embedding(text: str) -> Optional[list[float]]:
    """
    Retrieve a cached embedding for the given text.
    
    Returns None if the text is not in the cache.
    """
    key = _get_text_hash(text)
    cache = _get_cache()
    
    entry = cache.get(key)
    if entry is not None:
        logger.debug(f"Cache hit for text hash: {key[:8]}...")
        return entry.get("embedding")
    
    logger.debug(f"Cache miss for text hash: {key[:8]}...")
    return None


def cache_embedding(text: str, embedding: list[float]) -> None:
    """
    Store an embedding in the cache.
    
    The cache is persisted to disk after each write.
    """
    key = _get_text_hash(text)
    cache = _get_cache()
    
    cache[key] = {
        "text": text,
        "embedding": embedding
    }
    
    _save_cache(cache)
    logger.debug(f"Cached embedding for text hash: {key[:8]}...")


def clear_cache() -> int:
    """
    Clear all cached embeddings.
    
    Returns the number of entries that were removed.
    """
    cache = _get_cache()
    count = len(cache)
    
    _save_cache({})
    logger.info(f"Cleared {count} cached embeddings")
    
    return count


def get_cache_stats() -> dict:
    """Get statistics about the embedding cache."""
    cache = _get_cache()
    
    total_size = 0
    for entry in cache.values():
        embedding = entry.get("embedding", [])
        if isinstance(embedding, list):
            total_size += len(embedding) * 8  # Approximate size in bytes
    
    return {
        "entries": len(cache),
        "approximate_size_bytes": total_size
    }
