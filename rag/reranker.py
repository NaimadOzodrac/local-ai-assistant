from typing import Any

from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")


def rerank(question: str, chunks: list[str], top_k: int = 3) -> list[str]:
    """Re-rank chunks based on relevance to the question."""
    pairs = [[question, chunk] for chunk in chunks]

    scores = model.predict(pairs)

    scored_chunks: list[tuple[Any, str]] = list(zip(scores, chunks))

    scored_chunks.sort(reverse=True)

    return [chunk for score, chunk in scored_chunks[:top_k]]