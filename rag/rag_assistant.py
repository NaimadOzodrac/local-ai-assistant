from llm.client import ask_llm
from llm.client import stream_llm
from rag.search import search_documents
from rag.reranker import rerank
from rag.logger import get_logger
from config import MAX_CONTEXT_CHARS

logger = get_logger(__name__)


def build_context(docs: list[str]) -> str:
    """Build context string from document chunks with source references."""
    parts = []

    for i, doc in enumerate(docs):
        parts.append(f"[Source {i+1}]\n{doc}")
        logger.debug(f"Chunk {i+1}: {doc[:200]}")

    return "\n\n".join(parts)


def rerank_chunks(question: str, chunks: list[str]) -> list[str]:
    """Re-rank chunks based on relevance using LLM scoring."""
    scored = []

    for chunk in chunks:
        prompt = f"""
    Rate how relevant this text is to the question.

    Question:
    {question}

    Text:
    {chunk}

    Answer with only a number between 1 and 10.
    """

        score = ask_llm(prompt)
        scored.append((score, chunk))

    scored.sort(reverse=True)

    return [chunk for score, chunk in scored]


def ask_with_context(question: str) -> list[str]:
    """Answer a question using RAG with relevant document context."""
    results = search_documents(question)

    parents = []

    for meta in results["metadatas"][0]:
        parents.append(meta["parent_text"])

    parents = list(dict.fromkeys(parents))
    logger.info(f"Retrieved {len(parents)} parent chunks")
    best_chunks = rerank(question, parents, top_k=3)
    logger.info(f"Using top {len(best_chunks)} chunks after reranking")

    context = build_context(best_chunks)
    context = context[:MAX_CONTEXT_CHARS]

    prompt = f"""
    You are an assistant that answers questions using the provided context.

    Rules:
    - Answer ONLY using the information from the context.
    - If the answer is not in the context, say you don't know.
    - Be concise.

    Context:
    {context}

    Question:
    {question}

    Answer using the context.
    """

    stream_llm(prompt)

    return results["ids"][0]