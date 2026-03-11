from llm.client import ask_llm
from llm.client import stream_llm
from rag.search import search_documents
from rag.reranker import rerank
from config import MAX_CONTEXT_CHARS

def build_context(docs):
    parts = []

    for i, doc in enumerate(docs):
        parts.append(f"[Source {i+1}]\n{doc}")
        print(f"Chunk {i+1}:")
        print(doc[:200])
        print("------")

    return "\n\n".join(parts)

def rerank_chunks(question, chunks):
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


def ask_with_context(question):

    results = search_documents(question)

    parents = []

    for meta in results["metadatas"][0]:
        parents.append(meta["parent_text"])

    # eliminar duplicados sin romper orden
    parents = list(dict.fromkeys(parents))
    print(f"Retrieved parents: {len(parents)}")
    print("-------------------")
    print("-------------------")
    # rerank
    best_chunks = rerank(question, parents, top_k=3)
    print(f"Using top chunks: {len(best_chunks)}")
    print("-------------------")
    print("-------------------")

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