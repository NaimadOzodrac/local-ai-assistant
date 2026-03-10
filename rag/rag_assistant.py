from llm.client import ask_llm
from llm.client import stream_llm
from rag.search import search_documents


def ask_with_context(question):

    results = search_documents(question)

    docs = results["documents"][0][:4]
    ids = results["ids"][0]

    context = "\n\n".join(docs)[:1200]

    prompt = f"""
You are an assistant answering questions using provided documents.

Answer in the SAME language as the question.

Context:
{context}

Question:
{question}

Answer using the context.
"""

    stream_llm(prompt)

    return ids