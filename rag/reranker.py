from sentence_transformers import CrossEncoder

model = CrossEncoder("BAAI/bge-reranker-base")


def rerank(question, chunks, top_k=3):

    pairs = [[question, chunk] for chunk in chunks]

    scores = model.predict(pairs)

    scored_chunks = list(zip(scores, chunks))

    scored_chunks.sort(reverse=True)

    return [chunk for score, chunk in scored_chunks[:top_k]]