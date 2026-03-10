import chromadb
from rag.embeddings import embed_text

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("documents")

def search_documents(query, n_results=5):

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results