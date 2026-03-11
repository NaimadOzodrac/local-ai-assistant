import chromadb
from rag.embeddings import embed_text
from config import TOP_K

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_collection("documents")

def search_documents(query, n_results=TOP_K):

    query_embedding = embed_text(query)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )

    return results