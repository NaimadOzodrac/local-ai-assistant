import requests
from config import OLLAMA_EMBEDDING_URL, EMBED_MODEL

def embed_text(text, i=None, total=None):

    if i is not None:
        print(f"Embedding {i}/{total}")

    response = requests.post(
        OLLAMA_EMBEDDING_URL,
        json={
            "model": EMBED_MODEL,
            "prompt": text
        }
    )

    return response.json()["embedding"]