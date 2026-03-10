import requests

OLLAMA_URL = "http://localhost:11434/api/embeddings"
MODEL = "nomic-embed-text"

def embed_text(text, i=None, total=None):

    if i is not None:
        print(f"Embedding {i}/{total}")

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "prompt": text
        }
    )

    return response.json()["embedding"]