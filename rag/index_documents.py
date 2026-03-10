import os
import chromadb
from rag.chunker import chunk_text
from rag.pdf_loader import load_pdf
from rag.embeddings import embed_text

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("documents")

docs_path = "documents"

all_chunks = []

for file in os.listdir(docs_path):

    path = os.path.join(docs_path, file)

    if file.endswith(".txt"):

        with open(path, encoding="utf-8") as f:
            text = f.read()

    elif file.endswith(".pdf"):

        text = load_pdf(path)

    else:
        continue

    chunks = chunk_text(text)

    all_chunks.extend(chunks)

ids = [f"chunk_{i}" for i in range(len(all_chunks))]

embeddings = []

for i, chunk in enumerate(all_chunks):
    embeddings.append(embed_text(chunk, i+1, len(all_chunks)))

collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    ids=ids
)

print(f"Indexed {len(all_chunks)} chunks")