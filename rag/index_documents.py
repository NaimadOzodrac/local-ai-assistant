import os
from typing import Any

import chromadb

from rag.chunker import chunk_text
from rag.pdf_loader import load_pdf
from rag.embeddings import embed_text
from config import PARENT_CHUNK_SIZE, PARENT_OVERLAP, CHILD_CHUNK_SIZE, CHILD_OVERLAP

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("documents")

docs_path = "documents"

all_chunks: list[str] = []
metadatas: list[dict[str, Any]] = []
embeddings: list[list[float]] = []
ids: list[str] = []

chunk_counter = 0

for file in os.listdir(docs_path):

    path = os.path.join(docs_path, file)

    if file.endswith(".txt"):

        with open(path, encoding="utf-8") as f:
            text = f.read()

    elif file.endswith(".pdf"):

        text = load_pdf(path)

    else:
        continue

    parents = chunk_text(
        text,
        chunk_size=PARENT_CHUNK_SIZE,
        overlap=PARENT_OVERLAP
    )

    for parent_id, parent in enumerate(parents):

        children = chunk_text(
            parent,
            chunk_size=CHILD_CHUNK_SIZE,
            overlap=CHILD_OVERLAP
        )

        for child_id, child in enumerate(children):

            all_chunks.append(child)

            ids.append(f"chunk_{chunk_counter}")

            metadatas.append({
                "file": file,
                "parent_id": parent_id,
                "parent_text": parent
            })

            chunk_counter += 1


for i, chunk in enumerate(all_chunks):
    embeddings.append(embed_text(chunk, i+1, len(all_chunks)))


collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

print(f"Indexed {len(all_chunks)} chunks")