import os
from typing import Any

import chromadb

from rag.chunker import chunk_text
from rag.pdf_loader import load_pdf
from rag.embeddings import embed_texts_batch
from rag.exceptions import DocumentLoadError
from rag.logger import get_logger
from config import PARENT_CHUNK_SIZE, PARENT_OVERLAP, CHILD_CHUNK_SIZE, CHILD_OVERLAP

logger = get_logger(__name__)

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection("documents")

docs_path = "documents"

if not os.path.exists(docs_path):
    raise DocumentLoadError(f"Documents directory does not exist: {docs_path}")

if not os.path.isdir(docs_path):
    raise DocumentLoadError(f"Documents path is not a directory: {docs_path}")

all_chunks: list[str] = []
metadatas: list[dict[str, Any]] = []
embeddings: list[list[float]] = []
ids: list[str] = []

chunk_counter = 0

for file in os.listdir(docs_path):

    path = os.path.join(docs_path, file)

    if not os.path.isfile(path):
        continue

    try:
        if file.endswith(".txt"):

            with open(path, encoding="utf-8") as f:
                text = f.read()

        elif file.endswith(".pdf"):

            text = load_pdf(path)

        else:
            continue
    except Exception as e:
        logger.warning(f"Failed to load {file}: {e}")
        continue

    if not text or not text.strip():
        logger.warning(f"Empty document {file}, skipping")
        continue

    try:
        parents = chunk_text(
            text,
            chunk_size=PARENT_CHUNK_SIZE,
            overlap=PARENT_OVERLAP
        )
    except ValueError as e:
        logger.warning(f"Failed to chunk {file}: {e}")
        continue

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


if not all_chunks:
    raise DocumentLoadError("No documents were successfully indexed")

logger.info(f"Generating embeddings for {len(all_chunks)} chunks...")
embeddings = embed_texts_batch(all_chunks)


collection.add(
    documents=all_chunks,
    embeddings=embeddings,
    metadatas=metadatas,
    ids=ids
)

logger.info(f"Indexed {len(all_chunks)} chunks")