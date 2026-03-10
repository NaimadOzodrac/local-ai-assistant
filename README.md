# Local AI Document Assistant

A local Retrieval-Augmented Generation (RAG) assistant that can answer questions about your documents using a local language model.

The system runs entirely on your machine using **Ollama**, **Chroma**, and **Python**, allowing you to query PDFs and text files without sending data to external APIs.

---

## Features

* Run a **local LLM** using Ollama
* Index **PDF and text documents**
* Semantic search using **vector embeddings**
* Retrieval-Augmented Generation (RAG)
* **Interactive terminal chat**
* **Streaming responses** (token-by-token output)
* Shows **source chunks** used to generate the answer

---

## Architecture

The assistant uses a standard RAG pipeline:

```
Documents (PDF / TXT)
        ↓
Text extraction
        ↓
Chunking
        ↓
Embeddings (nomic-embed-text)
        ↓
Vector database (Chroma)
        ↓
Semantic search
        ↓
Local LLM (Mistral via Ollama)
        ↓
Answer with sources
```

---

## Tech Stack

* Python
* Ollama
* Mistral 7B
* nomic-embed-text embeddings
* Chroma vector database
* PyPDF

---

## Installation

Clone the repository:

```
git clone https://github.com/YOUR_USERNAME/local-ai-assistant.git
cd local-ai-assistant
```

Install dependencies:

```
pip install -r requirements.txt
```

Install required Ollama models:

```
ollama pull mistral
ollama pull nomic-embed-text
```

---

## Index Documents

Place your documents in the `documents/` folder.

Supported formats:

* `.txt`
* `.pdf`

Then run:

```
python -m rag.index_documents
```

This will:

* extract text
* split it into chunks
* generate embeddings
* store them in Chroma

---

## Run the Assistant

Start the interactive chat:

```
python chat.py
```

Example:

```
Local AI Assistant

You: Where did tango originate?

AI:
Tango originated in Buenos Aires and Montevideo in the late 19th century.

Sources:
['chunk_12', 'chunk_44']
```

Responses are streamed token by token.

---

## Project Structure

```
local-ai-assistant

documents/           # Input documents (PDF / TXT)

llm/
  client.py          # LLM client (Ollama + streaming)

rag/
  chunker.py         # Text chunking
  embeddings.py      # Embedding generation
  index_documents.py # Document indexing
  pdf_loader.py      # PDF text extraction
  rag_assistant.py   # RAG logic
  search.py          # Vector search

experiments/
  test_llm.py
  test_rag.py

chat.py              # Interactive assistant
```

---

## Example Workflow

1. Add documents to `documents/`
2. Run the indexer
3. Start the chat assistant
4. Ask questions about your documents

---

## Notes

This project is a **learning exercise** exploring how local AI systems work, including:

* local LLM inference
* embeddings
* vector search
* RAG pipelines
* document ingestion

---

## Future Improvements

Possible improvements:

* conversation memory
* hybrid search (vector + keyword)
* better chunking strategies
* web interface
* multi-document source references

---

## License

MIT License
