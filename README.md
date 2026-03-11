# Local AI Document Assistant

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Ollama](https://img.shields.io/badge/Ollama-Local%20AI-orange.svg)](https://ollama.ai/)

**Enterprise-Grade Local RAG System for Intelligent Document Q&A**

[Aplicación de demostración de capacidades de desarrollo en IA local y sistemas RAG]

</div>

---

## Overview

This project demonstrates a production-ready **Retrieval-Augmented Generation (RAG)** system that runs entirely on local infrastructure. It showcases advanced AI engineering patterns including vector search, semantic embeddings, intelligent document processing, and local LLM inference.

### Key Capabilities

- **Local-First Architecture**: Zero external API dependencies - all processing happens on your machine
- **Intelligent Document Processing**: Support for PDF and text files with hierarchical chunking
- **Semantic Search**: Vector-based similarity search using state-of-the-art embeddings
- **Reranking**: Cross-encoder reranking for improved relevance
- **Streaming Responses**: Real-time token-by-token LLM output
- **Source Attribution**: Answers include references to source documents

---

## Technology Stack

| Component | Technology | Purpose |
|-----------|------------|---------|
| **Runtime** | Python 3.12+ | Core application logic |
| **LLM** | Mistral 7B (via Ollama) | Local language model inference |
| **Embeddings** | nomic-embed-text | Text vectorization |
| **Vector DB** | ChromaDB | Persistent vector storage & search |
| **Reranker** | BAAI/bge-reranker-base | Relevance scoring & ranking |
| **PDF Processing** | PyPDF | Document text extraction |
| **HTTP Client** | httpx | Async/sync API communication |
| **Data Validation** | Pydantic | Schema definitions |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        LOCAL AI DOCUMENT ASSISTANT                          │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌───────────┐ │
│  │   Documents  │───▶│ PDF Loader   │───▶│   Chunker    │───▶│ Embeddings│ │
│  │  (PDF/TXT)   │    │   (PyPDF)    │    │ (Hierarchical)│   │  (Ollama) │ │
│  └──────────────┘    └──────────────┘    └──────────────┘    └───────────┘ │
│                                                                     │       │
│                                                                     ▼       │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                        CHROMA VECTOR DATABASE                          ││
│  │                  (Persistent Local Vector Store)                      ││
│  └─────────────────────────────────────────────────────────────────────────┘│
│                                                                     │       │
│                              ┌────────────────────────────────────────┘    │
│                              ▼                                                 │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐                  │
│  │    Search    │───▶│  Reranker    │───▶│   Context    │                  │
│  │  (Vector SIM)│    │(Cross-Encoder)│   │   Builder    │                  │
│  └──────────────┘    └──────────────┘    └──────────────┘                  │
│                                                     │                        │
│                                                     ▼                        │
│                              ┌────────────────────────────────────────┐     │
│                              │            MISTRAL 7B (OLLAMA)        │     │
│                              │         (Local LLM Generation)        │     │
│                              └────────────────────────────────────────┘     │
│                                                     │                        │
│                                                     ▼                        │
│                              ┌────────────────────────────────────────┐     │
│                              │          TERMINAL INTERFACE            │     │
│                              │     (Streaming Output + Sources)       │     │
│                              └────────────────────────────────────────┘     │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Pipeline Details

1. **Document Ingestion**: PDF and TXT files are loaded and text is extracted
2. **Hierarchical Chunking**: Documents are split into parent chunks (1800 chars) with overlapping child chunks (600 chars) for better context preservation
3. **Embedding Generation**: Text chunks are vectorized using `nomic-embed-text` model
4. **Vector Storage**: Embeddings are persisted in ChromaDB for fast similarity search
5. **Query Processing**: User questions are embedded and searched against the vector database
6. **Reranking**: Top results are reranked using cross-encoder for improved relevance
7. **Context Assembly**: Best matching chunks are assembled into a context window
8. **Answer Generation**: Local LLM generates answers based on retrieved context
9. **Streaming Output**: Responses are streamed token-by-token to the user

---

## Features

### Core Features

- **Local LLM Inference**: Privacy-focused AI that never sends data to external services
- **Multi-Format Support**: Process PDF documents and plain text files
- **Hierarchical Chunking**: Advanced document segmentation preserving semantic context
- **Vector Semantic Search**: Find relevant content using embedding similarity
- **Cross-Encoder Reranking**: Improve search results with BAAI/bge-reranker-base
- **Streaming Responses**: Real-time token generation for immediate feedback
- **Source Attribution**: Track which documents informed each answer
- **Embedding Cache**: Avoid redundant embedding computation
- **Robust Error Handling**: Graceful degradation with custom exceptions
- **Configurable Parameters**: Fine-tune chunk sizes, timeouts, and model settings

### Engineering Patterns Demonstrated

- **Async/Await**: Non-blocking LLM and HTTP operations
- **ThreadPoolExecutor**: Parallel embedding generation
- **Retry with Exponential Backoff**: Resilience to transient failures
- **Custom Exception Hierarchy**: Clear error categorization
- **Structured Logging**: Configurable logging with levels
- **Configuration Management**: Centralized settings via config.py
- **Pydantic Models**: Type-safe configuration schemas

---

## Project Structure

```
local-ai-assistant/
│
├── config.py                 # Centralized configuration management
├── chat.py                   # Interactive CLI chat interface
│
├── llm/
│   ├── __init__.py
│   └── client.py             # Ollama LLM client (sync/async + streaming)
│
├── rag/
│   ├── __init__.py
│   ├── chunker.py            # Hierarchical text chunking
│   ├── embeddings.py         # Embedding generation with caching
│   ├── exceptions.py         # Custom exception hierarchy
│   ├── index_documents.py    # Document indexing pipeline
│   ├── logger.py             # Logging configuration
│   ├── models.py            # Pydantic configuration models
│   ├── pdf_loader.py        # PDF text extraction
│   ├── rag_assistant.py     # Core RAG logic
│   ├── reranker.py           # Cross-encoder reranking
│   ├── search.py             # Vector similarity search
│   └── cache.py             # Embedding cache management
│
├── cli/
│   └── assistant.py          # CLI utilities
│
├── experiments/
│   ├── test_llm.py          # LLM testing utilities
│   └── test_rag.py          # RAG pipeline testing
│
├── documents/                # Input documents (PDF/TXT)
├── chroma_db/               # Persisted vector database
├── embeddings_cache.json    # Cached embeddings
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

---

## Installation

### Prerequisites

- Python 3.12 or higher
- [Ollama](https://ollama.ai/) installed and running locally

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/local-ai-assistant.git
cd local-ai-assistant
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Install Ollama Models

```bash
# Pull the Mistral language model
ollama pull mistral

# Pull the embedding model
ollama pull nomic-embed-text
```

### 4. Start Ollama (if not running)

```bash
ollama serve
```

---

## Usage

### Step 1: Add Documents

Place your PDF or text files in the `documents/` directory:

```bash
cp your_document.pdf documents/
```

### Step 2: Index Documents

Run the document indexing pipeline:

```bash
python -m rag.index_documents
```

This will:
- Extract text from all supported documents
- Split documents into hierarchical chunks
- Generate embeddings for each chunk
- Store vectors in the local Chroma database

### Step 3: Start Chat Session

Launch the interactive assistant:

```bash
python chat.py
```

### Example Session

```
Local AI Document Assistant
Type 'exit' to quit

You: What is the main topic of the documents?

AI:
The documents primarily discuss advanced software engineering 
topics including AI systems, local LLM inference, and retrieval-
augmented generation pipelines.

Sources: ['chunk_12', 'chunk_8', 'chunk_45']

Response time: 4.32 seconds
```

---

## Configuration

All settings are centralized in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `MODEL` | `mistral` | LLM model name |
| `EMBED_MODEL` | `nomic-embed-text` | Embedding model |
| `PARENT_CHUNK_SIZE` | `1800` | Parent chunk size (characters) |
| `CHILD_CHUNK_SIZE` | `600` | Child chunk size (characters) |
| `TOP_K` | `10` | Number of search results |
| `MAX_CONTEXT_CHARS` | `4000` | Maximum context length |
| `LLM_TIMEOUT` | `300` | LLM request timeout (seconds) |
| `EMBEDDING_TIMEOUT` | `120` | Embedding timeout (seconds) |
| `MAX_RETRIES` | `3` | Retry attempts |
| `INDEXING_MAX_WORKERS` | `4` | Parallel workers for embeddings |

---

## API Reference

### Core Functions

#### `rag.index_documents.main()`

Index all documents in the `documents/` directory.

#### `rag.rag_assistant.ask_with_context(question: str) -> list[str]`

Answer a question using RAG. Returns source chunk IDs.

#### `rag.search.search_documents(query: str, n_results: int) -> dict`

Perform vector similarity search.

#### `llm.client.ask_llm(prompt: str) -> str`

Get a complete LLM response.

#### `llm.client.stream_llm(prompt: str) -> None`

Stream LLM response token-by-token.

---

## Engineering Highlights

### Error Handling

Custom exception hierarchy for precise error handling:

```
RAGError (base)
├── LLMError
├── EmbeddingError
├── DocumentLoadError
├── SearchError
└── CollectionEmptyError
```

### Resilience Patterns

- **Exponential Backoff**: Retry with increasing delays
- **Timeout Management**: Configurable timeouts per operation
- **Graceful Degradation**: Meaningful error messages
- **Logging**: Structured logging at DEBUG/INFO/WARNING/ERROR levels

### Performance Optimizations

- **Embedding Cache**: Avoid recomputing embeddings for identical text
- **Parallel Processing**: ThreadPoolExecutor for batch embeddings
- **Hierarchical Chunking**: Balance between context and precision
- **Streaming**: Immediate feedback without waiting for full response

---

## Development

### Running Tests

```bash
# Test LLM integration
python -m experiments.test_llm

# Test RAG pipeline
python -m experiments.test_rag
```

### Clearing Cache

```bash
python -c "from rag.cache import clear_cache; clear_cache()"
```

---

## Future Enhancements

This project is designed for extensibility. Potential additions:

- **Conversation Memory**: Multi-turn conversation context
- **Web Interface**: REST API + frontend
- **Hybrid Search**: Combine vector + keyword search
- **Advanced Chunking**: Sentence-aware splitting
- **Multi-modal Support**: Image and table understanding
- **Evaluation Metrics**: RAGAS-style performance metrics

---

## License

MIT License - See [LICENSE](LICENSE) for details.

---

## Acknowledgments

- [Ollama](https://ollama.ai/) - Local LLM runtime
- [Chroma](https://www.trychroma.com/) - Vector database
- [Hugging Face](https://huggingface.co/) - Reranker model

---

<div align="center">

**Built with modern AI engineering practices**

</div>
