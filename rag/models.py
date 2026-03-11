from pydantic import BaseModel


class DocumentMetadata(BaseModel):
    """Metadata associated with a document chunk."""
    file: str
    parent_id: int
    parent_text: str


class ChunkConfig(BaseModel):
    """Configuration for text chunking."""
    parent_chunk_size: int = 1800
    parent_overlap: int = 200
    child_chunk_size: int = 600
    child_overlap: int = 80


class LLMConfig(BaseModel):
    """Configuration for LLM settings."""
    model: str = "mistral"
    temperature: float = 0.7
    num_predict: int = 120


class EmbeddingConfig(BaseModel):
    """Configuration for embedding model."""
    model: str = "nomic-embed-text"
    url: str = "http://localhost:11434/api/embeddings"


class OllamaConfig(BaseModel):
    """Configuration for Ollama service."""
    url: str = "http://localhost:11434/api/generate"


class SearchConfig(BaseModel):
    """Configuration for search settings."""
    top_k: int = 10
    max_context_chars: int = 4000
