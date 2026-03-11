"""Custom exceptions for the RAG application."""


class RAGError(Exception):
    """Base exception for RAG-related errors."""
    pass


class LLMError(RAGError):
    """Raised when the LLM service fails."""
    pass


class EmbeddingError(RAGError):
    """Raised when embedding generation fails."""
    pass


class DocumentLoadError(RAGError):
    """Raised when document loading fails."""
    pass


class SearchError(RAGError):
    """Raised when search operations fail."""
    pass


class CollectionEmptyError(SearchError):
    """Raised when searching an empty collection."""
    pass
