"""
ingestion/embedder.py

Phase 1 — Step 3: Embedding.

Takes Chunk objects -> returns (chunks, embeddings) ready for the vector store.

Backends (set EMBEDDING_BACKEND in .env):
1. google — text-embedding-004, dim=768 (FREE Tier via Google AI Studio API Key)
2. vertex — textembedding-gecko@003, dim=768 (GCP Vertex AI)
3. openai — text-embedding-3-small, dim=1536
"""

from __future__ import annotations

import os
import time
from abc import ABC, abstractmethod


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseEmbedder(ABC):
    """Common interface for all embedding backends."""

    @property
    @abstractmethod
    def dimension(self) -> int:
        """Embedding vector dimension — must match FAISS index and Pinecone index."""

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Embed a list of text strings.
        Returns a list of float vectors, same length and order as input.
        """

    def embed_query(self, text: str) -> list[float]:
        """Convenience wrapper for single-text embedding (used at retrieval time)."""
        return self.embed_texts([text])[0]


# ---------------------------------------------------------------------------
# Google AI Studio Backend (Free Tier API Key)
# ---------------------------------------------------------------------------

class GoogleGenAIEmbedder(BaseEmbedder):
    """
    Google AI Studio Developer API embedding (Free Tier compatible).
    Model: text-embedding-004 -> dim=768

    Requires:
    GOOGLE_API_KEY or GEMINI_API_KEY in environment variables.
    """

    BATCH_SIZE = 100  # Conservative batch size to stay within RPM limits

    def __init__(self):
        from google import genai
        api_key = os.environ.get("gemini_api_key") 
        if not api_key:
            raise ValueError("GOOGLE_API_KEY or GEMINI_API_KEY must be set in environment.")
        
        self._client = genai.Client(api_key=api_key)
        self._model = os.environ.get("GOOGLE_EMBEDDING_MODEL", "text-embedding-004")
        self._dim = 768

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [t if t.strip() else " " for t in texts]
        embeddings: list[list[float]] = []
        
        for i in range(0, len(safe_texts), self.BATCH_SIZE):
            batch = safe_texts[i : i + self.BATCH_SIZE]
            print(f"  [embedder:google] batch {i // self.BATCH_SIZE + 1} — {len(batch)} texts")
            
            # Request embeddings batch from Google API
            response = self._client.models.embed_content(
                model=self._model,
                contents=batch
            )
            
            # Extract floats
            batch_embeddings = [item.values for item in response.embeddings]
            embeddings.extend(batch_embeddings)
            
            # Rate limit buffer for free tier
            if i + self.BATCH_SIZE < len(safe_texts):
                time.sleep(0.5)
                
        return embeddings


# ---------------------------------------------------------------------------
# Vertex AI backend
# ---------------------------------------------------------------------------

class VertexEmbedder(BaseEmbedder):
    BATCH_SIZE = 250

    def __init__(self):
        from vertexai.language_models import TextEmbeddingModel
        import vertexai
        vertexai.init(project=os.environ.get("GOOGLE_CLOUD_PROJECT"))
        model_name = os.environ.get("VERTEX_EMBEDDING_MODEL", "textembedding-gecko@003")
        self._model = TextEmbeddingModel.from_pretrained(model_name)
        self._dim = 768

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        embeddings: list[list[float]] = []
        for i in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[i : i + self.BATCH_SIZE]
            print(f"  [embedder:vertex] batch {i // self.BATCH_SIZE + 1} — {len(batch)} texts")
            results = self._model.get_embeddings(batch)
            embeddings.extend([r.values for r in results])
            if i + self.BATCH_SIZE < len(texts):
                time.sleep(0.5)
        return embeddings


# ---------------------------------------------------------------------------
# OpenAI backend
# ---------------------------------------------------------------------------

class OpenAIEmbedder(BaseEmbedder):
    BATCH_SIZE = 512

    def __init__(self):
        from openai import OpenAI
        self._client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self._model = os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
        self._dim = 1536 if "3-small" in self._model else 3072

    @property
    def dimension(self) -> int:
        return self._dim

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        safe_texts = [t if t.strip() else " " for t in texts]
        embeddings: list[list[float]] = []
        for i in range(0, len(safe_texts), self.BATCH_SIZE):
            batch = safe_texts[i : i + self.BATCH_SIZE]
            print(f"  [embedder:openai] batch {i // self.BATCH_SIZE + 1} — {len(batch)} texts")
            resp = self._client.embeddings.create(input=batch, model=self._model)
            batch_embeddings = [item.embedding for item in resp.data]
            embeddings.extend(batch_embeddings)
            if i + self.BATCH_SIZE < len(safe_texts):
                time.sleep(0.2)
        return embeddings


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def get_embedder() -> BaseEmbedder:
    """
    Return the configured embedding backend.
    Reads EMBEDDING_BACKEND from env (google | vertex | openai).
    """
    backend = os.environ.get("EMBEDDING_BACKEND", "google").lower()
    if backend == "google":
        return GoogleGenAIEmbedder()
    elif backend == "vertex":
        return VertexEmbedder()
    elif backend == "openai":
        return OpenAIEmbedder()
    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND='{backend}'. Valid: 'google', 'vertex', 'openai'"
        )


# ---------------------------------------------------------------------------
# Embed + store convenience function
# ---------------------------------------------------------------------------

def embed_and_store_chunks(chunks) -> tuple[list, list[list[float]]]:
    from stores.vector_store import get_vector_store

    embedder = get_embedder()
    store = get_vector_store()
    store.load()

    texts = [c.text for c in chunks]
    metadatas = [c.metadata for c in chunks]

    print(f"  [embedder] embedding {len(texts)} chunks with {type(embedder).__name__}...")
    embeddings = embedder.embed_texts(texts)

    print(f"  [embedder] writing to vector store ({type(store).__name__})...")
    store.add_documents(texts, metadatas, embeddings)
    store.persist()

    print(f"  [embedder] ✓ {len(chunks)} chunks stored.")
    return chunks, embeddings