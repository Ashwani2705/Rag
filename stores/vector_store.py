"""
Vector store abstraction layer.

Swap the backend by changing VECTOR_STORE_BACKEND in your .env:
  VECTOR_STORE_BACKEND=faiss   (default, local, no API key needed)
  VECTOR_STORE_BACKEND=pinecone (production, requires PINECONE_API_KEY + PINECONE_INDEX)

All agents and tools import VectorStore from here — zero code changes needed on switch.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Any

# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class BaseVectorStore(ABC):
    """Common interface every backend must satisfy."""

    @abstractmethod
    def add_documents(self, texts: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        """Persist a batch of pre-computed embeddings + metadata."""

    @abstractmethod
    def similarity_search(self, query_embedding: list[float], k: int = 5) -> list[dict]:
        """
        Return the top-k most similar documents.

        Each result dict must contain:
          - text      : str   — the chunk content
          - metadata  : dict  — chunk_id, source, page, etc.
          - score     : float — similarity score (higher = more similar)
        """

    @abstractmethod
    def persist(self) -> None:
        """Save state to disk / remote (no-op for managed stores)."""

    @abstractmethod
    def load(self) -> None:
        """Load state from disk / remote."""


# ---------------------------------------------------------------------------
# FAISS backend
# ---------------------------------------------------------------------------

class FAISSVectorStore(BaseVectorStore):
    """
    Local FAISS store backed by a flat L2 index.
    Metadata and texts are kept in parallel lists (simple but sufficient for demo).

    To upgrade to a persistent FAISS index, call persist() after ingestion —
    it saves index.faiss + metadata.json to FAISS_INDEX_PATH.
    """

    def __init__(self, dimension: int = 768, index_path: str = "faiss_index"):
        import faiss  # pip install faiss-cpu
        import json

        self.dimension = dimension
        self.index_path = index_path
        self.index = faiss.IndexFlatIP(dimension)   # Inner-product ≈ cosine on normalised vecs
        self._texts: list[str] = []
        self._metadatas: list[dict] = []
        self._faiss = faiss
        self._json = json

    def _normalise(self, vecs: list[list[float]]):
        import numpy as np
        arr = np.array(vecs, dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return (arr / (norms + 1e-10)).astype("float32")

    def add_documents(self, texts: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        vecs = self._normalise(embeddings)
        self.index.add(vecs)
        self._texts.extend(texts)
        self._metadatas.extend(metadatas)

    def similarity_search(self, query_embedding: list[float], k: int = 5) -> list[dict]:
        import numpy as np
        if self.index.ntotal == 0:
            return []
        q = self._normalise([query_embedding])
        scores, indices = self.index.search(q, min(k, self.index.ntotal))
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            results.append({
                "text": self._texts[idx],
                "metadata": self._metadatas[idx],
                "score": float(score),
            })
        return results

    def persist(self) -> None:
        import json, os
        os.makedirs(self.index_path, exist_ok=True)
        self._faiss.write_index(self.index, f"{self.index_path}/index.faiss")
        with open(f"{self.index_path}/metadata.json", "w") as f:
            json.dump({"texts": self._texts, "metadatas": self._metadatas}, f)

    def load(self) -> None:
        import json, os
        idx_file = f"{self.index_path}/index.faiss"
        meta_file = f"{self.index_path}/metadata.json"
        if not os.path.exists(idx_file):
            return   # fresh start
        self.index = self._faiss.read_index(idx_file)
        with open(meta_file) as f:
            data = json.load(f)
        self._texts = data["texts"]
        self._metadatas = data["metadatas"]

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """Fetch chunks by their chunk_ids from stored metadata."""
        id_set = set(chunk_ids)
        results = []
        for t, meta in zip(self._texts, self._metadatas):
            cid = meta.get("chunk_id")
            if cid in id_set:
                results.append({
                    "chunk_id": cid,
                    "text": t,
                    "source": meta.get("source", ""),
                    "metadata": meta,
                })
        return results


# ---------------------------------------------------------------------------
# Pinecone backend  (uncomment when switching)
# ---------------------------------------------------------------------------

class PineconeVectorStore(BaseVectorStore):
    """
    Managed Pinecone vector store.

    Required env vars:
      PINECONE_API_KEY
      PINECONE_INDEX       — name of an existing serverless index
      PINECONE_NAMESPACE   — optional, defaults to "rag-demo"

    Switch by setting VECTOR_STORE_BACKEND=pinecone in .env.
    """

    def __init__(self):
        from pinecone import Pinecone  # pip install pinecone-client
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self._index = pc.Index(os.environ["PINECONE_INDEX"])
        self._namespace = os.environ.get("PINECONE_NAMESPACE", "rag-demo")

    def add_documents(self, texts: list[str], metadatas: list[dict], embeddings: list[list[float]]) -> None:
        vectors = [
            {
                "id": meta.get("chunk_id", str(i)),
                "values": emb,
                "metadata": {**meta, "text": text},
            }
            for i, (text, meta, emb) in enumerate(zip(texts, metadatas, embeddings))
        ]
        # Pinecone recommends batches of ≤100
        batch_size = 100
        for i in range(0, len(vectors), batch_size):
            self._index.upsert(vectors=vectors[i : i + batch_size], namespace=self._namespace)

    def similarity_search(self, query_embedding: list[float], k: int = 5) -> list[dict]:
        resp = self._index.query(
            vector=query_embedding,
            top_k=k,
            include_metadata=True,
            namespace=self._namespace,
        )
        results = []
        for match in resp.matches:
            meta = dict(match.metadata)
            text = meta.pop("text", "")
            results.append({"text": text, "metadata": meta, "score": match.score})
        return results

    def persist(self) -> None:
        pass   # Pinecone is managed — nothing to persist locally

    def load(self) -> None:
        pass   # Pinecone is managed — always live


# ---------------------------------------------------------------------------
# Factory — the single import point for all agents/tools
# ---------------------------------------------------------------------------

def get_vector_store(**kwargs) -> BaseVectorStore:
    """
    Read VECTOR_STORE_BACKEND from env and return the correct implementation.

    Usage in any agent or tool:
        from stores.vector_store import get_vector_store
        store = get_vector_store()
        store.load()
    """
    backend = os.environ.get("VECTOR_STORE_BACKEND", "faiss").lower()
    if backend == "faiss":
        return FAISSVectorStore(
            dimension=int(os.environ.get("EMBEDDING_DIM", 768)),
            index_path=os.environ.get("FAISS_INDEX_PATH", "faiss_index"),
        )
    elif backend == "pinecone":
        return PineconeVectorStore()
    else:
        raise ValueError(
            f"Unknown VECTOR_STORE_BACKEND='{backend}'. "
            "Valid options: 'faiss', 'pinecone'"
        )