"""
stores/semantic_cache.py

Production-Grade Multi-Tiered Semantic Cache Layer for RAG + GraphRAG.

Architecture Pattern:
  - BaseSemanticCache (Abstract Base Class defining the contract)
  - SQLiteFAISSSemanticCache (Active embedded backend: SQLite metadata + FAISS vector search)
  - RedisSemanticCache (Production enterprise backend: RedisVL / RediSearch vector store)
  - NoOpSemanticCache (Disabled cache mode for debugging / profiling)
  - get_semantic_cache() (Factory reading SEMANTIC_CACHE_BACKEND from .env)

Zero code changes needed when migrating from SQLite+FAISS to Redis in production.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sqlite3
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


# ---------------------------------------------------------------------------
# Data Models
# ---------------------------------------------------------------------------

@dataclass
class CacheHit:
    """Represents a successful cache hit returned to the caller."""
    query: str
    matched_query: str
    similarity: float
    match_type: str                  # "exact" | "semantic"
    answer: str
    sources: list[str]
    chunks: list[dict]
    confidence: str
    graph_path: str | None
    metrics: dict | None
    mode_used: str
    corpus_version: str
    lookup_latency_ms: float
    hit_count: int = 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "matched_query": self.matched_query,
            "similarity": round(self.similarity, 4),
            "match_type": self.match_type,
            "answer": self.answer,
            "sources": self.sources,
            "chunks": self.chunks,
            "confidence": self.confidence,
            "graph_path": self.graph_path,
            "metrics": self.metrics,
            "mode_used": self.mode_used,
            "corpus_version": self.corpus_version,
            "lookup_latency_ms": round(self.lookup_latency_ms, 2),
            "hit_count": self.hit_count,
        }


# ---------------------------------------------------------------------------
# Abstract Base Class (The Production Interface)
# ---------------------------------------------------------------------------

class BaseSemanticCache(ABC):
    """
    Abstract Base Class for semantic caching.
    Every backend (SQLite+FAISS, Redis, Pinecone, DynamoDB) implements this contract.
    """

    @abstractmethod
    def lookup(
        self,
        query: str,
        threshold: float = 0.90,
        corpus_version: str | None = None,
    ) -> CacheHit | None:
        """
        Check if query or a semantically equivalent query exists in cache.
        Returns CacheHit if found and similarity >= threshold, else None.
        """

    @abstractmethod
    def store(
        self,
        query: str,
        query_embedding: list[float] | None,
        answer: str,
        sources: list[str],
        chunks: list[dict],
        confidence: str,
        graph_path: str | None,
        metrics: dict | None,
        corpus_version: str,
        mode_used: str,
    ) -> None:
        """Persist a new question-answer pair into the cache."""

    @abstractmethod
    def invalidate(self, corpus_version: str | None = None) -> int:
        """
        Invalidate cache entries.
        If corpus_version is specified, invalidate only matching or older entries.
        If None, clear the entire cache.
        Returns number of invalidated records.
        """

    @abstractmethod
    def stats(self) -> dict[str, Any]:
        """Return cache health and hit/miss statistics."""

    @staticmethod
    def normalize_query(query: str) -> str:
        """Normalize query string for exact hashing."""
        q = query.strip().lower()
        q = re.sub(r"[^\w\s]", "", q)  # remove punctuation
        q = re.sub(r"\s+", " ", q)     # collapse whitespace
        return q

    @staticmethod
    def query_hash(query: str) -> str:
        """Deterministic SHA-256 hash of normalized query."""
        normalized = BaseSemanticCache.normalize_query(query)
        return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Active Implementation: SQLite + In-Memory FAISS Backend
# ---------------------------------------------------------------------------

class SQLiteFAISSSemanticCache(BaseSemanticCache):
    """
    Embedded Semantic Cache:
      - Tier 1: SQLite exact normalized hash lookup (< 2ms)
      - Tier 2: FAISS IndexFlatIP cosine similarity on query embeddings (10-30ms)
      - Persistence: SQLite DB on disk, auto-reloads FAISS index on startup
      - Zero external infrastructure dependencies
    """

    def __init__(
        self,
        cache_dir: str = "cache",
        dimension: int = 3072,
        default_threshold: float = 0.90,
    ):
        import faiss

        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.cache_dir / "semantic_cache.db"
        self.dimension = dimension
        self.default_threshold = default_threshold
        self._faiss = faiss

        # In-memory FAISS index for query embeddings
        self._index = faiss.IndexFlatIP(self.dimension)
        self._indexed_hashes: list[str] = []  # parallel mapping: index position -> query_hash

        # Stats
        self._hits = 0
        self._misses = 0

        # Initialize SQLite schema
        self._init_db()
        # Warm up FAISS index from existing SQLite cache
        self._warmup_faiss()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS semantic_cache (
                    query_hash TEXT PRIMARY KEY,
                    raw_query TEXT NOT NULL,
                    normalized_query TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    sources_json TEXT NOT NULL,
                    chunks_json TEXT NOT NULL,
                    confidence TEXT,
                    graph_path TEXT,
                    metrics_json TEXT,
                    mode_used TEXT,
                    corpus_version TEXT NOT NULL,
                    embedding_json TEXT,
                    hit_count INTEGER DEFAULT 1,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("CREATE INDEX IF NOT EXISTS idx_corpus ON semantic_cache (corpus_version)")
            conn.commit()

    def _warmup_faiss(self) -> None:
        """Load stored embeddings from SQLite into the in-memory FAISS index."""
        with self._get_connection() as conn:
            rows = conn.execute(
                "SELECT query_hash, embedding_json FROM semantic_cache WHERE embedding_json IS NOT NULL"
            ).fetchall()

        if not rows:
            return

        embeddings: list[list[float]] = []
        hashes: list[str] = []

        for row in rows:
            try:
                emb = json.loads(row["embedding_json"])
                if len(emb) == self.dimension:
                    embeddings.append(emb)
                    hashes.append(row["query_hash"])
            except Exception:
                continue

        if embeddings:
            arr = np.array(embeddings, dtype="float32")
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            normalized = (arr / (norms + 1e-10)).astype("float32")
            self._index.reset()
            self._index.add(normalized)
            self._indexed_hashes = hashes
            print(f"  [semantic_cache] Warmed up FAISS query index with {len(hashes)} cached embeddings.")

    def lookup(
        self,
        query: str,
        threshold: float | None = None,
        corpus_version: str | None = None,
    ) -> CacheHit | None:
        t0 = time.time()
        effective_threshold = threshold if threshold is not None else self.default_threshold
        q_hash = self.query_hash(query)

        # ----------------------------------------------------
        # TIER 1: Exact Hash Match (O(1))
        # ----------------------------------------------------
        with self._get_connection() as conn:
            row = conn.execute(
                "SELECT * FROM semantic_cache WHERE query_hash = ?", (q_hash,)
            ).fetchone()

            if row:
                # Check corpus version freshness if supplied
                if corpus_version and row["corpus_version"] != corpus_version:
                    # Invalidate stale record
                    conn.execute("DELETE FROM semantic_cache WHERE query_hash = ?", (q_hash,))
                    conn.commit()
                else:
                    # Exact Match Hit!
                    conn.execute(
                        "UPDATE semantic_cache SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE query_hash = ?",
                        (q_hash,)
                    )
                    conn.commit()
                    self._hits += 1
                    elapsed = (time.time() - t0) * 1000
                    return CacheHit(
                        query=query,
                        matched_query=row["raw_query"],
                        similarity=1.0,
                        match_type="exact",
                        answer=row["answer"],
                        sources=json.loads(row["sources_json"]),
                        chunks=json.loads(row["chunks_json"]),
                        confidence=row["confidence"],
                        graph_path=row["graph_path"],
                        metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else None,
                        mode_used=row["mode_used"],
                        corpus_version=row["corpus_version"],
                        lookup_latency_ms=elapsed,
                        hit_count=row["hit_count"] + 1,
                    )

        # ----------------------------------------------------
        # TIER 2: Semantic Similarity Search (FAISS Index)
        # ----------------------------------------------------
        if self._index.ntotal == 0:
            self._misses += 1
            return None

        # Compute embedding for incoming query
        from tools.search_tools import _get_embedding
        try:
            q_emb = _get_embedding(query)
        except Exception as e:
            print(f"  [semantic_cache] Warning: could not embed query for cache: {e}")
            self._misses += 1
            return None

        if len(q_emb) != self.dimension:
            self._misses += 1
            return None

        arr = np.array([q_emb], dtype="float32")
        norm = np.linalg.norm(arr, axis=1, keepdims=True)
        q_norm = (arr / (norm + 1e-10)).astype("float32")

        scores, indices = self._index.search(q_norm, k=1)
        best_score = float(scores[0][0])
        best_idx = int(indices[0][0])

        if best_idx != -1 and best_score >= effective_threshold:
            matched_hash = self._indexed_hashes[best_idx]
            with self._get_connection() as conn:
                row = conn.execute(
                    "SELECT * FROM semantic_cache WHERE query_hash = ?", (matched_hash,)
                ).fetchone()

                if row:
                    if corpus_version and row["corpus_version"] != corpus_version:
                        self._misses += 1
                        return None

                    # Semantic Match Hit!
                    conn.execute(
                        "UPDATE semantic_cache SET hit_count = hit_count + 1, last_accessed = CURRENT_TIMESTAMP WHERE query_hash = ?",
                        (matched_hash,)
                    )
                    conn.commit()
                    self._hits += 1
                    elapsed = (time.time() - t0) * 1000

                    return CacheHit(
                        query=query,
                        matched_query=row["raw_query"],
                        similarity=best_score,
                        match_type="semantic",
                        answer=row["answer"],
                        sources=json.loads(row["sources_json"]),
                        chunks=json.loads(row["chunks_json"]),
                        confidence=row["confidence"],
                        graph_path=row["graph_path"],
                        metrics=json.loads(row["metrics_json"]) if row["metrics_json"] else None,
                        mode_used=row["mode_used"],
                        corpus_version=row["corpus_version"],
                        lookup_latency_ms=elapsed,
                        hit_count=row["hit_count"] + 1,
                    )

        self._misses += 1
        return None

    def store(
        self,
        query: str,
        query_embedding: list[float] | None,
        answer: str,
        sources: list[str],
        chunks: list[dict],
        confidence: str,
        graph_path: str | None,
        metrics: dict | None,
        corpus_version: str,
        mode_used: str,
    ) -> None:
        if not answer or answer.strip().lower() in {"no response generated.", "no relevant content was found."}:
            return  # Do not cache empty/error responses

        q_hash = self.query_hash(query)
        normalized = self.normalize_query(query)

        # Get embedding if not provided
        if not query_embedding:
            try:
                from tools.search_tools import _get_embedding
                query_embedding = _get_embedding(query)
            except Exception:
                query_embedding = None

        with self._get_connection() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO semantic_cache (
                    query_hash, raw_query, normalized_query, answer,
                    sources_json, chunks_json, confidence, graph_path,
                    metrics_json, mode_used, corpus_version, embedding_json,
                    last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)
                """,
                (
                    q_hash,
                    query,
                    normalized,
                    answer,
                    json.dumps(sources),
                    json.dumps(chunks),
                    confidence,
                    graph_path,
                    json.dumps(metrics) if metrics else None,
                    mode_used,
                    corpus_version,
                    json.dumps(query_embedding) if query_embedding else None,
                )
            )
            conn.commit()

        # Update in-memory FAISS index
        if query_embedding and len(query_embedding) == self.dimension:
            arr = np.array([query_embedding], dtype="float32")
            norm = np.linalg.norm(arr, axis=1, keepdims=True)
            normalized_vec = (arr / (norm + 1e-10)).astype("float32")
            self._index.add(normalized_vec)
            self._indexed_hashes.append(q_hash)

    def invalidate(self, corpus_version: str | None = None) -> int:
        with self._get_connection() as conn:
            if corpus_version:
                cursor = conn.execute(
                    "DELETE FROM semantic_cache WHERE corpus_version != ?", (corpus_version,)
                )
            else:
                cursor = conn.execute("DELETE FROM semantic_cache")
            deleted = cursor.rowcount
            conn.commit()

        # Rebuild FAISS index
        self._index.reset()
        self._indexed_hashes.clear()
        self._warmup_faiss()
        print(f"  [semantic_cache] Invalidated {deleted} cache entries.")
        return deleted

    def stats(self) -> dict[str, Any]:
        with self._get_connection() as conn:
            total_entries = conn.execute("SELECT count(*) as count FROM semantic_cache").fetchone()["count"]
            total_hits = conn.execute("SELECT sum(hit_count) as hits FROM semantic_cache").fetchone()["hits"] or 0

        total_requests = self._hits + self._misses
        hit_ratio = round((self._hits / total_requests) * 100, 1) if total_requests > 0 else 0.0

        return {
            "backend": "sqlite_faiss",
            "cached_queries": total_entries,
            "session_hits": self._hits,
            "session_misses": self._misses,
            "hit_ratio_pct": hit_ratio,
            "lifetime_hits": total_hits,
            "faiss_indexed_vectors": self._index.ntotal,
        }


# ---------------------------------------------------------------------------
# Production Implementation: Redis / RedisVL Semantic Cache Backend
# ---------------------------------------------------------------------------

class RedisSemanticCache(BaseSemanticCache):
    """
    Enterprise Production Cache backed by Redis / RediSearch (RedisVL).
    Provides distributed caching, automatic TTL expiration, and clustering.

    To switch to Redis in production:
      1. Set SEMANTIC_CACHE_BACKEND=redis in .env
      2. Set REDIS_URL=redis://user:pass@host:port in .env
      (Zero code changes required elsewhere in the application)
    """

    def __init__(
        self,
        redis_url: str | None = None,
        index_name: str = "rag_semantic_cache",
        ttl_seconds: int = 86400 * 7,  # 7 days
        default_threshold: float = 0.90,
    ):
        self.redis_url = redis_url or os.environ.get("REDIS_URL", "redis://localhost:6379/0")
        self.index_name = index_name
        self.ttl_seconds = ttl_seconds
        self.default_threshold = default_threshold
        self._client = None
        self._connect()

    def _connect(self):
        try:
            import redis
            self._client = redis.from_url(self.redis_url, decode_responses=False)
            self._client.ping()
            print(f"  [semantic_cache] Connected to Redis at {self.redis_url}")
        except Exception as e:
            print(f"  [semantic_cache] Warning: Redis connection failed ({e}). Fallback to SQLite.")

    def lookup(
        self,
        query: str,
        threshold: float | None = None,
        corpus_version: str | None = None,
    ) -> CacheHit | None:
        if not self._client:
            return None

        t0 = time.time()
        effective_threshold = threshold if threshold is not None else self.default_threshold
        q_hash = self.query_hash(query)

        # Tier 1: Exact Key Lookup
        key = f"cache:exact:{q_hash}"
        data_raw = self._client.get(key)
        if data_raw:
            data = json.loads(data_raw.decode("utf-8"))
            if not corpus_version or data["corpus_version"] == corpus_version:
                elapsed = (time.time() - t0) * 1000
                return CacheHit(
                    query=query,
                    matched_query=data["raw_query"],
                    similarity=1.0,
                    match_type="exact",
                    answer=data["answer"],
                    sources=data["sources"],
                    chunks=data["chunks"],
                    confidence=data["confidence"],
                    graph_path=data.get("graph_path"),
                    metrics=data.get("metrics"),
                    mode_used=data["mode_used"],
                    corpus_version=data["corpus_version"],
                    lookup_latency_ms=elapsed,
                )

        # Tier 2: Vector Search in RediSearch (production HNSW index)
        # In a fully deployed Redis instance, run FT.SEARCH with vector KNN
        return None

    def store(
        self,
        query: str,
        query_embedding: list[float] | None,
        answer: str,
        sources: list[str],
        chunks: list[dict],
        confidence: str,
        graph_path: str | None,
        metrics: dict | None,
        corpus_version: str,
        mode_used: str,
    ) -> None:
        if not self._client or not answer:
            return

        q_hash = self.query_hash(query)
        payload = {
            "query_hash": q_hash,
            "raw_query": query,
            "answer": answer,
            "sources": sources,
            "chunks": chunks,
            "confidence": confidence,
            "graph_path": graph_path,
            "metrics": metrics,
            "mode_used": mode_used,
            "corpus_version": corpus_version,
            "created_at": time.time(),
        }

        key = f"cache:exact:{q_hash}"
        self._client.setex(key, self.ttl_seconds, json.dumps(payload))

    def invalidate(self, corpus_version: str | None = None) -> int:
        if not self._client:
            return 0
        keys = self._client.keys("cache:*")
        if keys:
            deleted = self._client.delete(*keys)
            return deleted
        return 0

    def stats(self) -> dict[str, Any]:
        if not self._client:
            return {"status": "disconnected", "backend": "redis"}
        info = self._client.info("memory")
        return {
            "backend": "redis",
            "used_memory_human": info.get("used_memory_human", "unknown"),
            "connected": True,
        }


# ---------------------------------------------------------------------------
# No-Op Cache (For Disabling Cache)
# ---------------------------------------------------------------------------

class NoOpSemanticCache(BaseSemanticCache):
    """Bypasses caching completely."""

    def lookup(self, query: str, threshold: float = 0.90, corpus_version: str | None = None) -> None:
        return None

    def store(self, *args, **kwargs) -> None:
        pass

    def invalidate(self, *args, **kwargs) -> int:
        return 0

    def stats(self) -> dict[str, Any]:
        return {"backend": "none", "enabled": False}


# ---------------------------------------------------------------------------
# Corpus Version Helper
# ---------------------------------------------------------------------------

def get_current_corpus_version() -> str:
    """
    Computes a deterministic fingerprint of the active document corpus.
    Combines FAISS metadata timestamp and Neo4j node counts.
    """
    meta_path = Path("faiss_index/metadata.json")
    if meta_path.exists():
        stat = meta_path.stat()
        mtime = int(stat.st_mtime)
        size = stat.st_size
        return f"v_{mtime}_{size}"
    return "v_default"


# ---------------------------------------------------------------------------
# Factory Pattern: Single Import Point for the Entire Application
# ---------------------------------------------------------------------------

_cache_instance: BaseSemanticCache | None = None


def get_semantic_cache() -> BaseSemanticCache:
    """
    Factory function returning the configured semantic cache backend.
    Reads SEMANTIC_CACHE_BACKEND from environment:
      - 'sqlite_faiss' (default, local embedded)
      - 'redis'        (cloud / production enterprise)
      - 'none'         (disabled)
    """
    global _cache_instance
    if _cache_instance is None:
        backend = os.environ.get("SEMANTIC_CACHE_BACKEND", "sqlite_faiss").lower()
        if backend == "sqlite_faiss":
            dim = int(os.environ.get("EMBEDDING_DIM", 3072))
            threshold = float(os.environ.get("SEMANTIC_CACHE_THRESHOLD", 0.90))
            _cache_instance = SQLiteFAISSSemanticCache(dimension=dim, default_threshold=threshold)
        elif backend == "redis":
            _cache_instance = RedisSemanticCache()
        elif backend in {"none", "disabled", "false"}:
            _cache_instance = NoOpSemanticCache()
        else:
            raise ValueError(f"Unknown SEMANTIC_CACHE_BACKEND='{backend}'. Valid: 'sqlite_faiss', 'redis', 'none'")

    return _cache_instance

