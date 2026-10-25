"""
stores/store_manager.py

Phase 2 — Unified store coordinator.

StoreManager is the single entry point for all store operations.
It owns both the vector store and graph store, handles startup
initialisation, health checks, and wipe/reset for testing.

Usage:
    from stores.store_manager import StoreManager

    # On app startup
    sm = StoreManager()
    sm.initialise()           # schema + load indexes

    # Health check (used by FastAPI /health)
    report = sm.health()

    # Wipe all stores (test only)
    sm.wipe_all(confirm=True)
"""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class HealthReport:
    vector_store: dict
    graph_store: dict

    @property
    def all_ok(self) -> bool:
        return (
            self.vector_store.get("status") == "ok"
            and self.graph_store.get("status") == "ok"
        )

    def summary(self) -> str:
        lines = ["=== Store Health ==="]
        lines.append(f"Vector store : {self.vector_store}")
        lines.append(f"Graph store  : {self.graph_store}")
        lines.append(f"Overall      : {'[OK] HEALTHY' if self.all_ok else '[X] DEGRADED'}")
        return "\n".join(lines)


class StoreManager:
    """
    Coordinates the vector store (FAISS/Pinecone) and graph store (Neo4j).

    Lifecycle:
        sm = StoreManager()
        sm.initialise()      ← call once at startup

    After initialise():
        sm.vector   → BaseVectorStore (FAISSVectorStore or PineconeVectorStore)
        sm.graph    → Neo4jGraphStore
    """

    def __init__(self):
        from stores.vector_store import get_vector_store
        from stores.graph_store  import get_graph_store

        self.vector = get_vector_store()
        self.graph  = get_graph_store()
        self._initialised = False

    def initialise(self) -> None:
        """
        Run startup tasks for both stores:
          - Vector store: load existing index from disk/remote
          - Graph store:  ensure schema (indexes + constraints)

        Call once when the FastAPI app starts (lifespan handler)
        or at the top of any CLI script.
        """
        if self._initialised:
            return

        print("[store_manager] initialising stores...")

        # Vector store — load persisted index
        try:
            self.vector.load()
            vb = os.environ.get("VECTOR_STORE_BACKEND", "faiss")
            print(f"  [store_manager] vector store ({vb}) loaded.")
        except Exception as e:
            print(f"  [store_manager] WARNING: vector store load failed: {e}")

        # Graph store — ensure schema
        try:
            self.graph.init_schema()
        except Exception as e:
            print(f"  [store_manager] WARNING: graph schema init failed: {e}")

        self._initialised = True
        print("[store_manager] [OK] both stores ready.")

    # ── Health ──────────────────────────────────────────────────────────────

    def health(self) -> HealthReport:
        """
        Check both stores and return a HealthReport.
        Does NOT raise — returns error details in the report dict.
        """
        # Vector store health
        try:
            backend = os.environ.get("VECTOR_STORE_BACKEND", "faiss")
            vector_health = {
                "status":  "ok",
                "backend": backend,
            }
            # For FAISS: check index is loaded
            if hasattr(self.vector, "index"):
                vector_health["vectors_indexed"] = int(self.vector.index.ntotal)
            # For Pinecone: we can't count cheaply, just mark ok
        except Exception as e:
            vector_health = {"status": "error", "detail": str(e)}

        # Graph store health
        graph_health = self.graph.health_check()

        return HealthReport(vector_store=vector_health, graph_store=graph_health)

    # ── Maintenance ─────────────────────────────────────────────────────────

    def wipe_all(self, confirm: bool = False) -> None:
        """
        Wipe BOTH stores completely. For test runs and resets only.
        Pass confirm=True to proceed.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to wipe all stores.")

        print("[store_manager] [!] Wiping all stores...")

        # Vector store — re-initialise in-memory (FAISS) or delete namespace (Pinecone)
        from stores.vector_store import get_vector_store, FAISSVectorStore, PineconeVectorStore
        if isinstance(self.vector, FAISSVectorStore):
            import faiss
            dim = self.vector.dimension
            self.vector.index = faiss.IndexFlatIP(dim)
            self.vector._texts = []
            self.vector._metadatas = []
            # Delete persisted files
            import shutil, os
            idx_path = self.vector.index_path
            if os.path.exists(idx_path):
                shutil.rmtree(idx_path)
            print("  [store_manager] FAISS index cleared.")
        elif isinstance(self.vector, PineconeVectorStore):
            self.vector._index.delete(delete_all=True, namespace=self.vector._namespace)
            print("  [store_manager] Pinecone namespace cleared.")

        # Graph store
        self.graph.wipe(confirm=True)
        print("[store_manager] [OK] all stores wiped.")

    def wipe_vector_only(self, confirm: bool = False) -> None:
        """Wipe only the vector store (keep graph intact)."""
        if not confirm:
            raise ValueError("Pass confirm=True.")
        from stores.vector_store import FAISSVectorStore
        if isinstance(self.vector, FAISSVectorStore):
            import faiss, shutil, os
            self.vector.index = faiss.IndexFlatIP(self.vector.dimension)
            self.vector._texts = []
            self.vector._metadatas = []
            if os.path.exists(self.vector.index_path):
                shutil.rmtree(self.vector.index_path)
        print("  [store_manager] vector store cleared.")

    def wipe_graph_only(self, confirm: bool = False) -> None:
        """Wipe only the graph store (keep vector store intact)."""
        if not confirm:
            raise ValueError("Pass confirm=True.")
        self.graph.wipe(confirm=True)

    # ── Stats ────────────────────────────────────────────────────────────────

    def get_chunks_by_ids(self, chunk_ids: list[str]) -> list[dict]:
        """
        Lookup chunk text by chunk_ids from vector store and/or Neo4j graph store.
        Returns a list of dicts: [{"chunk_id": str, "text": str, "source": str}]
        """
        if not chunk_ids:
            return []
        found: list[dict] = []
        if hasattr(self.vector, "get_chunks_by_ids"):
            found.extend(self.vector.get_chunks_by_ids(chunk_ids))
        
        found_ids = {c["chunk_id"] for c in found}
        missing = [cid for cid in chunk_ids if cid not in found_ids]
        if missing and hasattr(self.graph, "get_chunks_by_ids"):
            found.extend(self.graph.get_chunks_by_ids(missing))
        
        return found

    def stats(self) -> dict:
        """Return combined stats from both stores."""
        graph_stats = self.graph.get_stats()
        vector_count = getattr(self.vector, "index", None)
        vector_count = int(vector_count.ntotal) if vector_count is not None else "n/a"

        return {
            "vector_store": {
                "backend":    os.environ.get("VECTOR_STORE_BACKEND", "faiss"),
                "vectors":    vector_count,
            },
            "graph_store": {
                "chunks":     graph_stats.chunk_count,
                "entities":   graph_stats.entity_count,
                "relations":  graph_stats.relation_count,
                "neo4j":      graph_stats.neo4j_version,
                "indexes":    graph_stats.index_status,
            },
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_manager_instance: StoreManager | None = None


def get_store_manager() -> StoreManager:
    """
    Return the module-level StoreManager singleton.
    Call initialise() before first use.
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = StoreManager()
    return _manager_instance