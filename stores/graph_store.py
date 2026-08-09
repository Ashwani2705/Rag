"""
stores/graph_store.py

Phase 2 — Neo4j graph store manager.

Responsibilities:
  - Connection management (singleton driver with retry)
  - Schema initialisation (indexes + constraints on first run)
  - CRUD for Chunk nodes, Entity nodes, Relation edges
  - Query interface used by tools/search_tools.py graph_search
  - Health check + stats endpoint
  - Wipe / reset for test runs

All writes use MERGE — fully idempotent. Re-ingesting the same
document never creates duplicate nodes or edges.

Usage:
    from stores.graph_store import get_graph_store
    gs = get_graph_store()
    gs.health_check()
    results = gs.entity_search("API Gateway", max_hops=2)
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any


# ---------------------------------------------------------------------------
# Data models returned by query methods
# ---------------------------------------------------------------------------

@dataclass
class GraphSearchResult:
    start_entity: str
    relations: list[str]
    end_entity: str
    chunk_id: str
    text: str
    hop_count: int


@dataclass
class GraphStats:
    chunk_count: int
    entity_count: int
    relation_count: int
    index_status: list[str]
    neo4j_version: str


# ---------------------------------------------------------------------------
# Graph store
# ---------------------------------------------------------------------------

class Neo4jGraphStore:
    """
    Manages all Neo4j operations for the RAG + GraphRAG project.

    Node labels:
      :Chunk   — a document chunk (chunk_id, text, source, doc_type, doc_id)
      :Entity  — a named entity  (name, label, chunk_ids[])

    Relationship types (dynamic — set during graph build):
      RELATES_TO, DEPENDS_ON, OWNED_BY, PART_OF, CREATED_BY, REPORTS_TO,
      LEADS, USED_BY, IMPACTS — all carry {chunk_id, context} properties

    Fixed structural edges:
      (:Chunk)-[:CONTAINS]->(:Entity)

    Constraints:
      - Entity.name is unique (enforced via Neo4j constraint)
      - Chunk.chunk_id is unique (enforced via Neo4j constraint)
    """

    _instance: Neo4jGraphStore | None = None

    def __init__(
        self,
        uri: str,
        user: str,
        password: str,
        max_retries: int = 5,
        retry_delay: float = 2.0,
    ):
        self.uri = uri
        self.user = user
        self.password = password
        self._driver = None
        self._connect(max_retries, retry_delay)

    # ── Connection ──────────────────────────────────────────────────────────

    def _connect(self, max_retries: int, retry_delay: float) -> None:
        from neo4j import GraphDatabase, exceptions as neo4j_exc

        last_error = None
        for attempt in range(1, max_retries + 1):
            try:
                driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
                driver.verify_connectivity()
                self._driver = driver
                print(f"  [graph_store] connected to Neo4j at {self.uri}")
                return
            except Exception as e:
                last_error = e
                print(f"  [graph_store] connection attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)

        raise ConnectionError(
            f"Could not connect to Neo4j after {max_retries} attempts. "
            f"Last error: {last_error}\n"
            f"Is Neo4j running? Try: docker compose up -d neo4j"
        )

    def close(self) -> None:
        if self._driver:
            self._driver.close()
            self._driver = None

    # ── Schema ──────────────────────────────────────────────────────────────

    def init_schema(self) -> None:
        """
        Create indexes and uniqueness constraints.
        Safe to call on every startup — uses IF NOT EXISTS.

        Constraints prevent duplicate nodes even if graph_builder
        runs concurrent batches.
        """
        with self._driver.session() as s:
            # Uniqueness constraints (also create an index automatically)
            s.run("""
                CREATE CONSTRAINT entity_name_unique IF NOT EXISTS
                FOR (e:Entity) REQUIRE e.name IS UNIQUE
            """)
            s.run("""
                CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS
                FOR (c:Chunk) REQUIRE c.chunk_id IS UNIQUE
            """)
            # Extra index on Entity.label for label-filtered queries
            s.run("""
                CREATE INDEX entity_label IF NOT EXISTS
                FOR (e:Entity) ON (e.label)
            """)
            # Full-text index for fallback keyword search on Chunk.text
            s.run("""
                CREATE FULLTEXT INDEX chunk_text_ft IF NOT EXISTS
                FOR (c:Chunk) ON EACH [c.text]
            """)
        print("  [graph_store] schema initialised (constraints + indexes).")

    # ── Write operations ────────────────────────────────────────────────────

    def upsert_chunk(self, chunk_id: str, text: str, source: str, doc_type: str, doc_id: str) -> None:
        """Insert or update a Chunk node. Idempotent via MERGE."""
        with self._driver.session() as s:
            s.run(
                """
                MERGE (c:Chunk {chunk_id: $chunk_id})
                SET c.text     = $text,
                    c.source   = $source,
                    c.doc_type = $doc_type,
                    c.doc_id   = $doc_id
                """,
                chunk_id=chunk_id, text=text, source=source,
                doc_type=doc_type, doc_id=doc_id,
            )

    def upsert_chunks_batch(self, chunks: list[dict]) -> None:
        """
        Batch upsert Chunk nodes. Faster than individual upsert_chunk calls.

        Each dict in chunks must have: chunk_id, text, source, doc_type, doc_id
        """
        with self._driver.session() as s:
            s.run(
                """
                UNWIND $chunks AS row
                MERGE (c:Chunk {chunk_id: row.chunk_id})
                SET c.text     = row.text,
                    c.source   = row.source,
                    c.doc_type = row.doc_type,
                    c.doc_id   = row.doc_id
                """,
                chunks=chunks,
            )

    def upsert_entity(self, name: str, label: str, chunk_id: str) -> None:
        """
        Upsert an Entity node.
        Appends chunk_id to the chunk_ids list if not already present.
        Also creates the (:Chunk)-[:CONTAINS]->(:Entity) structural edge.
        """
        with self._driver.session() as s:
            s.run(
                """
                MERGE (e:Entity {name: $name})
                ON CREATE SET e.label     = $label,
                              e.chunk_ids = [$chunk_id]
                ON MATCH  SET e.label     = $label,
                              e.chunk_ids = CASE
                                  WHEN $chunk_id IN e.chunk_ids THEN e.chunk_ids
                                  ELSE e.chunk_ids + [$chunk_id]
                              END
                WITH e
                MATCH (c:Chunk {chunk_id: $chunk_id})
                MERGE (c)-[:CONTAINS]->(e)
                """,
                name=name, label=label, chunk_id=chunk_id,
            )

    def upsert_entities_batch(self, entities: list[dict]) -> None:
        """
        Batch upsert Entity nodes + CONTAINS edges.
        Each dict: {name, label, chunk_id}
        """
        with self._driver.session() as s:
            s.run(
                """
                UNWIND $entities AS row
                MERGE (e:Entity {name: row.name})
                ON CREATE SET e.label     = row.label,
                              e.chunk_ids = [row.chunk_id]
                ON MATCH  SET e.label     = row.label,
                              e.chunk_ids = CASE
                                  WHEN row.chunk_id IN e.chunk_ids THEN e.chunk_ids
                                  ELSE e.chunk_ids + [row.chunk_id]
                              END
                """,
                entities=entities,
            )
            # CONTAINS edges in a second pass (UNWIND + MATCH)
            s.run(
                """
                UNWIND $entities AS row
                MATCH (e:Entity {name: row.name})
                MATCH (c:Chunk  {chunk_id: row.chunk_id})
                MERGE (c)-[:CONTAINS]->(e)
                """,
                entities=entities,
            )

    def upsert_relation(
        self,
        source_name: str,
        relation_type: str,
        target_name: str,
        chunk_id: str,
        context: str = "",
    ) -> None:
        """
        Upsert a typed relationship between two Entity nodes.
        relation_type should be UPPER_SNAKE_CASE (e.g. DEPENDS_ON).
        """
        # Cypher does not allow parameterised relationship types — use safe formatting
        safe_rel = self._safe_rel_type(relation_type)
        with self._driver.session() as s:
            s.run(
                f"""
                MATCH (src:Entity {{name: $source}})
                MATCH (tgt:Entity {{name: $target}})
                MERGE (src)-[r:{safe_rel} {{chunk_id: $chunk_id}}]->(tgt)
                SET r.context = $context
                """,
                source=source_name, target=target_name,
                chunk_id=chunk_id, context=context[:500],
            )

    def upsert_relations_batch(self, relations: list[dict]) -> None:
        """
        Batch upsert relations. Groups by relation_type to minimise Cypher calls.
        Each dict: {source, relation_type, target, chunk_id, context}
        """
        from collections import defaultdict
        by_type: dict[str, list[dict]] = defaultdict(list)
        for r in relations:
            by_type[r["relation_type"]].append(r)

        with self._driver.session() as s:
            for rel_type, batch in by_type.items():
                safe_rel = self._safe_rel_type(rel_type)
                s.run(
                    f"""
                    UNWIND $rels AS row
                    MATCH (src:Entity {{name: row.source}})
                    MATCH (tgt:Entity {{name: row.target}})
                    MERGE (src)-[r:{safe_rel} {{chunk_id: row.chunk_id}}]->(tgt)
                    SET r.context = row.context
                    """,
                    rels=[
                        {
                            "source":   r["source"],
                            "target":   r["target"],
                            "chunk_id": r["chunk_id"],
                            "context":  r.get("context", "")[:500],
                        }
                        for r in batch
                    ],
                )

    # ── Query operations ────────────────────────────────────────────────────

    def entity_search(
        self,
        entity_names: list[str],
        max_hops: int = 2,
        top_k: int = 10,
    ) -> list[GraphSearchResult]:
        """
        Core GraphRAG query: traverse the graph from matched entity nodes.

        Returns paths from matched start entities to related entities,
        along with the source chunk text for each endpoint.

        This is the method called by tools/search_tools.py → graph_search().
        """
        max_hops = min(max_hops, 3)  # safety cap
        with self._driver.session() as s:
            result = s.run(
                f"""
                MATCH path = (start:Entity)-[*1..{max_hops}]-(end:Entity)
                WHERE start.name IN $names
                  AND start <> end
                WITH start, end,
                     length(path) AS hops,
                     [r IN relationships(path) | type(r)] AS rel_types
                UNWIND end.chunk_ids AS cid
                MATCH (c:Chunk {{chunk_id: cid}})
                RETURN
                    start.name  AS start_entity,
                    rel_types   AS relations,
                    end.name    AS end_entity,
                    c.chunk_id  AS chunk_id,
                    c.text      AS text,
                    hops        AS hop_count
                ORDER BY hops ASC, start_entity ASC
                LIMIT $limit
                """,
                names=entity_names,
                limit=top_k,
            )
            return [
                GraphSearchResult(
                    start_entity=r["start_entity"],
                    relations=list(r["relations"]),
                    end_entity=r["end_entity"],
                    chunk_id=r["chunk_id"],
                    text=r["text"],
                    hop_count=r["hop_count"],
                )
                for r in result
            ]

    def fulltext_entity_search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        Fallback: full-text search on Chunk.text when entity extraction
        returns no results (e.g. query has no named entities).
        Uses the chunk_text_ft full-text index.
        """
        with self._driver.session() as s:
            result = s.run(
                """
                CALL db.index.fulltext.queryNodes('chunk_text_ft', $query)
                YIELD node, score
                RETURN node.chunk_id AS chunk_id,
                       node.text     AS text,
                       node.source   AS source,
                       score
                ORDER BY score DESC
                LIMIT $limit
                """,
                query=query,
                limit=top_k,
            )
            return [dict(r) for r in result]

    def get_entity_neighbours(self, entity_name: str, rel_type: str | None = None) -> list[dict]:
        """
        Return immediate (1-hop) neighbours of an entity.
        Optionally filter by relationship type.
        Useful for the Streamlit graph visualiser.
        """
        if rel_type:
            safe_rel = self._safe_rel_type(rel_type)
            cypher = f"""
                MATCH (e:Entity {{name: $name}})-[r:{safe_rel}]-(n:Entity)
                RETURN n.name AS neighbour, type(r) AS relation, r.context AS context
            """
        else:
            cypher = """
                MATCH (e:Entity {name: $name})-[r]-(n:Entity)
                RETURN n.name AS neighbour, type(r) AS relation, r.context AS context
            """
        with self._driver.session() as s:
            result = s.run(cypher, name=entity_name)
            return [dict(r) for r in result]

    def get_chunks_for_entity(self, entity_name: str) -> list[dict]:
        """Return all chunks that contain a given entity."""
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (e:Entity {name: $name})<-[:CONTAINS]-(c:Chunk)
                RETURN c.chunk_id AS chunk_id, c.text AS text, c.source AS source
                """,
                name=entity_name,
            )
            return [dict(r) for r in result]

    def get_all_entities(self, label: str | None = None, limit: int = 500) -> list[dict]:
        """
        Return all entities (for the graph visualiser).
        Optionally filter by NER label (PERSON, ORG, etc.).
        """
        if label:
            cypher = "MATCH (e:Entity {label: $label}) RETURN e.name AS name, e.label AS label LIMIT $limit"
        else:
            cypher = "MATCH (e:Entity) RETURN e.name AS name, e.label AS label LIMIT $limit"
        with self._driver.session() as s:
            result = s.run(cypher, label=label, limit=limit)
            return [dict(r) for r in result]

    def get_all_relations(self, limit: int = 1000) -> list[dict]:
        """
        Return all entity-relation-entity triples (for the graph visualiser).
        """
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (src:Entity)-[r]->(tgt:Entity)
                WHERE type(r) <> 'CONTAINS'
                RETURN src.name AS source, type(r) AS relation, tgt.name AS target
                LIMIT $limit
                """,
                limit=limit,
            )
            return [dict(r) for r in result]

    # ── Health & stats ──────────────────────────────────────────────────────

    def health_check(self) -> dict:
        """
        Returns a health dict. Used by the FastAPI /health endpoint.
        Raises on connection failure.
        """
        try:
            with self._driver.session() as s:
                version = s.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]
            stats = self.get_stats()
            return {
                "status":        "ok",
                "neo4j_version": version,
                "chunks":        stats.chunk_count,
                "entities":      stats.entity_count,
                "relations":     stats.relation_count,
            }
        except Exception as e:
            return {"status": "error", "detail": str(e)}

    def get_stats(self) -> GraphStats:
        """Return node/edge counts and index status."""
        with self._driver.session() as s:
            chunks   = s.run("MATCH (c:Chunk)  RETURN count(c) AS n").single()["n"]
            entities = s.run("MATCH (e:Entity) RETURN count(e) AS n").single()["n"]
            rels     = s.run(
                "MATCH (:Entity)-[r]->(:Entity) WHERE type(r) <> 'CONTAINS' RETURN count(r) AS n"
            ).single()["n"]
            indexes  = [
                r["name"]
                for r in s.run("SHOW INDEXES YIELD name, state WHERE state = 'ONLINE'")
            ]
            version  = s.run("CALL dbms.components() YIELD versions RETURN versions[0] AS v").single()["v"]

        return GraphStats(
            chunk_count=chunks,
            entity_count=entities,
            relation_count=rels,
            index_status=indexes,
            neo4j_version=version,
        )

    # ── Maintenance ─────────────────────────────────────────────────────────

    def wipe(self, confirm: bool = False) -> None:
        """
        Delete ALL nodes and relationships.
        Requires confirm=True to prevent accidental wipes.
        Use for test resets only.
        """
        if not confirm:
            raise ValueError("Pass confirm=True to wipe the graph store. This is irreversible.")
        with self._driver.session() as s:
            s.run("MATCH (n) DETACH DELETE n")
        print("  [graph_store] ⚠ All nodes and relationships deleted.")

    def delete_by_doc_id(self, doc_id: str) -> int:
        """
        Delete all Chunk nodes (and their CONTAINS edges) for a given doc_id.
        Entity nodes are NOT deleted — they may be referenced by other docs.
        Returns the number of chunks deleted.
        """
        with self._driver.session() as s:
            result = s.run(
                """
                MATCH (c:Chunk {doc_id: $doc_id})
                WITH c, c.chunk_id AS cid
                DETACH DELETE c
                RETURN count(cid) AS deleted
                """,
                doc_id=doc_id,
            )
            return result.single()["deleted"]

    # ── Helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _safe_rel_type(rel_type: str) -> str:
        """
        Sanitise relationship type for safe Cypher interpolation.
        Only allows uppercase letters, digits, and underscores.
        This prevents Cypher injection via relation_type parameter.
        """
        import re
        safe = re.sub(r"[^A-Z0-9_]", "_", rel_type.upper())
        if not safe or safe[0].isdigit():
            safe = "REL_" + safe
        return safe


# ---------------------------------------------------------------------------
# Factory — singleton pattern
# ---------------------------------------------------------------------------

_graph_store_instance: Neo4jGraphStore | None = None


def get_graph_store() -> Neo4jGraphStore:
    """
    Return the singleton Neo4jGraphStore instance.
    Reads connection details from environment variables.

    Call init_schema() once on startup (done by StoreManager).
    """
    global _graph_store_instance
    if _graph_store_instance is None:
        _graph_store_instance = Neo4jGraphStore(
            uri=os.environ.get("NEO4J_URI",      "bolt://localhost:7687"),
            user=os.environ.get("NEO4J_USER",     "neo4j"),
            password=os.environ.get("NEO4J_PASSWORD", "password"),
        )
    return _graph_store_instance