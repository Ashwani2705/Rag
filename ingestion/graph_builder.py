"""
ingestion/graph_builder.py

Phase 1 — Step 4: Knowledge graph construction.

Takes Chunk objects → extracts entities + relations → writes to Neo4j.

Schema:
  Nodes:
    (:Chunk   {chunk_id, text, source, doc_type, ...metadata})
    (:Entity  {name, label, chunk_ids: [...]})

  Edges:
    (:Chunk)-[:CONTAINS]->(:Entity)
    (:Entity)-[:RELATES_TO {relation, chunk_id, context}]->(:Entity)

Extraction strategy (two modes, set via GRAPH_EXTRACTION_MODE):
  "spacy"  — fast, offline, good for ORG/PERSON/GPE/PRODUCT entities
             + co-occurrence-based relations (entities in same sentence → related)
  "llm"    — slower, uses Gemini Flash, extracts explicit typed relations
             (e.g. "depends_on", "part_of", "created_by")
             Recommended for demo quality. Use spacy for CI/batch runs.

Critical constraint:
  Every Entity node stores chunk_ids as a list property.
  graph_search in tools/search_tools.py relies on this to fetch chunk text
  after a graph traversal. If this mapping breaks, GraphRAG returns empty results.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Literal

from ingestion.chunker import Chunk


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    name: str
    label: str          # PERSON, ORG, PRODUCT, GPE, CONCEPT, etc.
    chunk_id: str


@dataclass
class Relation:
    source: str         # entity name
    relation: str       # e.g. RELATES_TO, DEPENDS_ON, PART_OF, CREATED_BY
    target: str         # entity name
    chunk_id: str
    context: str        # the sentence/phrase that evidences this relation


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------

class GraphBuilder:
    """
    Builds a Neo4j knowledge graph from Chunk objects.

    Args:
        extraction_mode : "spacy" (fast) or "llm" (higher quality)
        batch_size      : number of chunks to process per Neo4j transaction
    """

    def __init__(
        self,
        extraction_mode: Literal["spacy", "llm"] = "spacy",
        batch_size: int = 50,
    ):
        self.extraction_mode = extraction_mode
        self.batch_size = batch_size
        self._driver = self._connect()

    # ── Public API ──────────────────────────────────────────────────────────

    def build(self, chunks: list[Chunk]) -> None:
        """
        Full pipeline: extract entities + relations from all chunks,
        then write to Neo4j in batches.
        """
        print(f"  [graph_builder] extraction_mode={self.extraction_mode}, {len(chunks)} chunks")
        self._ensure_schema()

        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            print(f"  [graph_builder] processing chunks {i}–{i + len(batch) - 1}")
            entities_all: list[Entity] = []
            relations_all: list[Relation] = []

            for chunk in batch:
                entities, relations = self._extract(chunk)
                entities_all.extend(entities)
                relations_all.extend(relations)

            self._write_chunks(batch)
            self._write_entities(entities_all)
            self._write_relations(relations_all)

        print(f"  [graph_builder] ✓ graph build complete.")
        self._driver.close()

    def close(self) -> None:
        self._driver.close()

    # ── Extraction ──────────────────────────────────────────────────────────

    def _extract(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        if self.extraction_mode == "llm":
            return self._extract_llm(chunk)
        return self._extract_spacy(chunk)

    def _extract_spacy(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        """
        Uses spaCy NER for entity extraction.
        Relations: any two entities appearing in the same sentence are linked
        with a generic RELATES_TO edge + the sentence as context.

        Constraint: Only extracts surface-form relations (co-occurrence).
        For typed relations (depends_on, part_of), switch to "llm" mode.
        """
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
        except Exception:
            raise RuntimeError(
                "spaCy model not found. Run: python -m spacy download en_core_web_sm"
            )

        VALID_LABELS = {"PERSON", "ORG", "PRODUCT", "GPE", "EVENT", "WORK_OF_ART", "LAW", "FAC"}
        doc = nlp(chunk.text)

        entities: list[Entity] = []
        relations: list[Relation] = []

        for sent in doc.sents:
            sent_entities = [
                ent for ent in sent.ents if ent.label_ in VALID_LABELS
            ]
            # Deduplicate within sentence
            unique = {e.text.strip(): e.label_ for e in sent_entities if e.text.strip()}
            sent_entity_objs = [Entity(name=k, label=v, chunk_id=chunk.chunk_id) for k, v in unique.items()]
            entities.extend(sent_entity_objs)

            # Co-occurrence relations — pair every entity with every other in the same sentence
            names = list(unique.keys())
            for j in range(len(names)):
                for k in range(j + 1, len(names)):
                    relations.append(Relation(
                        source=names[j],
                        relation="RELATES_TO",
                        target=names[k],
                        chunk_id=chunk.chunk_id,
                        context=sent.text.strip()[:300],  # cap context at 300 chars
                    ))

        return entities, relations

    def _extract_llm(self, chunk: Chunk) -> tuple[list[Entity], list[Relation]]:
        """
        Uses Gemini Flash to extract typed entities and explicit relations.
        Prompts the LLM to return structured JSON.

        Output contract (LLM must follow exactly):
        {
          "entities": [{"name": "...", "label": "PERSON|ORG|PRODUCT|CONCEPT|..."}],
          "relations": [{"source": "...", "relation": "depends_on|part_of|created_by|...", "target": "...", "context": "..."}]
        }

        Constraint: Adds ~0.5–1s latency per chunk. Use for final demo runs,
        not for large batch ingestion.
        """
        import google.generativeai as genai
        genai.configure(api_key=os.environ["gemini_api_key"])
        model = genai.GenerativeModel("gemini-2.5-flash")

        prompt = f"""Extract entities and relationships from the following text.

Return ONLY a JSON object with this exact structure — no markdown, no explanation:
{{
  "entities": [
    {{"name": "entity name", "label": "PERSON|ORG|PRODUCT|CONCEPT|GPE|EVENT"}}
  ],
  "relations": [
    {{"source": "entity1", "relation": "depends_on|part_of|created_by|relates_to|used_by|impacts", "target": "entity2", "context": "the sentence that shows this relation"}}
  ]
}}

Rules:
- Only extract entities that appear explicitly in the text
- Only extract relations that are explicitly stated, not implied
- Context must be a direct quote from the text (max 200 chars)
- If no entities or relations found, return empty lists

Text:
\"\"\"
{chunk.text[:3000]}
\"\"\"
"""
        try:
            resp = model.generate_content(prompt)
            raw = resp.text.strip().replace("```json", "").replace("```", "").strip()
            data = json.loads(raw)
        except (json.JSONDecodeError, Exception) as e:
            print(f"  [graph_builder] LLM extraction failed for chunk {chunk.chunk_id}: {e}")
            return [], []

        entities = [
            Entity(name=e["name"], label=e.get("label", "CONCEPT"), chunk_id=chunk.chunk_id)
            for e in data.get("entities", [])
            if e.get("name")
        ]
        relations = [
            Relation(
                source=r["source"],
                relation=r.get("relation", "RELATES_TO").upper().replace(" ", "_"),
                target=r["target"],
                chunk_id=chunk.chunk_id,
                context=r.get("context", "")[:300],
            )
            for r in data.get("relations", [])
            if r.get("source") and r.get("target")
        ]
        return entities, relations

    # ── Neo4j writes ────────────────────────────────────────────────────────

    def _write_chunks(self, chunks: list[Chunk]) -> None:
        """
        Upsert Chunk nodes. Uses MERGE so re-ingestion is idempotent.
        """
        with self._driver.session() as session:
            session.execute_write(
                lambda tx: [
                    tx.run(
                        """
                        MERGE (c:Chunk {chunk_id: $chunk_id})
                        SET c.text     = $text,
                            c.source   = $source,
                            c.doc_type = $doc_type,
                            c.doc_id   = $doc_id
                        """,
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        source=chunk.source,
                        doc_type=chunk.doc_type,
                        doc_id=chunk.doc_id,
                    )
                    for chunk in chunks
                ]
            )

    def _write_entities(self, entities: list[Entity]) -> None:
        """
        Upsert Entity nodes. chunk_ids is a list — MERGE preserves existing
        chunk_ids and appends new ones so the same entity can point to
        multiple source chunks.
        """
        with self._driver.session() as session:
            for entity in entities:
                session.execute_write(
                    lambda tx, e=entity: tx.run(
                        """
                        MERGE (ent:Entity {name: $name})
                        ON CREATE SET ent.label     = $label,
                                      ent.chunk_ids = [$chunk_id]
                        ON MATCH  SET ent.label     = $label,
                                      ent.chunk_ids = CASE
                                          WHEN $chunk_id IN ent.chunk_ids THEN ent.chunk_ids
                                          ELSE ent.chunk_ids + [$chunk_id]
                                      END
                        """,
                        name=e.name,
                        label=e.label,
                        chunk_id=e.chunk_id,
                    )
                )

    def _write_relations(self, relations: list[Relation]) -> None:
        """
        Upsert RELATES_TO (or typed) edges between Entity nodes.
        Skips if either endpoint entity doesn't exist (shouldn't happen in practice).
        """
        with self._driver.session() as session:
            for rel in relations:
                session.execute_write(
                    lambda tx, r=rel: tx.run(
                        f"""
                        MATCH (src:Entity {{name: $source}})
                        MATCH (tgt:Entity {{name: $target}})
                        MERGE (src)-[r:{rel.relation} {{chunk_id: $chunk_id}}]->(tgt)
                        SET r.context = $context
                        """,
                        source=rel.source,
                        target=rel.target,
                        chunk_id=rel.chunk_id,
                        context=rel.context,
                    )
                )

    # ── Schema / connection ──────────────────────────────────────────────────

    def _ensure_schema(self) -> None:
        """
        Create indexes on first run. Idempotent — safe to call every ingestion.
        Indexes on Entity.name and Chunk.chunk_id are critical for Cypher query speed.
        """
        with self._driver.session() as session:
            session.run("CREATE INDEX entity_name IF NOT EXISTS FOR (e:Entity) ON (e.name)")
            session.run("CREATE INDEX chunk_id    IF NOT EXISTS FOR (c:Chunk)  ON (c.chunk_id)")
        print("  [graph_builder] schema indexes ensured.")

    def _connect(self):
        from neo4j import GraphDatabase
        driver = GraphDatabase.driver(
            os.environ.get("NEO4J_URI",      "bolt://localhost:7687"),
            auth=(
                os.environ.get("NEO4J_USER",     "neo4j"),
                os.environ.get("NEO4J_PASSWORD", "password"),
            ),
        )
        driver.verify_connectivity()
        print("  [graph_builder] connected to Neo4j.")
        return driver