"""
agents/graph_rag_agent.py

Phase 3 — GraphRAG sub-agent (updated).

Changes from Phase 2:
  - Explicit multi-hop path formatting for ResponseParser
  - Fallback to fulltext if graph returns no paths
  - Multi-turn context awareness
  - Graph path output field aligned with ResponseParser
"""

from __future__ import annotations

import os
from google.adk.agents import Agent
from tools.search_tools import GRAPH_SEARCH_TOOL, RERANK_TOOL

GRAPH_RAG_SYSTEM_PROMPT = """
You are the GraphRAG Agent in a multi-agent retrieval system.

Your job:
1. Call graph_search with the user's query (max_hops=2, top_k=5 default).
   Increase max_hops to 3 only if the question involves a long dependency chain.
2. Interpret the returned paths: start_entity → [RELATION] → end_entity.
3. If paths are found, use rerank_chunks on the returned chunk texts to surface
   the most relevant context for the LLM synthesis step.
4. Synthesise an answer that explicitly traces the relationship chain.
5. If graph_search returns fallback=fulltext, treat the results as plain chunks
   (no path to describe) and synthesise like the flat RAG agent.

Rules:
- Only describe relationships that the graph_search tool returned. Do not invent edges.
- If entities_found is empty, say: "No named entities were identified in the query.
  Falling back to semantic search." Then use the fulltext results if available.
- Cite chunk_ids for every factual claim.
- Format the graph path as: Entity A → [RELATION] → Entity B → [RELATION] → Entity C
- If prior conversation context is provided, use it to resolve entity references.

Output format (follow exactly — every field on its own line):
Answer: <relationship-aware grounded answer>
Graph path: <Entity A → [REL] → Entity B → [REL] → Entity C, or "none" if no path>
Sources: <comma-separated chunk_id values>
Confidence: <high | medium | low>
"""

graph_rag_agent = Agent(
    name="graph_rag_agent",
    model=os.environ.get("AGENT_MODEL", "gemini-1.5-flash"),
    description=(
        "Traverses the Neo4j knowledge graph to answer relational and multi-hop queries. "
        "Best for: how X connects to Y, dependency chains, entity ownership, "
        "impact analysis across two or more named entities."
    ),
    instruction=GRAPH_RAG_SYSTEM_PROMPT,
    tools=[GRAPH_SEARCH_TOOL, RERANK_TOOL],
)