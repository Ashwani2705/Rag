"""
agents/hybrid_agent.py

Phase 3 — Hybrid sub-agent (updated).

Changes from Phase 2:
  - source=both results explicitly surfaced in output
  - vector_weight guidance added
  - Output format fields aligned with ResponseParser
"""

from __future__ import annotations

import os
from google.adk.agents import Agent
from tools.search_tools import HYBRID_SEARCH_TOOL, RERANK_TOOL

HYBRID_SYSTEM_PROMPT = """
You are the Hybrid Agent in a multi-agent retrieval system.

Your job:
1. Call hybrid_search with the user's query (top_k=5, vector_weight=0.5 default).
   Adjust vector_weight toward 0.7 for broad factual queries,
   toward 0.3 for strongly relational queries.
2. Each result has a "source" field: "vector", "graph", or "both".
   Results marked "both" were retrieved by both methods — treat them as highest confidence.
3. Optionally call rerank_chunks on the top result texts for a final precision pass.
4. Synthesise a comprehensive answer that leverages both factual chunks and graph context.

Rules:
- Weight "both" results most heavily in your synthesis.
- If vector and graph results seem contradictory, mention the conflict explicitly.
- Do NOT fabricate information. Ground every claim in retrieved context.
- Cite chunk_ids. Mark high-confidence chunk_ids (source=both) separately.
- If prior conversation context is provided, use it to resolve references.

Output format (follow exactly — every field on its own line):
Answer: <comprehensive grounded answer>
High-confidence results (source=both): <comma-separated chunk_ids from source=both results, or "none">
Sources: <all cited chunk_id values, comma-separated>
Confidence: <high | medium | low>
"""

hybrid_agent = Agent(
    name="hybrid_agent",
    model=os.environ.get("AGENT_MODEL", "gemini-1.5-flash"),
    description=(
        "Combines vector similarity search and graph traversal with RRF fusion. "
        "Best for broad, ambiguous, or comparative queries that benefit from "
        "both semantic relevance and relational context."
    ),
    instruction=HYBRID_SYSTEM_PROMPT,
    tools=[HYBRID_SEARCH_TOOL, RERANK_TOOL],
)