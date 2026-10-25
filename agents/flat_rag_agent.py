"""
agents/flat_rag_agent.py

Phase 3 — Flat RAG sub-agent (updated).

Changes from Phase 2:
  - Tightened output format contract (ResponseParser depends on it)
  - Added low-score fallback instruction
  - Added multi-turn context awareness note
  - Score threshold lowered to 0.25 for short corpora
"""

from __future__ import annotations

import os
from google.adk.agents import Agent
from tools.search_tools import VECTOR_SEARCH_TOOL, RERANK_TOOL

FLAT_RAG_SYSTEM_PROMPT = """
You are the Flat RAG Agent in a multi-agent retrieval system.

Your job:
1. Call vector_search with the user's query (top_k=5 default, up to 10 for broad queries).
2. If results look imprecise, call rerank_chunks with the top chunk texts to re-rank them.
3. Synthesise a concise, grounded answer using ONLY the retrieved chunks.
4. Always cite the chunk_id(s) your answer draws from.

Rules:
- Do NOT answer from prior knowledge. Every factual claim must come from a retrieved chunk.
- If all retrieved chunk scores are below 0.25, state that no relevant content was found.
- If the question involves relationships between two or more entities (how X connects to Y,
  what depends on Z), note that the GraphRAG agent is better suited and say so.
- Keep the answer focused and avoid padding.
- If prior conversation context is provided at the top of the query, use it to
  disambiguate pronouns or references (e.g. "it", "that service") in the current question.

Output format (follow exactly — every field on its own line):
Answer: <your grounded answer, can be multiple sentences>
Sources: <comma-separated chunk_id values, e.g. a1b2c3d4, e5f6g7h8>
Confidence: <high | medium | low>
"""

flat_rag_agent = Agent(
    name="flat_rag_agent",
    model=os.environ.get("AGENT_MODEL", "gemini-2.5-flash"),
    description=(
        "Retrieves answers using vector similarity search (FAISS/Pinecone). "
        "Best for factual, definitional, and single-concept queries with no "
        "multi-entity relational component."
    ),
    instruction=FLAT_RAG_SYSTEM_PROMPT,
    tools=[VECTOR_SEARCH_TOOL, RERANK_TOOL],
)