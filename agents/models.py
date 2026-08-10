"""
agents/models.py

Structured response contracts for all agents.

Every agent returns an AgentResponse. The orchestrator wraps the
sub-agent's output in a RouteDecision before it delegates.
FastAPI and the Streamlit UI consume these models directly.

Design principle:
  Agents produce text. This module parses that text into typed
  objects so the rest of the system never does string slicing.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class RetrievalMode(str, Enum):
    FLAT    = "flat_rag"
    GRAPH   = "graph_rag"
    HYBRID  = "hybrid"
    UNKNOWN = "unknown"


class Confidence(str, Enum):
    HIGH    = "high"
    MEDIUM  = "medium"
    LOW     = "low"
    UNKNOWN = "unknown"


# ---------------------------------------------------------------------------
# Core response model
# ---------------------------------------------------------------------------

@dataclass
class AgentResponse:
    """
    Unified response returned by every sub-agent.

    Fields populated by the agent's text output parser.
    If parsing fails, raw_text contains the full unparsed response.
    """
    answer:          str                   # the grounded answer
    sources:         list[str]             # chunk_ids cited
    confidence:      Confidence            # high | medium | low
    retrieval_mode:  RetrievalMode         # which agent produced this
    graph_path:      str | None   = None   # GraphRAG only — entity chain
    high_conf_ids:   list[str]    = field(default_factory=list)  # hybrid: source=both
    raw_text:        str          = ""     # original LLM output (always kept)
    metadata:        dict[str, Any] = field(default_factory=dict)

    @property
    def has_answer(self) -> bool:
        return bool(self.answer and self.answer.strip())

    def to_dict(self) -> dict:
        return {
            "answer":         self.answer,
            "sources":        self.sources,
            "confidence":     self.confidence.value,
            "retrieval_mode": self.retrieval_mode.value,
            "graph_path":     self.graph_path,
            "high_conf_ids":  self.high_conf_ids,
            "metadata":       self.metadata,
        }


@dataclass
class RouteDecision:
    """
    The orchestrator's routing decision before delegation.
    Stored in session state for observability.
    """
    query:           str
    chosen_agent:    str           # flat_rag_agent | graph_rag_agent | hybrid_agent
    reasoning:       str           # why this agent was chosen
    entity_signals:  list[str]     # entities found in query
    relational_flag: bool          # True if relational signals detected


# ---------------------------------------------------------------------------
# Response parsers
# ---------------------------------------------------------------------------

class ResponseParser:
    """
    Parses LLM text output from sub-agents into AgentResponse objects.

    Each agent follows a structured output format defined in its system prompt.
    This parser extracts the fields using lightweight regex — no JSON required.
    """

    # Flat RAG format:
    #   Answer: <text>
    #   Sources: chunk_id1, chunk_id2
    #   Confidence: high | medium | low

    # GraphRAG format:
    #   Answer: <text>
    #   Graph path: A → [REL] → B → [REL] → C
    #   Sources: chunk_id1
    #   Confidence: high

    # Hybrid format:
    #   Answer: <text>
    #   High-confidence results (source=both): chunk_id1, chunk_id2
    #   Sources: chunk_id1, chunk_id2, chunk_id3
    #   Confidence: medium

    @classmethod
    def parse(cls, text: str, mode: RetrievalMode) -> AgentResponse:
        """Parse LLM output text into a typed AgentResponse."""
        answer          = cls._extract_field(text, "Answer")
        confidence_str  = cls._extract_field(text, "Confidence").lower()
        sources_str     = cls._extract_field(text, "Sources")
        graph_path      = cls._extract_field(text, "Graph path") or None
        high_conf_str   = cls._extract_field(text, r"High-confidence results \(source=both\)")

        confidence = Confidence(confidence_str) if confidence_str in Confidence._value2member_map_ else Confidence.UNKNOWN
        sources    = [s.strip() for s in sources_str.split(",") if s.strip()] if sources_str else []
        high_conf  = [s.strip() for s in high_conf_str.split(",") if s.strip()] if high_conf_str else []

        # If structured parsing fails, use the full text as the answer
        if not answer:
            answer = text.strip()

        return AgentResponse(
            answer=answer,
            sources=sources,
            confidence=confidence,
            retrieval_mode=mode,
            graph_path=graph_path,
            high_conf_ids=high_conf,
            raw_text=text,
        )

    @staticmethod
    def _extract_field(text: str, label: str) -> str:
        """
        Extract a field value from structured agent output.
        Handles single-line and multi-line values (up to next label).
        """
        pattern = rf"(?:^|\n){label}:\s*(.+?)(?=\n[A-Z][^:\n]*:|$)"
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            return match.group(1).strip()
        return ""

    @classmethod
    def parse_route_decision(cls, text: str, query: str) -> RouteDecision:
        """
        Parse the orchestrator's internal routing reasoning.
        The orchestrator is not required to output a structured format —
        we infer the chosen agent from which sub-agent name appears in the text.
        """
        text_lower = text.lower()
        if "flat_rag_agent" in text_lower or "flat rag" in text_lower:
            chosen = "flat_rag_agent"
        elif "graph_rag_agent" in text_lower or "graph rag" in text_lower or "graphrag" in text_lower:
            chosen = "graph_rag_agent"
        elif "hybrid_agent" in text_lower or "hybrid" in text_lower:
            chosen = "hybrid_agent"
        else:
            chosen = "unknown"

        # Extract entity names (capitalised words, 2+ chars)
        entities = re.findall(r"\b[A-Z][a-z]+(?:\s[A-Z][a-z]+)*\b", query)

        # Relational signal detection
        relational_signals = [
            "connect", "relate", "depend", "impact", "between", "through",
            "leads to", "caused by", "affect", "link", "path", "chain", "owned by"
        ]
        has_relational = any(sig in query.lower() for sig in relational_signals)

        return RouteDecision(
            query=query,
            chosen_agent=chosen,
            reasoning=text,
            entity_signals=entities,
            relational_flag=has_relational,
        )