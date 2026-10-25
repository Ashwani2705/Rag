"""
agents/runner.py

Phase 3 — Agent runner.

The AgentRunner is the single callable interface for the entire agent system.
FastAPI, the Streamlit UI, and CLI scripts all call runner.run(query, ...).

Responsibilities:
  1. Initialise StoreManager on first call (lazy, thread-safe)
  2. Get or create an ADK session
  3. Inject prior conversation context into the query
  4. Run the orchestrator agent via ADK Runner
  5. Parse the raw LLM text into a typed AgentResponse
  6. Record the turn in SessionManager
  7. Return the AgentResponse

Flow:
  caller → AgentRunner.run()
             ├── StoreManager.initialise()   (once)
             ├── SessionManager.get_or_create()
             ├── build_message(query + context)
             ├── ADK Runner.run_async()
             │     └── orchestrator_agent
             │           └── flat_rag_agent | graph_rag_agent | hybrid_agent
             │                 └── FunctionTools (vector_search, graph_search, ...)
             ├── ResponseParser.parse()
             └── SessionManager.record_turn()
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncIterator

from agents.models import (
    AgentResponse, RetrievalMode, Confidence, ResponseParser, RouteDecision
)
from agents.session_manager import get_session_manager


# ---------------------------------------------------------------------------
# Agent Runner
# ---------------------------------------------------------------------------

class AgentRunner:
    """
    Async runner for the full orchestrator → sub-agent pipeline.

    Usage (async):
        runner = AgentRunner()
        response = await runner.run("What is GraphRAG?", user_id="u1")

    Usage (sync, for CLI):
        response = runner.run_sync("What is GraphRAG?", user_id="u1")
    """

    APP_NAME = "rag-graphrag-demo"

    def __init__(self):
        self._store_manager = None
        self._adk_runner    = None
        self._lock          = asyncio.Lock()
        self._ready         = False

    # ── Initialisation ───────────────────────────────────────────────────────

    async def _ensure_ready(self) -> None:
        """
        Lazy initialisation — runs once on the first call.
        Protected by an async lock so concurrent startup calls are safe.
        """
        if self._ready:
            return

        async with self._lock:
            if self._ready:
                return   # double-check after acquiring lock

            print("[runner] initialising stores and agents...")

            # 1. Initialise stores
            from stores.store_manager import get_store_manager
            self._store_manager = get_store_manager()
            self._store_manager.initialise()

            # 2. Build ADK Runner with the orchestrator agent
            from google.adk.runners import Runner
            from agents.orchestrator_agent import orchestrator_agent
            from agents.session_manager import get_session_manager

            sm = get_session_manager()
            self._adk_runner = Runner(
                agent=orchestrator_agent,
                app_name=self.APP_NAME,
                session_service=sm.adk_service,
            )

            self._ready = True
            print("[runner] [OK] ready.")

    # ── Main run interface ───────────────────────────────────────────────────

    async def run(
        self,
        query: str,
        user_id: str = "default_user",
        session_id: str | None = None,
        context_turns: int = 3,
    ) -> AgentResponse:
        """
        Run a query through the full agent pipeline.

        Args:
            query         : the user's question
            user_id       : identifies the user (for session scoping)
            session_id    : existing session to continue, or None for a new one
            context_turns : how many prior turns to inject as context (default 3)

        Returns:
            AgentResponse with typed answer, sources, confidence, retrieval_mode
        """
        await self._ensure_ready()

        session_mgr = get_session_manager()
        session_id  = await session_mgr.get_or_create(user_id, session_id)

        # Build the message with optional prior context
        context = session_mgr.get_context_string(session_id, last_n=context_turns)
        full_query = self._build_query(query, context)

        # Run through ADK
        raw_text, route_decision = await self._run_adk(full_query, user_id, session_id, query)

        # Parse structured response
        mode     = self._infer_mode(route_decision)
        response = ResponseParser.parse(raw_text, mode)
        response.metadata["session_id"] = session_id
        response.metadata["user_id"]    = user_id

        # Record turn
        await session_mgr.record_turn(session_id, query, response, route_decision)

        return response

    async def run_stream(
        self,
        query: str,
        user_id: str = "default_user",
        session_id: str | None = None,
    ) -> AsyncIterator[str]:
        """
        Streaming variant — yields text chunks as they arrive from the LLM.
        Useful for the Streamlit UI to show progressive output.

        Note: The final AgentResponse is NOT returned here — use run() for that.
        This is display-only streaming.
        """
        await self._ensure_ready()

        session_mgr = get_session_manager()
        session_id  = await session_mgr.get_or_create(user_id, session_id)
        context     = session_mgr.get_context_string(session_id, last_n=3)
        full_query  = self._build_query(query, context)

        from google.genai.types import Content, Part
        message = Content(role="user", parts=[Part(text=full_query)])

        async for event in self._adk_runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            if event.content:
                for part in event.content.parts:
                    if part.text:
                        yield part.text

    # ── Internal ─────────────────────────────────────────────────────────────

    async def _run_adk(
        self,
        full_query: str,
        user_id: str,
        session_id: str,
        original_query: str,
    ) -> tuple[str, RouteDecision | None]:
        """
        Drive the ADK runner and collect the final response text.
        Returns (raw_text, route_decision).
        """
        from google.genai.types import Content, Part

        message  = Content(role="user", parts=[Part(text=full_query)])
        raw_text = ""
        route_decision: RouteDecision | None = None

        async for event in self._adk_runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=message,
        ):
            # Collect final response text
            if event.is_final_response() and event.content:
                for part in event.content.parts:
                    if part.text:
                        raw_text += part.text

            # Try to capture routing decision from intermediate events
            elif event.content and not event.is_final_response():
                for part in event.content.parts:
                    if part.text and route_decision is None:
                        route_decision = ResponseParser.parse_route_decision(
                            part.text, original_query
                        )

        return raw_text, route_decision

    @staticmethod
    def _build_query(query: str, context: str) -> str:
        """
        Prepend multi-turn context to the query if available.
        Keeps the injected context compact to minimise token usage.
        """
        if not context:
            return query
        return f"{context}\n\n[Current question]\n{query}"

    @staticmethod
    def _infer_mode(route_decision: RouteDecision | None) -> RetrievalMode:
        """Map the routed agent name to a RetrievalMode enum."""
        if route_decision is None:
            return RetrievalMode.UNKNOWN
        mapping = {
            "flat_rag_agent":  RetrievalMode.FLAT,
            "graph_rag_agent": RetrievalMode.GRAPH,
            "hybrid_agent":    RetrievalMode.HYBRID,
        }
        return mapping.get(route_decision.chosen_agent, RetrievalMode.UNKNOWN)

    # ── Sync wrapper (for CLI use) ───────────────────────────────────────────

    def run_sync(
        self,
        query: str,
        user_id: str = "default_user",
        session_id: str | None = None,
    ) -> AgentResponse:
        """
        Synchronous wrapper around run().
        Use in CLI scripts or pytest (with pytest-asyncio for full async tests).
        """
        return asyncio.run(self.run(query, user_id=user_id, session_id=session_id))

    # ── Health / metadata ────────────────────────────────────────────────────

    def store_health(self) -> dict:
        """Return store health without triggering full initialisation."""
        if self._store_manager is None:
            return {"status": "not_initialised"}
        return self._store_manager.health().to_dict() if hasattr(self._store_manager.health(), "to_dict") else {}
        return self._store_manager.health().to_dict()


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_runner: AgentRunner | None = None


def get_runner() -> AgentRunner:
    """Return the module-level AgentRunner singleton."""
    global _runner
    if _runner is None:
        _runner = AgentRunner()
    return _runner