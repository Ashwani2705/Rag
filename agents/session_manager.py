"""
agents/session_manager.py

Phase 3 — Session state management.

Wraps ADK's InMemorySessionService to provide:
  - Typed session state (query history, route decisions, response cache)
  - Multi-turn context: agents can see the last N exchanges
  - Session creation / retrieval / reset
  - Conversation history export (for eval and logging)

Why this matters:
  ADK agents are stateless between turns unless you explicitly write
  to session state. Without this, the orchestrator cannot use prior
  routing decisions to improve accuracy, and sub-agents cannot
  reference what was said earlier in the conversation.

Usage:
    from agents.session_manager import get_session_manager
    sm = get_session_manager()
    session = await sm.get_or_create("user-123", "session-abc")
    await sm.record_turn(session_id, query, response, route_decision)
    history = await sm.get_history(session_id, last_n=5)
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from typing import Any

from agents.models import AgentResponse, RouteDecision, RetrievalMode


# ---------------------------------------------------------------------------
# Typed session state models
# ---------------------------------------------------------------------------

@dataclass
class Turn:
    """A single question-answer exchange stored in session history."""
    turn_id:        str
    query:          str
    response:       AgentResponse
    route_decision: RouteDecision | None
    timestamp:      str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict:
        return {
            "turn_id":   self.turn_id,
            "query":     self.query,
            "answer":    self.response.answer,
            "mode":      self.response.retrieval_mode.value,
            "confidence": self.response.confidence.value,
            "sources":   self.response.sources,
            "agent":     self.route_decision.chosen_agent if self.route_decision else "unknown",
            "timestamp": self.timestamp,
        }


@dataclass
class SessionState:
    """
    Full state for one user session.
    Stored in ADK's session service under the key 'rag_state'.
    """
    session_id:  str
    user_id:     str
    created_at:  str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    turns:       list[Turn] = field(default_factory=list)
    metadata:    dict[str, Any] = field(default_factory=dict)

    @property
    def turn_count(self) -> int:
        return len(self.turns)

    def last_n_turns(self, n: int) -> list[Turn]:
        return self.turns[-n:] if self.turns else []

    def to_context_string(self, last_n: int = 3) -> str:
        """
        Format the last N turns as a compact context string for injection
        into the orchestrator's prompt. Keeps context window usage minimal.
        """
        turns = self.last_n_turns(last_n)
        if not turns:
            return ""
        lines = ["[Prior conversation context]"]
        for t in turns:
            lines.append(f"Q: {t.query}")
            lines.append(f"A ({t.response.retrieval_mode.value}): {t.response.answer[:300]}...")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Session Manager
# ---------------------------------------------------------------------------

class SessionManager:
    """
    Manages ADK sessions and typed session state.

    Uses ADK InMemorySessionService for session lifecycle and stores
    typed SessionState alongside it in a local dict (since ADK's session
    state is a flat dict, not a typed model).

    For production: swap InMemorySessionService with a persistent
    backend (e.g. Redis, Firestore via ADK's planned session backends).
    """

    APP_NAME = "rag-graphrag-demo"

    def __init__(self):
        from google.adk.sessions import InMemorySessionService
        self._adk_sessions  = InMemorySessionService()
        self._state_store:  dict[str, SessionState] = {}   # session_id → SessionState

    # ── Session lifecycle ────────────────────────────────────────────────────

    async def get_or_create(
        self,
        user_id: str,
        session_id: str | None = None,
    ) -> str:
        """
        Return an existing session_id or create a new one.
        If session_id is None, a UUID is generated.
        """
        if session_id is None:
            session_id = str(uuid.uuid4())

        if session_id not in self._state_store:
            # Create in ADK
            await self._adk_sessions.create_session(
                app_name=self.APP_NAME,
                user_id=user_id,
                session_id=session_id,
            )
            # Create typed state
            self._state_store[session_id] = SessionState(
                session_id=session_id,
                user_id=user_id,
            )

        return session_id

    async def reset(self, session_id: str) -> None:
        """Clear all turns in a session (keep session alive)."""
        if session_id in self._state_store:
            self._state_store[session_id].turns.clear()

    def get_state(self, session_id: str) -> SessionState | None:
        return self._state_store.get(session_id)

    def list_sessions(self, user_id: str) -> list[str]:
        return [
            sid for sid, state in self._state_store.items()
            if state.user_id == user_id
        ]

    # ── Turn recording ───────────────────────────────────────────────────────

    async def record_turn(
        self,
        session_id: str,
        query: str,
        response: AgentResponse,
        route_decision: RouteDecision | None = None,
    ) -> Turn:
        """
        Append a completed question-answer turn to session history.
        Returns the Turn object for immediate use (e.g. logging).
        """
        state = self._state_store.get(session_id)
        if state is None:
            raise ValueError(f"Session not found: {session_id}. Call get_or_create first.")

        turn = Turn(
            turn_id=str(uuid.uuid4()),
            query=query,
            response=response,
            route_decision=route_decision,
        )
        state.turns.append(turn)

        # Also write a compact summary to ADK session state
        # so agents can access it via tool_context if needed
        adk_state_patch = {
            "last_query":  query,
            "last_mode":   response.retrieval_mode.value,
            "last_answer": response.answer[:500],
            "turn_count":  state.turn_count,
        }
        adk_session = await self._adk_sessions.get_session(
            app_name=self.APP_NAME,
            user_id=state.user_id,
            session_id=session_id,
        )
        if adk_session:
            adk_session.state.update(adk_state_patch)

        return turn

    # ── Context helpers ──────────────────────────────────────────────────────

    def get_context_string(self, session_id: str, last_n: int = 3) -> str:
        """
        Return a compact multi-turn context string for prompt injection.
        Returns "" if session not found or no prior turns.
        """
        state = self._state_store.get(session_id)
        if not state:
            return ""
        return state.to_context_string(last_n=last_n)

    def get_history(self, session_id: str, last_n: int | None = None) -> list[dict]:
        """Return turn history as a list of dicts (for API and eval)."""
        state = self._state_store.get(session_id)
        if not state:
            return []
        turns = state.last_n_turns(last_n) if last_n else state.turns
        return [t.to_dict() for t in turns]

    # ── ADK session service accessor (used by Runner) ────────────────────────

    @property
    def adk_service(self):
        """Expose the raw ADK session service for the Runner."""
        return self._adk_sessions


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

_session_manager: SessionManager | None = None


def get_session_manager() -> SessionManager:
    global _session_manager
    if _session_manager is None:
        _session_manager = SessionManager()
    return _session_manager