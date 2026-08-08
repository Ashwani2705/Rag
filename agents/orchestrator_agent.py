"""
Orchestrator Agent — the entry point for all queries.

Routing logic:
  FLAT RAG    → factual / definitional / single-concept queries
  GRAPH RAG   → relational / multi-hop / dependency / "how X connects to Y"
  HYBRID      → broad / ambiguous / complex queries where both are needed

The orchestrator does NOT do retrieval itself. It classifies the query,
delegates to the right sub-agent, and returns the sub-agent's response.

Google ADK handles sub-agent delegation when sub-agents are listed in the
`sub_agents` parameter — the LLM selects which to call based on their
`description` fields and the routing instructions below.
"""

from __future__ import annotations

import os
from google.adk.agents import Agent
from google.adk.types import Content, Part

# from agents.flat_rag_agent  import flat_rag_agent
# from agents.graph_rag_agent import graph_rag_agent
# from agents.hybrid_agent    import hybrid_agent

# ---------------------------------------------------------------------------
# Routing system prompt
# ---------------------------------------------------------------------------

ORCHESTRATOR_SYSTEM_PROMPT = """
You are the Orchestrator Agent in a RAG + GraphRAG retrieval system.
Your ONLY job is to classify the incoming user query and delegate it to 
the correct sub-agent. You do not answer questions yourself.

=== SUB-AGENTS ===

1. flat_rag_agent
   USE FOR:
   - Factual lookups: "What is X?", "Define Y", "When did Z happen?"
   - Single-concept questions with no relational component
   - Questions where semantic similarity to document chunks is sufficient
   SIGNALS: "what", "define", "explain", "describe", "when", "where", "who"

2. graph_rag_agent
   USE FOR:
   - Multi-hop relational queries: "How does A connect to B?", "What does X depend on?"
   - Entity relationship questions: "What projects is Alice involved in?"
   - Dependency / impact analysis: "If X fails, what else is affected?"
   - Questions involving two or more named entities
   SIGNALS: "connect", "relate", "depend", "impact", "between", "through",
             "leads to", "caused by", multiple named entities in one query

3. hybrid_agent
   USE FOR:
   - Broad or ambiguous questions that may need both semantic and relational context
   - Questions where you cannot confidently assign to flat or graph
   - Comparative questions: "Compare X and Y in the context of Z"
   - "Tell me everything about X" style queries

=== ROUTING RULES ===

Step 1 — Count named entities in the query.
  - 0 or 1 entity + no relational language → flat_rag_agent
  - 2+ entities OR explicit relational language → graph_rag_agent
  - Ambiguous or broad → hybrid_agent

Step 2 — Check for relational signals (connect, relate, depend, impact, between,
  leads to, caused by, through, chain, path, affect, link).
  Any relational signal → graph_rag_agent (or hybrid if also broad).

Step 3 — Delegate. Do not answer the query yourself. Do not add commentary
  before or after the sub-agent's response. Return the sub-agent output as-is.

=== EXAMPLES ===

Query: "What is a transformer model?"
Route: flat_rag_agent  (single concept, definitional)

Query: "How does the authentication service connect to the user database?"
Route: graph_rag_agent  (two entities + "connect" signal)

Query: "What are Alice's responsibilities and how do her projects affect the Q3 roadmap?"
Route: hybrid_agent  (broad, multi-entity, both semantic + relational)

Query: "List all dependencies of the payment service"
Route: graph_rag_agent  ("dependencies" = relational, single named entity)

Query: "Summarise the architecture overview document"
Route: flat_rag_agent  (single document retrieval, no relational component)

=== FALLBACK ===

If the query is completely off-topic (not answerable from the document corpus),
respond with:
  "This query falls outside the indexed knowledge base. Please rephrase or 
   ask about topics covered in the ingested documents."
Do NOT delegate off-topic queries to sub-agents.
"""

# ---------------------------------------------------------------------------
# Orchestrator agent definition
# ---------------------------------------------------------------------------

orchestrator_agent = Agent(
    name="orchestrator_agent",
    model=os.environ.get("ORCHESTRATOR_MODEL", "gemini-1.5-pro"),   # Pro for routing accuracy
    description=(
        "Top-level routing agent. Classifies queries and delegates to "
        "flat_rag_agent, graph_rag_agent, or hybrid_agent."
    ),
    instruction=ORCHESTRATOR_SYSTEM_PROMPT,
    # sub_agents=[flat_rag_agent, graph_rag_agent, hybrid_agent],
)


# ---------------------------------------------------------------------------
# Convenience runner (for testing without FastAPI)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    """
    Quick smoke test:
        python -m agents.orchestrator_agent
    
    Requires env vars: GOOGLE_API_KEY (or VERTEX credentials), 
                       VECTOR_STORE_BACKEND, NEO4J_URI etc.
    """
    from google.adk.runners import Runner
    from google.adk.sessions import InMemorySessionService
    import asyncio

    SESSION_ID = "test-session-001"
    USER_ID    = "dev"

    session_service = InMemorySessionService()
    runner = Runner(
        agent=orchestrator_agent,
        app_name="rag-graphrag-demo",
        session_service=session_service,
    )

    test_queries = [
        "What is GraphRAG?",
        "How does the ingestion pipeline connect to the vector store?",
        "Tell me everything about the retrieval architecture and its trade-offs",
    ]

    async def run():
        await session_service.create_session(
            app_name="rag-graphrag-demo",
            user_id=USER_ID,
            session_id=SESSION_ID,
        )
        for query in test_queries:
            print(f"\n{'='*60}")
            print(f"Query : {query}")
            print(f"{'='*60}")
            
            message = Content(role="user", parts=[Part(text=query)])
            async for event in runner.run_async(
                user_id=USER_ID,
                session_id=SESSION_ID,
                new_message=message,
            ):
                if event.is_final_response() and event.content:
                    for part in event.content.parts:
                        if part.text:
                            print(f"Response: {part.text}")

    asyncio.run(run())