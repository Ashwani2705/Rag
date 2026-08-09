"""
agents/cli.py

Phase 3 — Interactive CLI for testing the full agent pipeline.

Runs the orchestrator → sub-agent pipeline in a REPL loop.
Shows routing decision, retrieval mode, confidence, and sources
alongside the answer.

Usage:
    python -m agents.cli
    python -m agents.cli --user ashwani --session demo-01
    python -m agents.cli --query "What is GraphRAG?"   # single-shot
    python -m agents.cli --compare                      # run the golden test set
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys

from dotenv import load_dotenv
load_dotenv()


# ── Colour helpers (no deps) ─────────────────────────────────────────────────
RESET = "\033[0m"
BOLD  = "\033[1m"
BLUE  = "\033[34m"
GREEN = "\033[32m"
CYAN  = "\033[36m"
YELL  = "\033[33m"
RED   = "\033[31m"
GREY  = "\033[90m"

def _c(text: str, colour: str) -> str:
    return f"{colour}{text}{RESET}" if sys.stdout.isatty() else text


def _print_response(query: str, response) -> None:
    """Pretty-print an AgentResponse."""
    mode_colours = {
        "flat_rag": BLUE, "graph_rag": CYAN,
        "hybrid": GREEN, "unknown": GREY
    }
    mode      = response.retrieval_mode.value
    mode_col  = mode_colours.get(mode, GREY)
    conf_col  = GREEN if response.confidence.value == "high" else YELL if response.confidence.value == "medium" else RED

    print()
    print(_c("─" * 60, GREY))
    print(_c(f"  Mode       : {mode}", mode_col))
    print(_c(f"  Confidence : {response.confidence.value}", conf_col))
    if response.graph_path and response.graph_path.lower() != "none":
        print(_c(f"  Graph path : {response.graph_path}", CYAN))
    if response.high_conf_ids:
        print(_c(f"  High-conf  : {', '.join(response.high_conf_ids)}", GREEN))
    if response.sources:
        print(_c(f"  Sources    : {', '.join(response.sources)}", GREY))
    print(_c("─" * 60, GREY))
    print()
    print(_c(f"Q: {query}", BOLD))
    print()
    print(response.answer)
    print()


# ── Golden test queries ───────────────────────────────────────────────────────
GOLDEN_QUERIES = [
    # Group A — Flat RAG expected
    ("What is FAISS?",                          "flat_rag"),
    ("What is GraphRAG?",                       "flat_rag"),
    ("What embedding model does this project use?", "flat_rag"),
    # Group B — GraphRAG expected
    ("What does the Feature Store depend on?",  "graph_rag"),
    ("How does the API Gateway connect to the Data Pipeline?", "graph_rag"),
    ("Who leads the ML Engineering team and what do they own?", "graph_rag"),
    ("What projects is Priya Sharma involved in?", "graph_rag"),
    # Group C — Hybrid expected
    ("What happens if the API Gateway goes down?", "hybrid"),
    ("Compare flat RAG and GraphRAG retrieval strategies", "hybrid"),
    ("Tell me about the ingestion pipeline and its dependencies", "hybrid"),
]


async def run_repl(user_id: str, session_id: str | None) -> None:
    """Interactive REPL loop."""
    from agents.runner import get_runner

    runner = get_runner()
    print(_c("\n=== RAG + GraphRAG Demo — Interactive Mode ===", BOLD))
    print(_c("Commands: 'exit' to quit | 'history' to see turns | 'reset' to clear session\n", GREY))

    while True:
        try:
            query = input(_c("Query > ", BOLD)).strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if not query:
            continue
        if query.lower() in {"exit", "quit", "q"}:
            print("Bye.")
            break
        if query.lower() == "history":
            from agents.session_manager import get_session_manager
            sm = get_session_manager()
            history = sm.get_history(session_id or "")
            if not history:
                print(_c("  No history yet.", GREY))
            for t in history:
                print(_c(f"  [{t['mode']}] Q: {t['query'][:60]}  →  {t['answer'][:80]}...", GREY))
            print()
            continue
        if query.lower() == "reset":
            from agents.session_manager import get_session_manager
            sm = get_session_manager()
            if session_id:
                await sm.reset(session_id)
                print(_c("  Session cleared.", GREY))
            continue

        try:
            response = await runner.run(query, user_id=user_id, session_id=session_id)
            if session_id is None:
                session_id = response.metadata.get("session_id")
            _print_response(query, response)
        except Exception as e:
            print(_c(f"  ERROR: {e}", RED))
            import traceback; traceback.print_exc()


async def run_single(query: str, user_id: str) -> None:
    """Single-shot query, then exit."""
    from agents.runner import get_runner
    runner   = get_runner()
    response = await runner.run(query, user_id=user_id)
    _print_response(query, response)


async def run_compare(user_id: str) -> None:
    """
    Run all golden queries and print a side-by-side comparison table
    showing expected mode vs actual mode.
    """
    from agents.runner import get_runner
    runner = get_runner()

    print(_c("\n=== Golden Query Comparison ===\n", BOLD))
    print(f"{'Query':<52} {'Expected':<12} {'Actual':<12} {'Match'}")
    print("─" * 90)

    correct = 0
    for query, expected in GOLDEN_QUERIES:
        try:
            response = await runner.run(query, user_id=user_id)
            actual   = response.retrieval_mode.value
            match    = "✓" if actual == expected else "✗"
            if actual == expected:
                correct += 1
            col = GREEN if match == "✓" else RED
            print(_c(f"{query[:50]:<52} {expected:<12} {actual:<12} {match}", col))
        except Exception as e:
            print(_c(f"{query[:50]:<52} {expected:<12} ERROR        ✗  ({e})", RED))

    print("─" * 90)
    print(_c(f"\nRouting accuracy: {correct}/{len(GOLDEN_QUERIES)} ({100*correct//len(GOLDEN_QUERIES)}%)\n", BOLD))


# ── Entrypoint ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG + GraphRAG agent CLI")
    parser.add_argument("--user",    default="dev",  help="User ID for session scoping")
    parser.add_argument("--session", default=None,   help="Session ID (auto-generated if omitted)")
    parser.add_argument("--query",   default=None,   help="Single query mode — run once and exit")
    parser.add_argument("--compare", action="store_true", help="Run golden test set comparison")
    args = parser.parse_args()

    if args.compare:
        asyncio.run(run_compare(args.user))
    elif args.query:
        asyncio.run(run_single(args.query, args.user))
    else:
        asyncio.run(run_repl(args.user, args.session))


if __name__ == "__main__":
    main()