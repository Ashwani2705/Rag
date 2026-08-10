"""
stores/health_check.py

CLI tool to verify both stores are up and populated after ingestion.

Usage:
    python -m stores.health_check              # full health report
    python -m stores.health_check --stats      # include document counts
    python -m stores.health_check --smoke      # run a live search on both stores
"""

from __future__ import annotations

import argparse
import os
import sys


def run_health(show_stats: bool = False, smoke: bool = False) -> int:
    """
    Returns 0 on success, 1 on any failure.
    """
    from dotenv import load_dotenv
    load_dotenv()

    from stores.store_manager import get_store_manager

    print("\n=== RAG + GraphRAG — Store Health Check ===\n")
    sm = get_store_manager()
    sm.initialise()

    # ── Basic health ─────────────────────────────────────────────────────────
    report = sm.health()
    print(report.summary())

    if not report.all_ok:
        print("\n✗ One or more stores are unhealthy. Check logs above.")
        return 1

    # ── Stats ─────────────────────────────────────────────────────────────────
    if show_stats:
        print("\n--- Store Statistics ---")
        stats = sm.stats()
        print(f"Vector store  backend  : {stats['vector_store']['backend']}")
        print(f"Vector store  vectors  : {stats['vector_store']['vectors']}")
        print(f"Graph store   chunks   : {stats['graph_store']['chunks']}")
        print(f"Graph store   entities : {stats['graph_store']['entities']}")
        print(f"Graph store   relations: {stats['graph_store']['relations']}")
        print(f"Neo4j version          : {stats['graph_store']['neo4j']}")
        print(f"Active indexes         : {', '.join(stats['graph_store']['indexes'])}")

    # ── Smoke test ────────────────────────────────────────────────────────────
    if smoke:
        print("\n--- Smoke Test: Live Retrieval ---")

        # 1. Vector search
        print("\n[1] Vector search — query: 'What is GraphRAG?'")
        from ingestion.embedder import get_embedder
        embedder = get_embedder()
        query_emb = embedder.embed_query("What is GraphRAG?")
        results = sm.vector.similarity_search(query_emb, k=3)
        if results:
            for i, r in enumerate(results, 1):
                print(f"  Result {i} (score={r['score']:.3f}): {r['text'][:120]}...")
        else:
            print("  ⚠ No results. Have you run ingestion yet?")

        # 2. Graph search
        print("\n[2] Graph search — entity: 'API Gateway'")
        graph_results = sm.graph.entity_search(["API Gateway"], max_hops=2, top_k=3)
        if graph_results:
            for i, r in enumerate(graph_results, 1):
                print(f"  Result {i}: {r.start_entity} →{r.relations}→ {r.end_entity}")
                print(f"    chunk: {r.text[:120]}...")
        else:
            print("  ⚠ No results. Run ingestion with --mode llm or --mode spacy first.")

        # 3. Neighbour check
        print("\n[3] Entity neighbours — 'Priya Sharma'")
        neighbours = sm.graph.get_entity_neighbours("Priya Sharma")
        if neighbours:
            for n in neighbours[:5]:
                print(f"  {n['neighbour']} [{n['relation']}]")
        else:
            print("  ⚠ Entity not found in graph.")

    print("\n✓ Health check complete.\n")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG + GraphRAG store health check")
    parser.add_argument("--stats", action="store_true", help="Show document/entity counts")
    parser.add_argument("--smoke", action="store_true", help="Run live retrieval smoke test")
    args = parser.parse_args()

    exit_code = run_health(show_stats=args.stats, smoke=args.smoke)
    sys.exit(exit_code)