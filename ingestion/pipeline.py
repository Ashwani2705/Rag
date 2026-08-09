"""
ingestion/pipeline.py

Ingestion entrypoint — runs all 4 Phase 1 steps in sequence:

  1. loader.py       → RawDocument list
  2. chunker.py      → Chunk list
  3. embedder.py     → embeddings → FAISS / Pinecone
  4. graph_builder.py → entities + relations → Neo4j

Usage:
  # From code:
  from ingestion.pipeline import run_ingestion
  run_ingestion(source="data/", extraction_mode="spacy")

  # From CLI:
  python -m ingestion.pipeline --source data/ --mode llm
  python -m ingestion.pipeline --source https://example.com/doc
  python -m ingestion.pipeline --source docs/report.pdf --mode spacy
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from ingestion.loader       import DocumentLoader
from ingestion.chunker      import DocumentChunker
from ingestion.embedder     import embed_and_store_chunks
from ingestion.graph_builder import GraphBuilder


def run_ingestion(
    source: str,
    extraction_mode: str = "spacy",
    chunk_size: int = 2048,
    chunk_overlap: int = 307,
    skip_graph: bool = False,
) -> dict:
    """
    Full Phase 1 pipeline.

    Args:
        source          : file path, directory path, or URL
        extraction_mode : "spacy" (fast) or "llm" (quality) for graph extraction
        chunk_size      : target chars per chunk (default 2048 ≈ 512 tokens)
        chunk_overlap   : overlap between chunks in chars (default 307 ≈ 15%)
        skip_graph      : set True to skip Neo4j step (vector-only ingestion)

    Returns:
        Summary dict with doc_count, chunk_count, entity_count, elapsed_seconds
    """
    t0 = time.time()
    print(f"\n{'='*55}")
    print(f"  RAG + GraphRAG — Ingestion Pipeline")
    print(f"  source          : {source}")
    print(f"  extraction_mode : {extraction_mode}")
    print(f"  skip_graph      : {skip_graph}")
    print(f"{'='*55}\n")

    # ── Step 1: Load ────────────────────────────────────────────────────────
    print("[1/4] Loading documents...")
    loader = DocumentLoader()
    if Path(source).is_dir():
        docs = loader.load_directory(source)
    else:
        docs = loader.load(source)
    print(f"  → {len(docs)} document(s) loaded\n")

    if not docs:
        print("  ⚠ No documents loaded. Check source path and file types.")
        return {"doc_count": 0, "chunk_count": 0, "elapsed_seconds": 0}

    # ── Step 2: Chunk ───────────────────────────────────────────────────────
    print("[2/4] Chunking documents...")
    chunker = DocumentChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = chunker.chunk_documents(docs)
    print(f"  → {len(chunks)} chunk(s) produced\n")

    if not chunks:
        print("  ⚠ No chunks produced. Documents may be empty or below min size.")
        return {"doc_count": len(docs), "chunk_count": 0, "elapsed_seconds": 0}

    # ── Step 3: Embed + store to vector store ───────────────────────────────
    print("[3/4] Embedding and storing in vector store...")
    chunks, embeddings = embed_and_store_chunks(chunks)
    print(f"  → {len(embeddings)} embeddings stored\n")

    # ── Step 4: Build knowledge graph ───────────────────────────────────────
    if skip_graph:
        print("[4/4] Skipping graph build (skip_graph=True)\n")
        entity_count = 0
        entity_count = 0 # Changed from "see Neo4j" to 0 for consistency
    else:
        print(f"[4/4] Building knowledge graph (mode={extraction_mode})...")
        builder = GraphBuilder(extraction_mode=extraction_mode)
        builder.build(chunks)
        entity_count = "see Neo4j"   # we don't count here to avoid extra query

    elapsed = round(time.time() - t0, 2)
    summary = {
        "doc_count":       len(docs),
        "chunk_count":     len(chunks),
        "entity_count":    entity_count,
        "elapsed_seconds": elapsed,
    }

    print(f"\n{'='*55}")
    print(f"  ✓ Ingestion complete in {elapsed}s")
    print(f"  docs    : {summary['doc_count']}")
    print(f"  chunks  : {summary['chunk_count']}")
    print(f"  entities: {summary['entity_count']}")
    print(f"{'='*55}\n")

    return summary


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG + GraphRAG ingestion pipeline")
    parser.add_argument("--source",     required=True, help="File path, directory, or URL to ingest")
    parser.add_argument("--mode",       default="spacy", choices=["spacy", "llm"], help="Entity extraction mode")
    parser.add_argument("--chunk-size", type=int, default=2048, help="Target chunk size in chars")
    parser.add_argument("--overlap",    type=int, default=307,  help="Chunk overlap in chars")
    parser.add_argument("--skip-graph", action="store_true",    help="Skip Neo4j graph build")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    run_ingestion(
        source=args.source,
        extraction_mode=args.mode,
        chunk_size=args.chunk_size,
        chunk_overlap=args.overlap,
        skip_graph=args.skip_graph,
    )