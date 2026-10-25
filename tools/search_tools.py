"""
ADK FunctionTool definitions.

Each function becomes a tool the orchestrator and sub-agents can invoke.
ADK auto-generates the tool schema from the function signature + docstring,
so keep types explicit and docstrings precise.

Tools:
  vector_search      — FAISS / Pinecone similarity search
  graph_search       — Neo4j Cypher entity + relation traversal
  hybrid_search      — vector + graph fusion with RRF re-ranking
  rerank_chunks      — cross-encoder re-ranking of candidate chunks
"""

from __future__ import annotations

import os
from typing import Any

from google.adk.tools import FunctionTool

# ---------------------------------------------------------------------------
# Shared helpers (lazy imports so tools don't break if a dep is missing)
# ---------------------------------------------------------------------------

# def _get_embedding(text: str) -> list[float]:
#     """Embed a query string using the configured model."""
#     backend = os.environ.get("EMBEDDING_BACKEND", "vertex").lower()
#     if backend == "vertex":
#         from vertexai.language_models import TextEmbeddingModel
#         model = TextEmbeddingModel.from_pretrained(
#             os.environ.get("VERTEX_EMBEDDING_MODEL", "textembedding-gecko@003")
#         )
#         return model.get_embeddings([text])[0].values
#     elif backend == "openai":
#         from openai import OpenAI
#         client = OpenAI()
#         resp = client.embeddings.create(
#             input=text,
#             model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
#         )
#         return resp.data[0].embedding
#     else:
#         raise ValueError(f"Unknown EMBEDDING_BACKEND='{backend}'")
def _get_embedding(text: str) -> list[float]:
    """Embed a query string using the configured model."""
    backend = os.environ.get("EMBEDDING_BACKEND", "google").lower()

    if backend in ["google", "gemini", "vertex"]:
        import google.generativeai as genai

        # Configures API key from environment variable
        api_key = os.environ.get("gemini_api_key") 
        if api_key:
            genai.configure(api_key=api_key)

        model_name = os.environ.get("GOOGLE_EMBEDDING_MODEL", "gemini-embedding-001")
        result = genai.embed_content(
            model=model_name,
            content=text,
            task_type="retrieval_query"
        )
        return result["embedding"]

    elif backend == "sentence-transformers":
        from sentence_transformers import SentenceTransformer
        model_name = os.environ.get("ST_EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        model = SentenceTransformer(model_name)
        return model.encode(text).tolist()

    elif backend == "openai":
        from openai import OpenAI
        client = OpenAI()
        resp = client.embeddings.create(
            input=text,
            model=os.environ.get("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"),
        )
        return resp.data[0].embedding

    else:
        raise ValueError(
            f"Unknown EMBEDDING_BACKEND='{backend}'. "
            f"Supported backends are: 'google', 'vertex', 'sentence-transformers', 'openai'."
        )


def _get_graph_store():
    from stores.graph_store import get_graph_store
    return get_graph_store()


# ---------------------------------------------------------------------------
# 1. Vector search tool
# ---------------------------------------------------------------------------

def vector_search(query: str, top_k: int = 5) -> dict[str, Any]:
    """
    Search the vector store for chunks most semantically similar to the query.

    Args:
        query:  The user question or search string.
        top_k:  Number of top results to return (default 5, max 20).

    Returns:
        A dict with keys:
          - results: list of {text, metadata, score}
          - backend: which vector store was used (faiss | pinecone)
          - query:   the original query string
    """
    from stores.vector_store import get_vector_store

    top_k = min(top_k, 20)
    store = get_vector_store()
    store.load()

    query_emb = _get_embedding(query)
    results = store.similarity_search(query_emb, k=top_k)

    return {
        "results": results,
        "backend": os.environ.get("VECTOR_STORE_BACKEND", "faiss"),
        "query": query,
    }


# ---------------------------------------------------------------------------
# 2. Graph search tool
# ---------------------------------------------------------------------------

def graph_search(query: str, max_hops: int = 2, top_k: int = 5) -> dict[str, Any]:
    """
    Extract named entities from the query, then traverse the Neo4j knowledge
    graph up to max_hops away to find related entities and their source chunks.

    Args:
        query:    The user question or search string.
        max_hops: How many relationship hops to traverse (default 2, max 3).
        top_k:    Maximum number of paths / chunk references to return.

    Returns:
        A dict with keys:
          - paths:   list of {start_entity, relation, end_entity, chunk_id, text}
          - entities_found: list of entity names matched in the graph
          - query:   the original query string
    """
    max_hops = min(max_hops, 3)
    top_k = min(top_k, 20)

    # Step 1 — extract entities from the query
    entities = _extract_entities(query)
    if not entities:
        # Fallback: full-text search on chunk text
        gs = _get_graph_store()
        ft_results = gs.fulltext_entity_search(query, top_k=top_k)
        return {"paths": ft_results, "entities_found": [], "query": query, "fallback": "fulltext"}

    # Step 2 — graph traversal via graph store
    gs = _get_graph_store()
    raw = gs.entity_search(entities, max_hops=max_hops, top_k=top_k)

    paths = [
        {
            "start_entity": r.start_entity,
            "relations":    r.relations,
            "end_entity":   r.end_entity,
            "chunk_id":     r.chunk_id,
            "text":         r.text,
            "hop_count":    r.hop_count,
        }
        for r in raw
    ]
    if not paths:
        ft_results = gs.fulltext_entity_search(query, top_k=top_k)
        return {"paths": ft_results, "entities_found": entities, "query": query, "fallback": "fulltext"}

    return {"paths": paths, "entities_found": entities, "query": query}


def _extract_entities(text: str) -> list[str]:
    """
    Lightweight entity extraction using spaCy (no LLM call, fast).
    Falls back to a Gemini prompt if spaCy is not installed.
    """
    try:
        import spacy
        nlp = spacy.load("en_core_web_sm")
        doc = nlp(text)
        return list({ent.text for ent in doc.ents if ent.label_ in {"PERSON", "ORG", "PRODUCT", "GPE", "EVENT", "WORK_OF_ART"}})
    except Exception:
        # Gemini fallback
        import google.generativeai as genai
        genai.configure(api_key=os.environ["gemini_api_key"])
        model = genai.GenerativeModel("gemini-2.5-flash")
        prompt = (
            f"Extract all named entities (people, organisations, products, places) "
            f"from the following text. Return only a comma-separated list, nothing else.\n\nText: {text}"
        )
        resp = model.generate_content(prompt)
        raw = resp.text.strip()
        return [e.strip() for e in raw.split(",") if e.strip()]


# ---------------------------------------------------------------------------
# 3. Hybrid search tool (vector + graph, RRF re-ranking)
# ---------------------------------------------------------------------------

def hybrid_search(query: str, top_k: int = 5, vector_weight: float = 0.5) -> dict[str, Any]:
    """
    Combine vector similarity search and graph traversal, then fuse results
    using Reciprocal Rank Fusion (RRF).

    Args:
        query:         The user question or search string.
        top_k:         Final number of results after fusion (default 5).
        vector_weight: Weight for vector results in RRF (0.0–1.0). Graph gets
                       (1 - vector_weight). Default 0.5 = equal weight.

    Returns:
        A dict with keys:
          - results: fused list of {text, metadata, score, source}
            where source is 'vector', 'graph', or 'both'
          - query:   the original query string
    """
    vector_weight = max(0.0, min(1.0, vector_weight))
    graph_weight  = 1.0 - vector_weight

    # Run both retrievers
    vec_out   = vector_search(query, top_k=top_k * 2)
    graph_out = graph_search(query,  max_hops=2, top_k=top_k * 2)

    # Build a unified candidate pool keyed by chunk_id
    candidates: dict[str, dict] = {}

    for rank, item in enumerate(vec_out["results"]):
        cid = item["metadata"].get("chunk_id", f"vec_{rank}")
        candidates.setdefault(cid, {"text": item["text"], "metadata": item["metadata"], "sources": set(), "rrf_score": 0.0})
        candidates[cid]["rrf_score"] += vector_weight * _rrf(rank)
        candidates[cid]["sources"].add("vector")

    for rank, path in enumerate(graph_out["paths"]):
        cid = path.get("chunk_id", f"graph_{rank}")
        meta = {"chunk_id": cid, "start_entity": path["start_entity"], "end_entity": path["end_entity"]}
        candidates.setdefault(cid, {"text": path["text"], "metadata": meta, "sources": set(), "rrf_score": 0.0})
        candidates[cid]["rrf_score"] += graph_weight * _rrf(rank)
        candidates[cid]["sources"].add("graph")

    # Sort by fused RRF score
    ranked = sorted(candidates.values(), key=lambda x: x["rrf_score"], reverse=True)[:top_k]

    results = []
    for item in ranked:
        results.append({
            "text":     item["text"],
            "metadata": item["metadata"],
            "score":    round(item["rrf_score"], 4),
            "source":   "both" if len(item["sources"]) > 1 else next(iter(item["sources"])),
        })

    return {"results": results, "query": query}


def _rrf(rank: int, k: int = 60) -> float:
    """Reciprocal Rank Fusion score. k=60 is the standard constant."""
    return 1.0 / (k + rank + 1)


# ---------------------------------------------------------------------------
# 4. Rerank tool (cross-encoder — optional, improves precision)
# ---------------------------------------------------------------------------

def rerank_chunks(query: str, chunks: list[str], top_k: int = 3) -> dict[str, Any]:
    """
    Re-rank a list of text chunks against the query using a cross-encoder model.
    Use after vector_search or hybrid_search to boost precision.

    Args:
        query:  The original user question.
        chunks: List of candidate chunk texts to re-rank.
        top_k:  Number of top chunks to return after re-ranking.

    Returns:
        A dict with keys:
          - ranked_chunks: list of {text, score} sorted by relevance descending
          - query: the original query string
    """
    try:
        from sentence_transformers import CrossEncoder
        model = CrossEncoder(os.environ.get("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2"))
        pairs = [(query, chunk) for chunk in chunks]
        scores = model.predict(pairs)
        ranked = sorted(zip(chunks, scores), key=lambda x: x[1], reverse=True)
        return {
            "ranked_chunks": [{"text": t, "score": float(s)} for t, s in ranked[:top_k]],
            "query": query,
        }
    except ImportError:
        # Fallback — return as-is if cross-encoder not installed
        return {
            "ranked_chunks": [{"text": c, "score": 1.0} for c in chunks[:top_k]],
            "query": query,
        }


# ---------------------------------------------------------------------------
# Export as ADK FunctionTool instances
# ---------------------------------------------------------------------------

VECTOR_SEARCH_TOOL = FunctionTool(func=vector_search)
GRAPH_SEARCH_TOOL  = FunctionTool(func=graph_search)
HYBRID_SEARCH_TOOL = FunctionTool(func=hybrid_search)
RERANK_TOOL        = FunctionTool(func=rerank_chunks)