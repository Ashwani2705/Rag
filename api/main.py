# api/main.py
import uuid
import os
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator_agent import orchestrator_agent
from agents.models import ResponseParser, RetrievalMode
from ingestion.pipeline import run_ingestion
from eval.live_evaluator import get_live_evaluator
from stores.store_manager import get_store_manager
from stores.semantic_cache import get_semantic_cache, get_current_corpus_version


app = FastAPI(title="RAG-GraphRAG Demo API")


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --------------------------------------------------
# ADK CONFIGURATION
# --------------------------------------------------

APP_NAME = "rag_graphrag_api"

session_service = InMemorySessionService()

runner = Runner(
    agent=orchestrator_agent,
    app_name=APP_NAME,
    session_service=session_service,
)


# --------------------------------------------------
# REQUEST / RESPONSE MODELS
# --------------------------------------------------

class QueryRequest(BaseModel):
    query: str
    mode: str = "hybrid"
    evaluate: bool = True         # Compute RAGAS metrics on the response
    bypass_cache: bool = False    # Force fresh generation bypassing semantic cache


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    mode_used: str
    sources: list[str]
    chunks: list[dict] = []       # Exact chunks: [{"chunk_id": str, "text": str, "source": str}]
    confidence: str | None = None
    graph_path: str | None = None
    latency_ms: float
    metrics: dict | None = None   # { faithfulness, answer_relevancy, context_precision, overall_score }
    is_cached: bool = False
    cache_match_type: str | None = None
    cache_similarity: float | None = None
    matched_query: str | None = None


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "RAG-GraphRAG Demo API is ready with Semantic Cache."
    }


# --------------------------------------------------
# UPLOAD (With Cache Invalidation)
# --------------------------------------------------

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):

    file_location = None

    try:
        os.makedirs("temp_uploads", exist_ok=True)

        file_location = os.path.join(
            "temp_uploads",
            file.filename
        )

        with open(file_location, "wb") as file_object:
            content = await file.read()
            file_object.write(content)

        result = run_ingestion(
            source=file_location,
            extraction_mode="spacy",
            chunk_size=2048,
            chunk_overlap=307,
            skip_graph=False
        )

        # Invalidate semantic cache so stale answers are never served after new documents
        cache = get_semantic_cache()
        cache.invalidate()

        return {
            "message": "File ingested successfully! Semantic cache refreshed.",
            "filename": file.filename,
            "chunks_processed": result["chunk_count"],
            "entities_processed": result.get(
                "entity_count",
                "See Neo4j"
            )
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )

    finally:
        if file_location and os.path.exists(file_location):
            os.remove(file_location)


# --------------------------------------------------
# QUERY (Semantic Cache Lookup -> Fallback to Agents)
# --------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):

    start_time = time.time()
    cache = get_semantic_cache()
    current_version = get_current_corpus_version()

    # --------------------------------------------------
    # Step 1: Check Semantic Cache
    # --------------------------------------------------
    if not request.bypass_cache:
        cached = cache.lookup(request.query, corpus_version=current_version)
        if cached:
            return QueryResponse(
                query_id=str(uuid.uuid4()),
                answer=cached.answer,
                mode_used=cached.mode_used,
                sources=cached.sources,
                chunks=cached.chunks,
                confidence=cached.confidence,
                graph_path=cached.graph_path,
                latency_ms=round(cached.lookup_latency_ms, 2),
                metrics=cached.metrics,
                is_cached=True,
                cache_match_type=cached.match_type,
                cache_similarity=round(cached.similarity, 4),
                matched_query=cached.matched_query,
            )

    # --------------------------------------------------
    # Step 2: Cache Miss -> Run Main Agent Pipeline
    # --------------------------------------------------
    try:
        user_id = "api_user"
        session_id = str(uuid.uuid4())

        # Create ADK session
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )

        from google.genai import types

        content = types.Content(
            role="user",
            parts=[
                types.Part(
                    text=request.query
                )
            ]
        )

        events = runner.run_async(
            user_id=user_id,
            session_id=session_id,
            new_message=content,
        )

        raw_answer = ""
        async for event in events:
            if event.is_final_response():
                if event.content and event.content.parts:
                    raw_answer = "".join(
                        part.text
                        for part in event.content.parts
                        if part.text
                    )

        # Map requested mode to enum
        mode_enum = RetrievalMode.HYBRID
        if request.mode == "flat":
            mode_enum = RetrievalMode.FLAT
        elif request.mode == "graph":
            mode_enum = RetrievalMode.GRAPH

        parsed = ResponseParser.parse(raw_answer, mode_enum)
        final_answer = parsed.answer if parsed.has_answer else raw_answer

        # Extract and validate 16-character hexadecimal chunk IDs
        import re
        raw_sources = parsed.sources
        valid_chunk_ids = []
        for s in raw_sources:
            valid_chunk_ids.extend(re.findall(r"\b[0-9a-fA-F]{16}\b", s))
        if not valid_chunk_ids:
            valid_chunk_ids = re.findall(r"\b[0-9a-fA-F]{16}\b", raw_answer)

        sources = list(dict.fromkeys(valid_chunk_ids))
        confidence = parsed.confidence.value
        graph_path = parsed.graph_path

        # Retrieve exact chunks from database
        sm = get_store_manager()
        sm.initialise()
        retrieved_chunks = sm.get_chunks_by_ids(sources) if sources else []

        # Live RAGAS Evaluation
        metrics = None
        if request.evaluate and final_answer:
            try:
                evaluator = get_live_evaluator()
                context_texts = [
                    f"[Chunk ID: {c['chunk_id']}]\n{c['text']}"
                    for c in retrieved_chunks
                ] if retrieved_chunks else None

                metrics = evaluator.evaluate(
                    query=request.query,
                    answer=final_answer,
                    contexts=context_texts,
                )
            except Exception as eval_err:
                print(f"[api] Live evaluation error: {eval_err}")

        # --------------------------------------------------
        # Step 3: Persist Fresh Result into Semantic Cache
        # --------------------------------------------------
        if final_answer:
            try:
                cache.store(
                    query=request.query,
                    query_embedding=None,
                    answer=final_answer,
                    sources=sources,
                    chunks=retrieved_chunks,
                    confidence=confidence or "unknown",
                    graph_path=graph_path,
                    metrics=metrics,
                    corpus_version=current_version,
                    mode_used=request.mode,
                )
            except Exception as cache_err:
                print(f"[api] Cache store error: {cache_err}")

        elapsed = (time.time() - start_time) * 1000

        return QueryResponse(
            query_id=str(uuid.uuid4()),
            answer=final_answer or "No response generated.",
            mode_used=request.mode,
            sources=sources,
            chunks=retrieved_chunks,
            confidence=confidence,
            graph_path=graph_path,
            latency_ms=round(elapsed, 2),
            metrics=metrics,
            is_cached=False,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# HEALTH & CACHE STATS
# --------------------------------------------------

@app.get("/health")
async def health_check():
    cache = get_semantic_cache()
    return {
        "status": "healthy",
        "cache": cache.stats(),
    }
