# # api/main.py
# import uuid
# import os
# import time
# from fastapi import FastAPI, HTTPException, UploadFile, File
# from fastapi.middleware.cors import CORSMiddleware
# from pydantic import BaseModel

# # Import your existing Agents and Pipeline
# from agents.orchestrator_agent import orchestrator_agent

# # orchestrator_agent is already an LlmAgent instance
# orchestrator = orchestrator_agent
# from ingestion.pipeline import run_ingestion  # <--- Importing your existing pipeline
# from config.settings import settings

# app = FastAPI(title="RAG-GraphRAG Demo API")

# # CORS for Streamlit
# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # --- Request Models ---
# class QueryRequest(BaseModel):
#     query: str
#     mode: str = "hybrid"

# class QueryResponse(BaseModel):
#     query_id: str
#     answer: str
#     mode_used: str
#     sources: list[str]
#     latency_ms: float

# # --- Endpoints ---

# @app.get("/")
# async def root():
#     return {"message": "RAG-GraphRAG Demo API is ready."}

# @app.post("/upload")
# async def upload_document(file: UploadFile = File(...)):
#     """
#     Accepts a file, saves it locally, and triggers your existing ingestion.pipeline.
#     """
#     try:
#         # 1. Save the uploaded file temporarily (required for your CLI-based pipeline)
#         file_location = f"temp_uploads/{file.filename}"
#         os.makedirs("temp_uploads", exist_ok=True)
        
#         with open(file_location, "wb+") as file_object:
#             file_object.write(file.file.read())
        
#         # 2. Trigger your existing pipeline programmatically
#         # NOTE: We are passing the 'source' as the local file path, 
#         # and using 'skip_graph=False' so it builds the Neo4j graph.
#         result = run_ingestion(
#             source=file_location,
#             extraction_mode="spacy",   # or "llm" if you prefer
#             chunk_size=2048,
#             chunk_overlap=307,
#             skip_graph=False
#         )
        
#         # 3. Clean up the temp file after successful ingestion
#         os.remove(file_location)
        
#         return {
#             "message": "File ingested successfully! You can now chat.",
#             "filename": file.filename,
#             "chunks_processed": result["chunk_count"],
#             "entities_processed": result.get("entity_count", "See Neo4j")
#         }
    
#     except Exception as e:
#         # Clean up temp file even if it fails
#         if os.path.exists(file_location):
#             os.remove(file_location)
#         raise HTTPException(status_code=500, detail=str(e))

# @app.post("/query", response_model=QueryResponse)
# async def process_query(request: QueryRequest):
#     """
#     Routes query to the ADK Orchestrator.
#     """
#     start_time = time.time()
#     try:
#         response = orchestrator.run(
#             user_query=request.query, 
#             mode=request.mode
#         )
#         elapsed = (time.time() - start_time) * 1000
        
#         return QueryResponse(
#             query_id=str(uuid.uuid4()),
#             answer=response.get("answer", "No response generated."),
#             mode_used=request.mode,
#             sources=response.get("sources", []),
#             latency_ms=round(elapsed, 2)
#         )
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))

# @app.get("/health")
# async def health_check():
#     return {"status": "healthy"}


import uuid
import os
import time

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from google.adk.runners import Runner
from google.adk.sessions import InMemorySessionService
# --- ADD THESE 2 LINES AT THE VERY TOP ---
from dotenv import load_dotenv
load_dotenv()

from agents.orchestrator_agent import orchestrator_agent
from ingestion.pipeline import run_ingestion


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


class QueryResponse(BaseModel):
    query_id: str
    answer: str
    mode_used: str
    sources: list[str]
    latency_ms: float


# --------------------------------------------------
# ROOT
# --------------------------------------------------

@app.get("/")
async def root():
    return {
        "message": "RAG-GraphRAG Demo API is ready."
    }


# --------------------------------------------------
# UPLOAD
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


        return {
            "message": "File ingested successfully! You can now chat.",
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
# QUERY
# --------------------------------------------------

@app.post("/query", response_model=QueryResponse)
async def process_query(request: QueryRequest):

    start_time = time.time()

    try:

        user_id = "api_user"

        session_id = str(uuid.uuid4())

        # Create ADK session
        await session_service.create_session(
            app_name=APP_NAME,
            user_id=user_id,
            session_id=session_id,
        )


        # --------------------------------------------------
        # Run ADK Agent
        # --------------------------------------------------

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


        answer = ""

        async for event in events:

            if event.is_final_response():

                if event.content and event.content.parts:

                    answer = "".join(
                        part.text
                        for part in event.content.parts
                        if part.text
                    )


        elapsed = (
            time.time() - start_time
        ) * 1000


        return QueryResponse(
            query_id=str(uuid.uuid4()),
            answer=answer or "No response generated.",
            mode_used=request.mode,
            sources=[],
            latency_ms=round(elapsed, 2)
        )


    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )


# --------------------------------------------------
# HEALTH
# --------------------------------------------------

@app.get("/health")
async def health_check():

    return {
        "status": "healthy"
    }