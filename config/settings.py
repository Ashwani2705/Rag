# config/settings.py
import os
from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    # API & Server
    api_host: str = Field("0.0.0.0", env="API_HOST")
    api_port: int = Field(8000, env="API_PORT")
    
    # Vector DB
    vector_db_type: str = Field("chromadb", env="VECTOR_DB_TYPE")
    chroma_persist_dir: str = Field("./chroma_db", env="CHROMA_PERSIST_DIR")
    
    # Graph DB
    neo4j_uri: str = Field("bolt://localhost:7687", env="NEO4J_URI")
    neo4j_user: str = Field("neo4j", env="NEO4J_USER")
    neo4j_password: str = Field("password", env="NEO4J_PASSWORD")
    
    # LLM & Embeddings
    embedding_model: str = Field("text-embedding-004", env="EMBEDDING_MODEL")
    llm_model: str = Field("gemini-1.5-flash", env="LLM_MODEL")
    vertexai_project_id: str = Field("", env="VERTEXAI_PROJECT_ID")
    vertexai_location: str = Field("us-central1", env="VERTEXAI_LOCATION")

    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()