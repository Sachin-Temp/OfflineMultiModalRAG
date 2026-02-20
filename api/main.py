"""
api/main.py

Main entry point for the FastAPI application.
Initializes the app, middleware (CORS), and routers.
Also handles startup/shutdown events for global resources.
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import sys
from pathlib import Path
# Add project root to sys.path
sys.path.append(str(Path(__file__).parent.parent))

from loguru import logger

from config.settings import (
    API_HOST, API_PORT, 
    UPLOAD_DIR, OUTPUT_DIR
)
# Ensure API_TITLE/VERSION are imported. 
# If not in settings, fallback.
try:
    from config.settings import PROJECT_NAME as API_TITLE
    from config.settings import VERSION as API_VERSION
except ImportError:
    API_TITLE = "Multimodal RAG"
    API_VERSION = "1.0.0"


from api.routers import (
    health, ingest, query, export, session
)
from modules.indexing.milvus_store import MilvusStore
from modules.indexing.tantivy_index import TantivyIndex
from modules.indexing.sqlite_store import SQLiteStore
from modules.retrieval.retrieval_engine import RetrievalEngine
from modules.llm.llm_engine import LLMEngine
from modules.citation.citation_engine import CitationEngine
from modules.export.export_engine import ExportEngine


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle manager:
    1. Initialize global resources (DB connections, Models) on startup.
    2. Clean up on shutdown.
    """
    logger.info("Starting up Multimodal RAG API...")
    
    # Ensure directories exist
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Initialize Stores
    # Note: initialization might require async or sync calls.
    # We store them in app.state for access via dependencies.
    
    try:
        app.state.milvus = MilvusStore()  # Connects to Milvus
        app.state.tantivy = TantivyIndex()
        app.state.sqlite = SQLiteStore()
        
        # Initialize Engines
        # Retrieval depends on stores
        app.state.retrieval_engine = RetrievalEngine(
            milvus_store=app.state.milvus,
            tantivy_index=app.state.tantivy,
            sqlite_store=app.state.sqlite
        )
        
        # LLM Engine depends on retrieval (and maybe VRAM manager internally)
        app.state.llm_engine = LLMEngine()
        
        # Citation & Export are lightweight or depend on others?
        # CitationEngine is mostly stateless logic
        app.state.citation_engine = CitationEngine()
        
        # ExportEngine might NOT need state, or maybe config?
        app.state.export_engine = ExportEngine()
        
        logger.info("All components initialized successfully.")
        
    except Exception as e:
        logger.critical(f"Startup failed: {e}")
        raise e
    
    yield
    
    logger.info("Shutting down...")
    # Add cleanup logic if needed (e.g. close DB connections)
    # app.state.milvus.close() ?
    pass


app = FastAPI(
    title=API_TITLE,
    version=API_VERSION,
    lifespan=lifespan,
    description="API for Offline Multimodal RAG System (PDF, DOCX, Images, Audio)"
)

# CORS - Allow all for development convenience
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include Routers
app.include_router(health.router,   prefix="/health",   tags=["Health"])
app.include_router(ingest.router,   prefix="/ingest",   tags=["Ingestion"])
app.include_router(query.router,    prefix="/query",    tags=["Query"])
app.include_router(export.router,   prefix="/export",   tags=["Export"])
app.include_router(session.router,  prefix="/session",  tags=["Session"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host=API_HOST,
        port=API_PORT,
        reload=True  # or use API_RELOAD from settings
    )
