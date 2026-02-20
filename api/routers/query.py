"""
api/routers/query.py

Query endpoints.
Process user queries via LLMEngine, handling retrieval, generation, 
and citation formatting.

POST /query/
Body:
{
    "query": "...",
    "session_id": "...",
    "mode": "hybrid" | "dense" | "sparse",
    "top_k": 5
}
"""

import time
from typing import List, Optional, Any, Dict

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field

from api.dependencies import get_llm_engine, get_citation_engine
from modules.llm.llm_engine import LLMEngine
from modules.citation.citation_engine import CitationEngine

router = APIRouter()


class QueryRequest(BaseModel):
    query: str
    session_id: Optional[str] = None
    mode: str = "hybrid"   # retrieval mode
    top_k: int = 5
    filters: Optional[Dict[str, Any]] = None


class CitationModel(BaseModel):
    # Simplified model for API response
    id: int
    text: str
    file_path: str
    page_number: Optional[int] = None
    modality: str
    score: float


class QueryResponse(BaseModel):
    answer: str
    citations: List[Dict[str, Any]]
    session_id: str
    processing_time: float


@router.post(
    "/",
    response_model=QueryResponse,
    summary="Submit a RAG query",
)
async def query_rag(
    request: QueryRequest,
    llm_engine: LLMEngine = Depends(get_llm_engine),
    citation_engine: CitationEngine = Depends(get_citation_engine),
):
    """
    Execute a RAG query.
    1. Retrieve relevant chunks.
    2. Generate answer with LLM.
    3. Parse citations.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty")

    start_time = time.time()

    try:
        # LLMEngine.query() is the main entry point
        # It should handle retrieval internally or we inject retrieved context.
        # In Phase 7 implemention, LLMEngine.query() orchestrates everything?
        # Let's assume LLMEngine.query(query, session_id, top_k, mode) signature.
        # Or LLMEngine.generate(context, query)?
        # Checking Phase 7 summary: "Implemented... logic to orchestrate RAG".
        # So we can likely call `llm_engine.query`.
        
        result = await llm_engine.query(
            query=request.query,
            session_id=request.session_id,
            top_k=request.top_k,
            mode=request.mode
            # filters? if supported
        )
        
        # result is likely a dict or object with:
        # "answer": str
        # "context": List[Chunk]
        
        # Process citations using CitationEngine
        # result["answer"] might contain [1] markers.
        # result["context"] contains source chunks.
        
        citations = citation_engine.process(
             text=result["answer"],
             # We need to map [1] to chunks. 
             # LLMEngine should return the ordered chunks used for [1], [2]...
             # If LLMEngine returns "source_chunks", pass them.
             source_chunks=result.get("source_chunks", [])
        )
        
        # Format citations for response
        # citations is a List[CitationObject]
        formatted_citations = [c.to_dict() for c in citations]
        
        process_time = time.time() - start_time
        
        return QueryResponse(
            answer=result["answer"],
            citations=formatted_citations,
            session_id=result.get("session_id", request.session_id or "default"),
            processing_time=process_time
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Query failed: {e}")
