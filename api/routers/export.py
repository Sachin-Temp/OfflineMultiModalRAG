"""
api/routers/export.py

Export endpoints.
Triggers ExportEngine to generate files (XLSX, DOCX, PPTX) from specific
query results or sessions, and returns the file as a download.

POST /export/
Body:
{
    "export_type": "xlsx" | "docx" | "pptx",
    "query": "...",
    "answer": "...",
    "citations": [ ... ],
    "session_id": "...",
    "include_images": true
}
"""

import os
from typing import Optional, Dict, Any, List

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel

from api.dependencies import get_export_engine
from modules.export.export_engine import ExportEngine
from config.settings import OUTPUT_DIR

router = APIRouter()

class ExportRequest(BaseModel):
    export_format: str  # xlsx, docx, pptx, csv
    query: str
    answer: str
    citations: List[Dict[str, Any]] = []
    session_id: Optional[str] = None
    context_tree: Optional[Dict[str, Any]] = None  # Optional full context

@router.post(
    "/",
    summary="Export RAG result to file",
    description="Generates a downloadable file (Office/CSV) for a given RAG answer.",
)
async def export_rag_result(
    request: ExportRequest,
    export_engine: ExportEngine = Depends(get_export_engine),
):
    """
    Generate and download export file.
    """
    if request.export_format.lower() not in ["xlsx", "docx", "pptx", "csv"]:
        raise HTTPException(status_code=400, detail="Invalid export_format. Use: xlsx, docx, pptx, csv")

    # Construct standard data dict expected by ExportEngine
    # See modules/export/export_engine.py:ExportEngine.export
    export_data = {
        "format": request.export_format,
        "query": request.query,
        "answer": request.answer,
        "citations": request.citations,
        "session_id": request.session_id,
        "context": request.context_tree
    }

    try:
        # ExportEngine.export returns an ExportResult object with .file_path
        # It's a synchronous method usually, but we can run it.
        # If it's heavy, fastAPI handles it or we use loop.run_in_executor
        # For simplicity in this demo:
        
        result = export_engine.export(export_data)
        
        if not result or not result.file_path or not os.path.exists(result.file_path):
             raise HTTPException(status_code=500, detail="Export generation failed (no file created)")

        # Return file download
        # filename is usually set by engine.
        return FileResponse(
            path=result.file_path,
            filename=os.path.basename(result.file_path),
            media_type="application/octet-stream" 
            # or usage specific: 
            # xlsx: application/vnd.openxmlformats-officedocument.spreadsheetml.sheet
            # but octet-stream is safe generic
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Export failed: {str(e)}")
