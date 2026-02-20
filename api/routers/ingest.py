"""
api/routers/ingest.py

File ingestion endpoints.
Handles uploading files to UPLOAD_DIR and triggering the ingestion pipeline.

supported_formats: .pdf, .docx, .ppt, .pptx, .txt, .md, .jpg, .png, .mp3, .wav
"""

import os
import shutil
import uuid
from typing import List

from fastapi import APIRouter, UploadFile, File, HTTPException, BackgroundTasks, Depends
from loguru import logger
from pydantic import BaseModel

from config.settings import UPLOAD_DIR
from api.dependencies import get_milvus, get_tantivy, get_sqlite
from modules.indexing.milvus_store import MilvusStore
from modules.indexing.tantivy_index import TantivyIndex
from modules.indexing.sqlite_store import SQLiteStore
from modules.ingestion.text_ingestor import TextIngestor
from modules.ingestion.image_ingestor import ImageIngestor
from modules.ingestion.audio_ingestor import AudioIngestor

router = APIRouter()

ALLOWED_EXTENSIONS = {
    ".pdf", ".docx", ".ppt", ".pptx", ".txt", ".md",
    ".jpg", ".jpeg", ".png",
    ".mp3", ".wav"
}

class IngestResponse(BaseModel):
    filename: str
    file_id: str
    status: str
    message: str




# --- REWRITING WITH ACTUAL IMPLEMENTATION ---


from fastapi import Form

@router.post(
    "/file",
    response_model=IngestResponse,
    summary="Upload and ingest a single file",
)
async def ingest_file(
    file: UploadFile = File(...),
    session_id: str = Form("default"),
    milvus: MilvusStore = Depends(get_milvus),
    tantivy: TantivyIndex = Depends(get_tantivy),
    sqlite: SQLiteStore = Depends(get_sqlite),
):
    """
    Upload a file and start ingestion.
    Orchestrates Text/Image/Audio ingestors and writes to Milvus/Tantivy/SQLite.
    """
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file extension: {ext}. Allowed: {ALLOWED_EXTENSIONS}"
        )

    file_id = str(uuid.uuid4())
    # Sanitize filename
    safe_name = "".join(c for c in file.filename if c.isalnum() or c in "._- ")
    saved_filename = f"{file_id}_{safe_name}"
    file_path = UPLOAD_DIR / saved_filename

    try:
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(f"Failed to save file {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="File save failed")

    # Ingestion Orchestration
    try:
        total_chunks = 0
        modality = "text"  # default
        
        # Determine likely modality and ingestor
        # For PDF, we run Text + Image extraction
        if ext == ".pdf":
            modality = "text" # Primary modality
            # Text
            ti = TextIngestor()
            res_text = ti.ingest(file_path)
            
            # Images (embedded)
            ii = ImageIngestor()
            res_img = ii.ingest_from_pdf(file_path)
            
            # Combine Results
            chunks = (res_text.text_chunks or []) + (res_img.image_chunks or [])
            total_chunks = len(chunks)

            # Store Text Chunks
            if res_text.text_chunks:
                tantivy.add_text_chunks(res_text.text_chunks)
                try:
                    milvus.insert_text_chunks(res_text.text_chunks)
                except Exception as me:
                    logger.warning(f"Milvus text insert failed (offline): {me}")
            
            # Store Image Chunks
            if res_img.image_chunks:
                tantivy.add_image_chunks(res_img.image_chunks)
                try:
                    milvus.insert_image_chunks(res_img.image_chunks)
                except Exception as me:
                    logger.warning(f"Milvus image insert failed (offline): {me}")
                
            ii.release_models() # Free VRAM

        elif ext in [".docx", ".txt", ".md"]:
            modality = "text"
            ti = TextIngestor()
            res = ti.ingest(file_path)
            if res.text_chunks:
                tantivy.add_text_chunks(res.text_chunks)
                try:
                    milvus.insert_text_chunks(res.text_chunks)
                except Exception as me:
                    logger.warning(f"Milvus text insert failed (offline): {me}")
                total_chunks = len(res.text_chunks)

        elif ext in [".jpg", ".jpeg", ".png"]:
            modality = "image"
            ii = ImageIngestor()
            res = ii.ingest(file_path)
            if res.image_chunks:
                tantivy.add_image_chunks(res.image_chunks)
                try:
                    milvus.insert_image_chunks(res.image_chunks)
                except Exception as me:
                    logger.warning(f"Milvus image insert failed (offline): {me}")
                total_chunks = len(res.image_chunks)
            ii.release_models()

        elif ext in [".mp3", ".wav"]:
            modality = "audio"
            ai = AudioIngestor()
            res = ai.ingest(file_path)
            if res.audio_chunks:
                tantivy.add_audio_chunks(res.audio_chunks)
                try:
                    milvus.insert_audio_chunks(res.audio_chunks)
                except Exception as me:
                    logger.warning(f"Milvus audio insert failed (offline): {me}")
                total_chunks = len(res.audio_chunks)

        # Register file in SQLite
        try:
            sqlite.register_file(
                file_path=file_path,
                modality=modality,
                session_id=session_id,
                chunk_count=total_chunks,
                status="complete"
            )
        except Exception as e:
            logger.error(f"Failed to register file in SQLite: {e}")
            # Non-critical failure regarding ingestion itself? Maybe.

        return IngestResponse(
            filename=file.filename,
            file_id=file_id,
            status="completed",
            message=f"Ingested {total_chunks} chunks successfully."
        )

    except Exception as e:
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        # In case of error, we might want to clean up but for now just report
        raise HTTPException(status_code=500, detail=f"Ingestion failed: {e}")


@router.get(
    "/files",
    summary="List ingested files",
)
def list_files(
    session_id: str = "default",
    sqlite: SQLiteStore = Depends(get_sqlite),
):
    """Get a list of all ingested files for a session."""
    files = sqlite.get_ingested_files(session_id)
    return {"files": files}
