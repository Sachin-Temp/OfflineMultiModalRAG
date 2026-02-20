"""
api/routers/session.py

Session management endpoints.

GET  /session/{session_id}/history
  Return last N chat turns for a session.

DELETE /session/{session_id}
  Delete all chat history and indexed chunks for a session.

GET /session/list
  Return all session IDs that have history.

POST /session/{session_id}/clear_history
  Clear chat history but keep indexed documents.
"""

import asyncio
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, status
from loguru import logger
from pydantic import BaseModel, Field

from api.dependencies import get_milvus, get_sqlite
from modules.indexing.milvus_store import MilvusStore
from modules.indexing.sqlite_store import SQLiteStore

router = APIRouter()


class HistoryResponse(BaseModel):
    session_id: str
    messages:   list
    total:      int


@router.get(
    "/{session_id}/history",
    response_model=HistoryResponse,
    summary="Get chat history for a session",
)
async def get_history(
    session_id: str,
    last_n:     int = 10,
    sqlite:     SQLiteStore = Depends(get_sqlite),
):
    """
    Return the last N message turns for a session.
    Messages are returned in chronological order (oldest first).
    """
    if not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="session_id cannot be empty",
        )

    last_n = max(1, min(last_n, 100))   # clamp 1–100

    try:
        messages = sqlite.get_history(session_id, last_n=last_n)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve history: {e}",
        )

    return HistoryResponse(
        session_id=session_id,
        messages=messages,
        total=len(messages),
    )


@router.delete(
    "/{session_id}",
    summary="Delete all data for a session",
    description=(
        "Deletes chat history AND all indexed chunks (Milvus) for this session. "
        "Tantivy does not support session-level deletion — those entries remain "
        "but will return no results after Milvus chunks are gone. "
    ),
)
async def delete_session(
    session_id: str,
    milvus: MilvusStore  = Depends(get_milvus),
    sqlite: SQLiteStore  = Depends(get_sqlite),
):
    """
    Full session deletion: chat history + Milvus chunks.
    """
    if not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="session_id cannot be empty",
        )

    loop = asyncio.get_event_loop()

    # Delete Milvus chunks
    try:
        milvus_counts = await loop.run_in_executor(
            None, lambda: milvus.delete_by_session(session_id)
        )
    except Exception as e:
        logger.warning(f"Milvus session deletion failed (non-fatal): {e}")
        milvus_counts = {"text": 0, "image": 0, "audio": 0}

    # Delete SQLite history
    try:
        deleted_messages = sqlite.delete_session_history(session_id)
    except Exception as e:
        logger.warning(f"SQLite history deletion failed (non-fatal): {e}")
        deleted_messages = 0

    logger.info(
        f"Session {session_id} deleted: "
        f"milvus={milvus_counts} messages={deleted_messages}"
    )

    return {
        "success":          True,
        "session_id":       session_id,
        "deleted_messages": deleted_messages,
        "deleted_chunks":   milvus_counts,
    }


@router.get(
    "/list",
    summary="List all sessions with chat history",
)
async def list_sessions(sqlite: SQLiteStore = Depends(get_sqlite)):
    """
    Return all session IDs that have at least one chat message.
    Sessions are ordered by most recent activity.
    """
    try:
        session_ids = sqlite.get_session_ids()
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list sessions: {e}",
        )

    return {
        "sessions": session_ids,
        "total":    len(session_ids),
    }


@router.post(
    "/{session_id}/clear_history",
    summary="Clear chat history but keep documents",
    description=(
        "Clears conversation history for a session without deleting "
        "the indexed documents. Useful for starting a new conversation "
        "about the same uploaded files."
    ),
)
async def clear_history(
    session_id: str,
    sqlite: SQLiteStore = Depends(get_sqlite),
):
    """
    Delete chat messages only — indexed documents remain.
    """
    if not session_id.strip():
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="session_id cannot be empty",
        )

    try:
        deleted = sqlite.delete_session_history(session_id)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to clear history: {e}",
        )

    return {
        "success":    True,
        "session_id": session_id,
        "deleted":    deleted,
        "message":    f"Cleared {deleted} messages. Indexed documents unchanged.",
    }
