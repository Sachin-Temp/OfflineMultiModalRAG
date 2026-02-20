"""
tests/conftest.py

Shared fixtures for integration and end-to-end tests.
Provides mock stores, engines, sample data, and a FastAPI TestClient
with all dependencies overridden for deterministic testing.
"""

import os
import json
import uuid
import tempfile
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional
from unittest.mock import MagicMock, AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

from models.schemas import (
    TextChunk, ImageChunk, AudioChunk,
    Modality, IngestionResult, BBox, WordTimestamp,
)
from modules.retrieval.retrieval_engine import GoldChunk, RetrievalResult


# ── Temporary Directory ────────────────────────────────────────────────────
@pytest.fixture
def tmp_dir(tmp_path):
    """Provide a clean temporary directory for file I/O tests."""
    return tmp_path


# ── SQLite Fixture (Real, Ephemeral) ───────────────────────────────────────
@pytest.fixture
def tmp_sqlite(tmp_path):
    """Create a real SQLiteStore backed by a temporary database file."""
    from modules.indexing.sqlite_store import SQLiteStore
    db_path = tmp_path / "test_rag.db"
    store = SQLiteStore(db_path=db_path)
    store.initialize()
    return store


# ── Mock Stores ────────────────────────────────────────────────────────────
@pytest.fixture
def mock_milvus_store():
    """MagicMock MilvusStore with sensible defaults."""
    m = MagicMock()
    m.get_collection_stats.return_value = {
        "text_chunks": {"count": 10, "status": "ok"},
        "image_chunks": {"count": 5, "status": "ok"},
        "audio_chunks": {"count": 3, "status": "ok"},
    }
    m.search_text.return_value = []
    m.search_images.return_value = []
    m.search_audio.return_value = []
    m.delete_by_session.return_value = {"text": 0, "image": 0, "audio": 0}
    m.insert_text_chunks.return_value = 0
    m.insert_image_chunks.return_value = 0
    m.insert_audio_chunks.return_value = 0
    return m


@pytest.fixture
def mock_tantivy():
    """MagicMock TantivyIndex."""
    m = MagicMock()
    m.get_stats.return_value = {"status": "ok", "doc_count": 100}
    m.search.return_value = []
    m.index_chunks.return_value = 0
    return m


@pytest.fixture
def mock_sqlite_store():
    """MagicMock SQLiteStore with chat history and link stubs."""
    m = MagicMock()
    m.get_stats.return_value = {"status": "ok", "counts": {"conversations": 3}}
    m.get_history.return_value = [
        {"role": "user", "message": "What is machine learning?", "cited_chunks": [], "timestamp": "2026-01-01T00:00:00"},
        {"role": "assistant", "message": "ML is a subset of AI [1].", "cited_chunks": ["chunk-001"], "timestamp": "2026-01-01T00:00:01"},
    ]
    m.get_session_ids.return_value = ["sess-001", "sess-002", "sess-003"]
    m.delete_session_history.return_value = 5
    m.add_message.return_value = 1
    m.get_linked_chunks.return_value = []
    m.initialize.return_value = None
    return m


# ── Mock Engines ───────────────────────────────────────────────────────────
@pytest.fixture
def mock_retrieval_engine():
    """MagicMock RetrievalEngine that returns a canned RetrievalResult."""
    m = MagicMock()
    m.retrieve.return_value = RetrievalResult(query="test query")
    return m


@pytest.fixture
def mock_llm_engine():
    """AsyncMock LLMEngine with a canned query response."""
    m = AsyncMock()
    m.query.return_value = {
        "answer": "This is a test answer based on the documents [1].",
        "context": [],
        "session_id": "test-session",
        "source_chunks": [{"id": 1, "text": "Source text", "modality": "text"}],
    }
    m.generate_stream.return_value = iter(["This ", "is ", "a ", "test ", "answer ", "[1]."])
    return m


@pytest.fixture
def mock_citation_engine():
    """MagicMock CitationEngine with a canned citation object."""
    m = MagicMock()
    c_obj = MagicMock()
    c_obj.to_dict.return_value = {
        "index": 1,
        "chunk_id": "chunk-001",
        "modality": "text",
        "source_file": "report.pdf",
        "page_number": 3,
        "text_preview": "Machine learning is...",
        "score": 0.92,
    }
    m.process.return_value = [c_obj]
    return m


@pytest.fixture
def mock_export_engine():
    """MagicMock ExportEngine that returns a canned ExportResult."""
    m = MagicMock()
    result = MagicMock()
    result.file_path = "test_export.xlsx"
    result.export_type = "xlsx"
    result.filename = "test_export.xlsx"
    result.file_size_bytes = 1024
    result.success = True
    result.error = None
    result.to_dict.return_value = {
        "file_path": "test_export.xlsx",
        "export_type": "xlsx",
        "filename": "test_export.xlsx",
        "file_size_bytes": 1024,
        "success": True,
    }
    m.export.return_value = result
    return m


# ── Sample Data ────────────────────────────────────────────────────────────
@pytest.fixture
def sample_text_chunks():
    """Two realistic TextChunk objects."""
    return [
        TextChunk(
            text="Machine learning is a subset of artificial intelligence that enables systems to learn from data.",
            source_file="report.pdf",
            page_number=1,
            chunk_id="txt-001",
            session_id="sess-001",
            char_start=0,
            char_end=95,
        ),
        TextChunk(
            text="Neural networks are inspired by the human brain and consist of interconnected layers of nodes.",
            source_file="report.pdf",
            page_number=2,
            chunk_id="txt-002",
            session_id="sess-001",
            char_start=0,
            char_end=94,
        ),
    ]


@pytest.fixture
def sample_image_chunks():
    """Two realistic ImageChunk objects."""
    return [
        ImageChunk(
            image_path="/data/media/img_001.png",
            source_file="report.pdf",
            page_number=1,
            chunk_id="img-001",
            session_id="sess-001",
            ocr_text="Figure 1: ML Pipeline",
            llm_description="A flowchart showing the machine learning pipeline.",
            thumbnail_path="/data/media/thumbs/img_001_thumb.png",
        ),
        ImageChunk(
            image_path="/data/media/img_002.png",
            source_file="slides.pptx",
            page_number=3,
            chunk_id="img-002",
            session_id="sess-001",
            ocr_text="Neural Network Architecture",
            llm_description="Diagram of a deep neural network with multiple layers.",
        ),
    ]


@pytest.fixture
def sample_audio_chunks():
    """Two realistic AudioChunk objects."""
    return [
        AudioChunk(
            text="Today we will discuss the fundamentals of machine learning and its applications.",
            audio_file="lecture.mp3",
            start_time=0.0,
            end_time=30.0,
            chunk_id="aud-001",
            session_id="sess-001",
            timestamp_display="00:00 - 00:30",
            speaker_id="speaker_1",
            confidence=0.95,
        ),
        AudioChunk(
            text="Deep learning has revolutionized natural language processing and computer vision.",
            audio_file="lecture.mp3",
            start_time=30.0,
            end_time=60.0,
            chunk_id="aud-002",
            session_id="sess-001",
            timestamp_display="00:30 - 01:00",
            speaker_id="speaker_1",
            confidence=0.93,
        ),
    ]


@pytest.fixture
def gold_chunks():
    """Three GoldChunk objects (text, image, audio) for citation/LLM testing."""
    return [
        GoldChunk(
            chunk_id="txt-001",
            modality="text",
            text="Machine learning is a subset of artificial intelligence.",
            source_file="report.pdf",
            page_number=1,
            reranker_score=0.95,
            rrf_score=0.8,
        ),
        GoldChunk(
            chunk_id="img-001",
            modality="image",
            text="Figure 1: ML Pipeline",
            source_file="report.pdf",
            page_number=1,
            reranker_score=0.88,
            rrf_score=0.7,
            image_path="/data/media/img_001.png",
            thumbnail_path="/data/media/thumbs/img_001_thumb.png",
            linked_chunk_ids=["txt-001"],
            link_types=["same_page"],
        ),
        GoldChunk(
            chunk_id="aud-001",
            modality="audio",
            text="Today we will discuss the fundamentals of machine learning.",
            source_file="lecture.mp3",
            page_number=0,
            reranker_score=0.82,
            rrf_score=0.65,
            start_time=0.0,
            end_time=30.0,
            timestamp_display="00:00 - 00:30",
        ),
    ]


@pytest.fixture
def retrieval_result(gold_chunks):
    """A RetrievalResult pre-loaded with gold_chunks."""
    return RetrievalResult(
        query="What is machine learning?",
        gold_chunks=gold_chunks,
        total_candidates=50,
    )


# ── FastAPI Test Client ───────────────────────────────────────────────────
@pytest.fixture
def test_client(
    mock_milvus_store,
    mock_tantivy,
    mock_sqlite_store,
    mock_llm_engine,
    mock_citation_engine,
    mock_export_engine,
):
    """
    FastAPI TestClient with all dependencies overridden.
    Uses the real app from api.main with mocked backends.
    """
    from api.main import app
    from api.dependencies import (
        get_milvus, get_tantivy, get_sqlite,
        get_retrieval_engine, get_llm_engine,
        get_citation_engine, get_export_engine,
    )

    mock_retrieval = MagicMock()
    mock_retrieval.retrieve.return_value = RetrievalResult(query="test")

    app.dependency_overrides[get_milvus] = lambda: mock_milvus_store
    app.dependency_overrides[get_tantivy] = lambda: mock_tantivy
    app.dependency_overrides[get_sqlite] = lambda: mock_sqlite_store
    app.dependency_overrides[get_retrieval_engine] = lambda: mock_retrieval
    app.dependency_overrides[get_llm_engine] = lambda: mock_llm_engine
    app.dependency_overrides[get_citation_engine] = lambda: mock_citation_engine
    app.dependency_overrides[get_export_engine] = lambda: mock_export_engine

    client = TestClient(app)
    yield client

    app.dependency_overrides = {}
