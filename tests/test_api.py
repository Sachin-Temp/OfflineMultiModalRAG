"""
tests/test_api.py

Functional tests for FastAPI endpoints.
Uses dependency overrides to mock heavy engines (LLM, Retrieval, etc.)
so tests run quickly without GPU/DB.
"""

import os
import pytest
from fastapi.testclient import TestClient
from unittest.mock import MagicMock, AsyncMock

from api.main import app
from api.dependencies import (
    get_milvus, get_tantivy, get_sqlite,
    get_retrieval_engine, get_llm_engine,
    get_citation_engine, get_export_engine
)

client = TestClient(app)

# --- Mocks ---

@pytest.fixture
def mock_milvus():
    m = MagicMock()
    m.get_collection_stats.return_value = {"test_collection": {"status": "ok"}}
    m.delete_by_session.return_value = {"text": 1, "image": 0, "audio": 0}
    return m

@pytest.fixture
def mock_tantivy():
    m = MagicMock()
    m.get_stats.return_value = {"status": "ok", "doc_count": 100}
    return m

@pytest.fixture
def mock_sqlite():
    m = MagicMock()
    m.get_stats.return_value = {"status": "ok", "counts": {"conversations": 5}}
    m.get_history.return_value = [{"role": "user", "content": "hello"}]
    m.delete_session_history.return_value = 5  # count
    m.get_session_ids.return_value = ["sess_1", "sess_2"]
    return m

@pytest.fixture
def mock_llm_engine():
    m = AsyncMock()
    # Mock query result
    m.query.return_value = {
        "answer": "This is a mocked answer [1].",
        "context": [],
        "session_id": "test_session",
        "source_chunks": [{"id": 1, "text": "Source text", "modality": "text"}]
    }
    return m

@pytest.fixture
def mock_citation_engine():
    m = MagicMock()
    # Return dummy CitationObject
    # We need a predictable object
    # Let's mock the process method to return a list of mocks that have .to_dict()
    
    c_obj = MagicMock()
    c_obj.to_dict.return_value = {
        "id": 1, 
        "text": "Source text", 
        "file_path": "doc.pdf",
        "page_number": 1,
        "modality": "text",
        "score": 0.9
    }
    m.process.return_value = [c_obj]
    return m

@pytest.fixture
def mock_export_engine():
    m = MagicMock()
    # Mock validation result
    res = MagicMock()
    res.file_path = "test_export.xlsx" 
    m.export.return_value = res
    return m


# --- Dependency Overrides ---

@pytest.fixture(autouse=True)
def override_dependencies(
    mock_milvus, mock_tantivy, mock_sqlite,
    mock_llm_engine, mock_citation_engine, mock_export_engine
):
    app.dependency_overrides[get_milvus] = lambda: mock_milvus
    app.dependency_overrides[get_tantivy] = lambda: mock_tantivy
    app.dependency_overrides[get_sqlite] = lambda: mock_sqlite
    app.dependency_overrides[get_llm_engine] = lambda: mock_llm_engine
    app.dependency_overrides[get_citation_engine] = lambda: mock_citation_engine
    app.dependency_overrides[get_export_engine] = lambda: mock_export_engine
    yield
    app.dependency_overrides = {}


# --- Tests ---

def test_health_live():
    response = client.get("/health/")
    assert response.status_code == 200
    assert response.json()["status"] == "alive"

def test_health_ready(mock_milvus):
    response = client.get("/health/ready")
    assert response.status_code == 200
    assert response.json()["status"] == "ready"
    mock_milvus.get_collection_stats.assert_called_once()

def test_health_stats():
    response = client.get("/health/stats")
    assert response.status_code == 200
    data = response.json()
    assert "milvus" in data
    assert "vram" in data

def test_session_history_get():
    response = client.get("/session/sess_1/history")
    assert response.status_code == 200
    data = response.json()
    assert data["session_id"] == "sess_1"
    assert len(data["messages"]) == 1

def test_session_delete():
    response = client.delete("/session/sess_1")
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert data["deleted_messages"] == 5

def test_query_flow(mock_llm_engine, mock_citation_engine):
    payload = {
        "query": "What is AI?",
        "session_id": "test_sess",
        "mode": "hybrid"
    }
    response = client.post("/query/", json=payload)
    assert response.status_code == 200
    data = response.json()
    
    assert data["answer"] == "This is a mocked answer [1]."
    assert len(data["citations"]) == 1
    assert data["citations"][0]["id"] == 1
    
    mock_llm_engine.query.assert_awaited_once()
    mock_citation_engine.process.assert_called_once()

def test_export_xlsx(mock_export_engine):
    # Mock os.path.exists for the file check in router
    # We can't easily mock os.path.exists here without patching it inside the module
    # But ExportEngine returns a path.
    # The router checks `if not os.path.exists(result.file_path)`.
    # So we must create a dummy file or patch os.path.exists.
    # Creating a dummy file is cleaner for integration test, but patch is faster.
    
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(os.path, "exists", lambda x: True)
        
        # Test request
        payload = {
            "export_format": "xlsx",
            "query": "q",
            "answer": "a"
        }
        # The router uses FileResponse. 
        # TestClient handles FileResponse by reading it? 
        # FileResponse reads the file. If file doesn't exist, it errors.
        # So we actually need a file.
        
        # Let's create a dummy file
        with open("test_export.xlsx", "w") as f:
            f.write("dummy content")
            
        try:
            response = client.post("/export/", json=payload)
            assert response.status_code == 200
            # Content-Type check?
            # assert response.headers["content-type"] == ...
        finally:
            if os.path.exists("test_export.xlsx"):
                os.remove("test_export.xlsx")

def test_ingest_upload_refused_extension():
    files = {"file": ("bad.exe", b"content", "application/octet-stream")}
    response = client.post("/ingest/file", files=files) 
    assert response.status_code == 400
    assert "Unsupported file extension" in response.json()["detail"]


from unittest.mock import patch

def test_ingest_pdf_success(mock_milvus, mock_tantivy, mock_sqlite):
    # Patch the ingestors used in the router
    with patch("api.routers.ingest.TextIngestor") as MockTextIngestor, \
         patch("api.routers.ingest.ImageIngestor") as MockImageIngestor:
        
        # Setup mocks
        mock_ti_instance = MockTextIngestor.return_value
        mock_ti_instance.ingest.return_value.text_chunks = []
        
        mock_ii_instance = MockImageIngestor.return_value
        mock_ii_instance.ingest_from_pdf.return_value.image_chunks = []
        
        files = {"file": ("test.pdf", b"%PDF-1.4...", "application/pdf")}
        response = client.post("/ingest/file", files=files)
        
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        
        # Verify calls
        mock_ti_instance.ingest.assert_called_once()
        mock_ii_instance.ingest_from_pdf.assert_called_once()
