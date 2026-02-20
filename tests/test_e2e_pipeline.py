"""
tests/test_e2e_pipeline.py

End-to-end pipeline tests simulating full user workflows
through the FastAPI API. Uses the test_client fixture from conftest.py
with all dependencies mocked.

Run with:
    python -m pytest tests/test_e2e_pipeline.py -v -m e2e
"""

import os
import json
from unittest.mock import MagicMock, patch

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# HEALTH CHECK FLOW
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
class TestHealthCheckFlow:
    """Verify the health-check endpoints work in sequence."""

    def test_liveness(self, test_client):
        """GET /health/ returns alive status."""
        resp = test_client.get("/health/")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "alive"

    def test_readiness(self, test_client):
        """GET /health/ready returns ready status when stores are up."""
        resp = test_client.get("/health/ready")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ready"

    def test_stats(self, test_client):
        """GET /health/stats returns system statistics."""
        resp = test_client.get("/health/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert "milvus" in data
        assert "vram" in data

    def test_full_health_sequence(self, test_client):
        """Sequential health check: liveness → readiness → stats."""
        r1 = test_client.get("/health/")
        r2 = test_client.get("/health/ready")
        r3 = test_client.get("/health/stats")

        assert r1.status_code == 200
        assert r2.status_code == 200
        assert r3.status_code == 200

        assert r1.json()["status"] == "alive"
        assert r2.json()["status"] == "ready"
        assert "vram" in r3.json()


# ═══════════════════════════════════════════════════════════════════════════
# SESSION LIFECYCLE
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
class TestSessionLifecycle:
    """Verify session management endpoints."""

    def test_get_session_history(self, test_client):
        """GET /session/{sid}/history returns messages."""
        resp = test_client.get("/session/sess-001/history")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == "sess-001"
        assert "messages" in data

    def test_delete_session(self, test_client):
        """DELETE /session/{sid} clears session data."""
        resp = test_client.delete("/session/sess-001")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert data["deleted_messages"] == 5


# ═══════════════════════════════════════════════════════════════════════════
# QUERY + CITATION FLOW
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
class TestQueryCitationFlow:
    """Verify the query → LLM → citation pipeline via API."""

    def test_query_returns_answer(self, test_client):
        """POST /query/ returns structured answer."""
        payload = {
            "query": "What is machine learning?",
            "session_id": "sess-e2e-001",
            "mode": "hybrid",
        }
        resp = test_client.post("/query/", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert len(data["answer"]) > 0

    def test_query_includes_citations(self, test_client):
        """POST /query/ response includes citation objects."""
        payload = {
            "query": "Explain neural networks",
            "session_id": "sess-e2e-002",
            "mode": "hybrid",
        }
        resp = test_client.post("/query/", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert "citations" in data
        assert isinstance(data["citations"], list)


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT FLOW
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
class TestExportFlow:
    """Verify the export pipeline via API."""

    def test_export_creates_file(self, test_client, tmp_dir):
        """POST /export/ returns a file download response."""
        import os
        # Create a real dummy file for FileResponse to read
        dummy_path = os.path.join(str(tmp_dir), "test_export.xlsx")
        with open(dummy_path, "wb") as f:
            f.write(b"PK\x03\x04" + b"\x00" * 100)  # Minimal zip header

        # Override the export engine's mock to return the real path
        from api.dependencies import get_export_engine
        from api.main import app

        mock_engine = MagicMock()
        mock_result = MagicMock()
        mock_result.file_path = dummy_path
        mock_result.success = True
        mock_engine.export.return_value = mock_result

        app.dependency_overrides[get_export_engine] = lambda: mock_engine

        payload = {
            "export_format": "xlsx",
            "query": "test query",
            "answer": "test answer",
        }

        with patch("os.path.exists", return_value=True):
            resp = test_client.post("/export/", json=payload)
        assert resp.status_code == 200


# ═══════════════════════════════════════════════════════════════════════════
# INGESTION FLOW
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
class TestIngestionFlow:
    """Verify file ingestion via API."""

    def test_reject_unsupported_extension(self, test_client):
        """POST /ingest/file rejects unsupported file types."""
        files = {"file": ("malware.exe", b"MZ\x90\x00", "application/octet-stream")}
        resp = test_client.post("/ingest/file", files=files)
        assert resp.status_code == 400
        assert "Unsupported" in resp.json()["detail"]

    def test_pdf_upload_accepted(self, test_client):
        """POST /ingest/file accepts PDF files."""
        with patch("api.routers.ingest.TextIngestor") as MockTI, \
             patch("api.routers.ingest.ImageIngestor") as MockII:

            MockTI.return_value.ingest.return_value.text_chunks = []
            MockII.return_value.ingest_from_pdf.return_value.image_chunks = []

            files = {"file": ("test.pdf", b"%PDF-1.4 test content", "application/pdf")}
            resp = test_client.post("/ingest/file", files=files)

            assert resp.status_code == 200
            data = resp.json()
            assert data["status"] == "completed"


# ═══════════════════════════════════════════════════════════════════════════
# ERROR HANDLING
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.e2e
class TestErrorHandling:
    """Verify graceful error handling in edge cases."""

    def test_empty_query_handled(self, test_client):
        """POST /query/ with empty query is handled gracefully."""
        payload = {
            "query": "",
            "session_id": "sess-err-001",
        }
        resp = test_client.post("/query/", json=payload)
        # Should either succeed with empty answer or return 4xx
        assert resp.status_code in (200, 400, 422)

    def test_missing_session_id_handled(self, test_client):
        """POST /query/ without session_id uses validation."""
        payload = {"query": "test"}
        resp = test_client.post("/query/", json=payload)
        # Should either use default session or return validation error
        assert resp.status_code in (200, 422)
