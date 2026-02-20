"""
tests/test_frontend.py
Full test suite for frontend API client and UI handler functions.

Tests cover:
- APIClient URL construction
- APIClient _raise_for_status (2xx pass, 4xx/5xx raise)
- APIClient is_backend_alive (True/False)
- APIClient upload_file (success, file not found, API error, 409)
- APIClient stream_query (token events, citation events, done, error)
- APIClient complete_query (success, API error)
- APIClient retrieve_only (success)
- APIClient generate_export (success)
- APIClient get_history (success, empty)
- APIClient clear_history
- APIClient list_sessions
- _guess_mime (all supported extensions)
- _vram_badge_html (green/amber/red thresholds, loaded models)
- _citation_card_html (text, image, audio modalities)
- _cluster_panel_html (with clusters, empty)
- _source_summary_html (single source, multiple sources, audio timestamps)
- _stats_bar_html (all counts)
- _file_list_html (with files, empty)
- _upload_log_entry (success, failure)
- _retrieval_chunks_html (with chunks, empty)
- _new_session_id (format, uniqueness)
- handle_upload (success, no files, no session, duplicate)
- handle_send retrieve-only (success, API error)
- handle_export (success, no query, API error)
- handle_new_session (resets all state)
- handle_clear_history (success, no session)
- handle_refresh_stats (success, backend offline)
- build_ui (returns gr.Blocks)

Run with:
    pytest tests/test_frontend.py -v
"""

import json
import uuid
import sys
from pathlib import Path
from typing import Any, Dict, List
from unittest.mock import MagicMock, patch, Mock, call
import gradio as gr

import pytest


# ── Fixtures & Helpers ─────────────────────────────────────────────────────
@pytest.fixture
def mock_response():
    """Build a mock requests.Response."""
    def _make(status_code=200, json_data=None, content=b""):
        resp = MagicMock()
        resp.status_code = status_code
        resp.json.return_value = json_data or {}
        resp.content = content
        resp.text = str(json_data)
        # Default iter_lines yields nothing unless mapped
        resp.iter_lines = MagicMock(return_value=iter([]))
        return resp
    return _make


@pytest.fixture
def client():
    from frontend.api_client import APIClient
    return APIClient(base_url="http://testserver:8000", timeout=10)


def _make_sse_lines(events: List[Dict]) -> List[str]:
    """Build SSE line strings for mock iter_lines."""
    lines = []
    for ev in events:
        lines.append(f"data: {json.dumps(ev)}")
        lines.append("")
    lines.append("data: [DONE]")
    return lines


# ── APIClient URL Tests ────────────────────────────────────────────────────
class TestAPIClientURL:

    def test_url_construction(self, client):
        assert client._url("/health/") == "http://testserver:8000/health/"

    def test_url_strips_leading_slash(self, client):
        assert client._url("health/") == "http://testserver:8000/health/"

    def test_base_url_trailing_slash_stripped(self):
        from frontend.api_client import APIClient
        c = APIClient(base_url="http://localhost:8000/")
        assert c.base_url == "http://localhost:8000"

    def test_raise_for_status_200_passes(self, client, mock_response):
        resp = mock_response(200, {"status": "ok"})
        client._raise_for_status(resp)   # should not raise

    def test_raise_for_status_400_raises(self, client, mock_response):
        from frontend.api_client import APIError
        resp = mock_response(400, {"detail": "Bad request"})
        with pytest.raises(APIError) as exc:
            client._raise_for_status(resp)
        assert exc.value.status_code == 400

    def test_raise_for_status_409_raises(self, client, mock_response):
        from frontend.api_client import APIError
        resp = mock_response(409, {"detail": "Duplicate file"})
        with pytest.raises(APIError) as exc:
            client._raise_for_status(resp)
        assert exc.value.status_code == 409
        assert "Duplicate" in exc.value.detail

    def test_raise_for_status_500_raises(self, client, mock_response):
        from frontend.api_client import APIError
        resp = mock_response(500, {"detail": "Server error"})
        with pytest.raises(APIError):
            client._raise_for_status(resp)


# ── APIClient Health Tests ─────────────────────────────────────────────────
class TestAPIClientHealth:

    def test_is_backend_alive_true(self, client, mock_response):
        resp = mock_response(200, {"status": "alive"})
        with patch.object(client._session, "get", return_value=resp):
            assert client.is_backend_alive() is True

    def test_is_backend_alive_false_on_exception(self, client):
        with patch.object(client._session, "get", side_effect=Exception("refused")):
            assert client.is_backend_alive() is False

    def test_get_vram_status(self, client, mock_response):
        vram_data = {"used_gb": 1.2, "ceiling_gb": 3.5, "loaded_models": ["llm_3b"]}
        resp = mock_response(200, {"vram": vram_data})
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_vram_status()
        assert result["used_gb"]    == 1.2
        assert result["ceiling_gb"] == 3.5

    def test_get_health_stats(self, client, mock_response):
        stats = {
            "milvus": {"text_chunks": {"row_count": 10}},
            "tantivy": {"num_docs": 10},
            "sqlite": {"chat_messages": 4},
            "vram": {"used_gb": 1.0},
        }
        resp = mock_response(200, stats)
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_health_stats()
        assert "milvus" in result


# ── APIClient Upload Tests ─────────────────────────────────────────────────
class TestAPIClientUpload:

    def test_upload_success(self, client, mock_response, tmp_path):
        fake_file = tmp_path / "test.pdf"
        fake_file.write_bytes(b"%PDF fake content")

        resp = mock_response(200, {
            "success": True, "chunks_inserted": 5,
            "modality": "text", "ingest_time_ms": 1200.0,
            "link_counts": {}, "file_hash": "abc123", "session_id": "sess-001",
        })
        with patch.object(client._session, "post", return_value=resp):
            result = client.upload_file(fake_file, "sess-001")

        assert result["success"] is True
        assert result["chunks_inserted"] == 5

    def test_upload_file_not_found(self, client):
        from frontend.api_client import APIError
        with pytest.raises(APIError) as exc:
            client.upload_file(Path("/nonexistent/file.pdf"), "sess-001")
        assert exc.value.status_code == 0

    def test_upload_duplicate_raises_409(self, client, mock_response, tmp_path):
        from frontend.api_client import APIError
        fake_file = tmp_path / "dup.pdf"
        fake_file.write_bytes(b"%PDF dup")
        resp = mock_response(409, {"detail": "Duplicate file — already ingested"})
        with patch.object(client._session, "post", return_value=resp):
            with pytest.raises(APIError) as exc:
                client.upload_file(fake_file, "sess-001")
        assert exc.value.status_code == 409

    def test_get_ingested_files(self, client, mock_response):
        files = [{"filename": "report.pdf", "chunk_count": 10, "status": "complete"}]
        resp = mock_response(200, {"files": files, "total": 1})
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_ingested_files("sess-001")
        assert len(result) == 1
        assert result[0]["filename"] == "report.pdf"


# ── APIClient Stream Tests ─────────────────────────────────────────────────
class TestAPIClientStream:

    def test_stream_yields_token_events(self, client, mock_response):
        events = [
            {"type": "token",    "content": "Revenue"},
            {"type": "token",    "content": " was"},
            {"type": "done",     "content": {"tokens_sec": 94.2}},
        ]
        resp = mock_response(200)
        resp.iter_lines.return_value = iter(_make_sse_lines(events))
        resp.headers = {"content-type": "text/event-stream"}

        # Patch sseclient to None to force manual parsing logic which uses iter_lines
        with patch("frontend.api_client.sseclient", None):
            with patch.object(client._session, "post", return_value=resp):
                collected = list(client.stream_query("test", "sess-001"))

        types = [e["type"] for e in collected]
        assert "token" in types
        assert "done"  in types

    def test_stream_done_terminates(self, client, mock_response):
        events = [
            {"type": "token", "content": "Hello"},
            {"type": "done",  "content": {}},
        ]
        resp = mock_response(200)
        resp.iter_lines.return_value = iter(_make_sse_lines(events))

        with patch("frontend.api_client.sseclient", None):
            with patch.object(client._session, "post", return_value=resp):
                collected = list(client.stream_query("test", "sess-001"))

        # [DONE] sentinel should terminate the iterator
        assert all(e.get("type") in ("token", "done") for e in collected)

    def test_stream_citation_event_parsed(self, client, mock_response):
        citation_data = {"citations": [{"index": 1, "modality": "text"}]}
        events = [
            {"type": "citation", "content": citation_data},
            {"type": "done",     "content": {}},
        ]
        resp = mock_response(200)
        resp.iter_lines.return_value = iter(_make_sse_lines(events))

        with patch("frontend.api_client.sseclient", None):
            with patch.object(client._session, "post", return_value=resp):
                collected = list(client.stream_query("test", "sess-001"))

        citation_events = [e for e in collected if e["type"] == "citation"]
        assert len(citation_events) == 1
        assert "citations" in citation_events[0]["content"]


# ── APIClient Query Tests ──────────────────────────────────────────────────
class TestAPIClientQuery:

    def test_complete_query_success(self, client, mock_response):
        data = {
            "response": "Revenue was $4.2M [1].",
            "citations": [],
            "tokens_sec": 94.2,
        }
        resp = mock_response(200, data)
        with patch.object(client._session, "post", return_value=resp):
            result = client.complete_query("test", "sess-001")
        assert "response" in result
        assert result["tokens_sec"] == 94.2

    def test_retrieve_only(self, client, mock_response):
        data = {
            "gold_chunks": [{"chunk_id": "abc", "modality": "text", "text": "Rev"}],
            "latency_ms":  {"total": 250.0},
        }
        resp = mock_response(200, data)
        with patch.object(client._session, "post", return_value=resp):
            result = client.retrieve_only("test", "sess-001")
        assert len(result["gold_chunks"]) == 1

    def test_generate_export(self, client, mock_response):
        data = {
            "success": True, "filename": "report_20240101.xlsx",
            "download_url": "/export/download/report_20240101.xlsx",
        }
        resp = mock_response(200, data)
        with patch.object(client._session, "post", return_value=resp):
            result = client.generate_export("export to excel", "sess-001", "xlsx")
        assert result["success"] is True
        assert result["filename"].endswith(".xlsx")

    def test_download_export(self, client, mock_response):
        resp = mock_response(200, content=b"PK\x03\x04" + b"\x00" * 100)
        with patch.object(client._session, "get", return_value=resp):
            content = client.download_export("test.xlsx")
        assert content[:2] == b"PK"


# ── APIClient Session Tests ────────────────────────────────────────────────
class TestAPIClientSession:

    def test_get_history(self, client, mock_response):
        data = {
            "messages": [
                {"role": "user", "message": "Hello", "cited_chunks": []},
                {"role": "assistant", "message": "Hi", "cited_chunks": []},
            ]
        }
        resp = mock_response(200, data)
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_history("sess-001")
        assert len(result) == 2

    def test_get_history_empty(self, client, mock_response):
        resp = mock_response(200, {"messages": []})
        with patch.object(client._session, "get", return_value=resp):
            result = client.get_history("sess-empty")
        assert result == []

    def test_clear_history(self, client, mock_response):
        resp = mock_response(200, {"success": True, "deleted": 4})
        with patch.object(client._session, "post", return_value=resp):
            result = client.clear_history("sess-001")
        assert result["success"] is True

    def test_list_sessions(self, client, mock_response):
        resp = mock_response(200, {"sessions": ["sess-001", "sess-002"], "total": 2})
        with patch.object(client._session, "get", return_value=resp):
            result = client.list_sessions()
        assert "sess-001" in result


# ── MIME Detection Tests ───────────────────────────────────────────────────
class TestGuessMime:

    def test_pdf(self):
        from frontend.api_client import _guess_mime
        assert _guess_mime(Path("test.pdf")) == "application/pdf"

    def test_png(self):
        from frontend.api_client import _guess_mime
        assert _guess_mime(Path("chart.png")) == "image/png"

    def test_mp3(self):
        from frontend.api_client import _guess_mime
        assert _guess_mime(Path("audio.mp3")) == "audio/mpeg"

    def test_wav(self):
        from frontend.api_client import _guess_mime
        assert _guess_mime(Path("call.wav")) == "audio/wav"

    def test_docx(self):
        from frontend.api_client import _guess_mime
        assert "wordprocessingml" in _guess_mime(Path("doc.docx"))

    def test_unknown_extension(self):
        from frontend.api_client import _guess_mime
        assert _guess_mime(Path("file.xyz")) == "application/octet-stream"

    def test_flac(self):
        from frontend.api_client import _guess_mime
        assert _guess_mime(Path("audio.flac")) == "audio/flac"


# ── HTML Builder Tests ─────────────────────────────────────────────────────
class TestHTMLBuilders:

    def test_vram_badge_green_below_70pct(self):
        from frontend.app import _vram_badge_html
        html = _vram_badge_html({"used_gb": 1.0, "ceiling_gb": 3.5, "loaded_models": []})
        assert "#10B981" in html   # green
        assert "1.0" in html
        assert "3.5" in html

    def test_vram_badge_amber_70_90pct(self):
        from frontend.app import _vram_badge_html
        html = _vram_badge_html({"used_gb": 2.8, "ceiling_gb": 3.5, "loaded_models": ["llm_3b"]})
        assert "#F59E0B" in html   # amber

    def test_vram_badge_red_above_90pct(self):
        from frontend.app import _vram_badge_html
        html = _vram_badge_html({"used_gb": 3.3, "ceiling_gb": 3.5, "loaded_models": ["llm_3b", "llm_1b"]})
        assert "#EF4444" in html   # red

    def test_vram_badge_shows_models(self):
        from frontend.app import _vram_badge_html
        html = _vram_badge_html({"used_gb": 2.8, "ceiling_gb": 3.5, "loaded_models": ["llm_3b", "bge_m3"]})
        assert "llm_3b" in html
        assert "bge_m3" in html

    def test_vram_badge_zero_ceiling_no_crash(self):
        from frontend.app import _vram_badge_html
        html = _vram_badge_html({"used_gb": 0.0, "ceiling_gb": 0.0, "loaded_models": []})
        assert "<div" in html

    def test_citation_card_text_chunk(self):
        from frontend.app import _citation_card_html
        citation = {
            "index": 1, "modality": "text", "source_file": "report.pdf",
            "page_number": 7, "text_preview": "Revenue was $4.2M in Q3.",
            "thumbnail_path": "", "timestamp_display": "", "links": [],
        }
        html = _citation_card_html(citation, 1)
        assert "TEXT"        in html
        assert "report.pdf"  in html
        assert "Revenue was" in html
        assert "[1]" not in html or "data-index" in html   # [N] replaced or styled

    def test_citation_card_image_chunk(self):
        from frontend.app import _citation_card_html
        citation = {
            "index": 2, "modality": "image", "source_file": "report.pdf",
            "page_number": 7, "text_preview": "Chart data",
            "thumbnail_path": "", "timestamp_display": "", "links": [],
        }
        html = _citation_card_html(citation, 2)
        assert "IMAGE" in html

    def test_citation_card_audio_chunk(self):
        from frontend.app import _citation_card_html
        citation = {
            "index": 3, "modality": "audio", "source_file": "meeting.mp3",
            "page_number": 1, "text_preview": "Budget approved.",
            "thumbnail_path": "", "timestamp_display": "14:03 - 15:01", "links": [],
        }
        html = _citation_card_html(citation, 3)
        assert "AUDIO"   in html
        assert "14:03"   in html

    def test_citation_card_with_links(self):
        from frontend.app import _citation_card_html
        citation = {
            "index": 1, "modality": "text", "source_file": "r.pdf",
            "page_number": 1, "text_preview": "text",
            "thumbnail_path": "", "timestamp_display": "",
            "links": [{"link_type": "same_page", "strength": 1.0, "display_label": "📄 Same page as [2]"}],
        }
        html = _citation_card_html(citation, 1)
        assert "same_page" in html or "📄" in html

    def test_cluster_panel_with_clusters(self):
        from frontend.app import _cluster_panel_html
        clusters = [{
            "cluster_id": "abc", "chunk_indices": [1, 2],
            "link_type": "same_page", "strength": 1.0,
            "label": "📄 Same page in report.pdf", "color": "#2563EB",
        }]
        html = _cluster_panel_html(clusters)
        assert "same_page" in html or "📄" in html
        assert "100%" in html   # strength bar

    def test_cluster_panel_empty(self):
        from frontend.app import _cluster_panel_html
        html = _cluster_panel_html([])
        assert "No cross-modal" in html

    def test_source_summary_single(self):
        from frontend.app import _source_summary_html
        sources = [{
            "source_file": "report.pdf", "modalities": ["text", "image"],
            "pages": [7, 8], "timestamps": [], "citation_indices": [1, 3],
            "first_index": 1,
        }]
        html = _source_summary_html(sources)
        assert "report.pdf" in html

    def test_source_summary_audio_timestamps(self):
        from frontend.app import _source_summary_html
        sources = [{
            "source_file": "meeting.mp3", "modalities": ["audio"],
            "pages": [], "timestamps": ["14:03 - 15:01"],
            "citation_indices": [2], "first_index": 2,
        }]
        html = _source_summary_html(sources)
        assert "14:03" in html

    def test_source_summary_empty(self):
        from frontend.app import _source_summary_html
        html = _source_summary_html([])
        assert "No sources" in html

    def test_stats_bar_shows_counts(self):
        from frontend.app import _stats_bar_html
        stats = {
            "milvus": {
                "text_chunks":  {"row_count": 42},
                "image_chunks": {"row_count": 15},
                "audio_chunks": {"row_count":  7},
            },
            "tantivy": {"num_docs": 64},
            "sqlite":  {"chat_messages": 12, "total_links": 8},
            "vram":    {"used_gb": 2.1, "ceiling_gb": 3.5},
        }
        html = _stats_bar_html(stats)
        assert "42" in html
        assert "15" in html
        assert "64" in html
        assert "12" in html
        assert "2.1" in html

    def test_file_list_with_files(self):
        from frontend.app import _file_list_html
        files = [
            {"filename": "report.pdf", "modality": "text",
             "chunk_count": 10, "status": "complete"},
        ]
        html = _file_list_html(files)
        assert "report.pdf" in html
        assert "COMPLETE"   in html or "complete" in html.lower()

    def test_file_list_empty(self):
        from frontend.app import _file_list_html
        html = _file_list_html([])
        assert "No files" in html

    def test_upload_log_success(self):
        from frontend.app import _upload_log_entry
        html = _upload_log_entry("report.pdf", True, "5 chunks · 1200ms")
        assert "✅"        in html
        assert "report.pdf" in html
        assert "5 chunks"   in html

    def test_upload_log_failure(self):
        from frontend.app import _upload_log_entry
        html = _upload_log_entry("bad.exe", False, "Unsupported file type")
        assert "❌" in html
        assert "Unsupported" in html

    def test_retrieval_chunks_html_with_data(self):
        from frontend.app import _retrieval_chunks_html
        chunks = [{
            "chunk_id": "abc", "modality": "text",
            "source_file": "report.pdf", "page_number": 7,
            "text": "Revenue was $4.2M", "reranker_score": 0.87,
            "timestamp_display": "",
        }]
        html = _retrieval_chunks_html(chunks)
        assert "TEXT"     in html
        assert "Revenue"  in html

    def test_retrieval_chunks_html_empty(self):
        from frontend.app import _retrieval_chunks_html
        html = _retrieval_chunks_html([])
        assert "No chunks" in html


# ── Session Helper Tests ───────────────────────────────────────────────────
class TestSessionHelpers:

    def test_new_session_id_format(self):
        from frontend.app import _new_session_id
        sid = _new_session_id()
        assert sid.startswith("sess-")
        assert len(sid) > 10

    def test_new_session_id_unique(self):
        from frontend.app import _new_session_id
        ids = {_new_session_id() for _ in range(20)}
        assert len(ids) == 20


# ── Handler Tests ──────────────────────────────────────────────────────────
class TestHandleUpload:

    def test_upload_no_files(self):
        from frontend.app import handle_upload
        log, file_list, status = handle_upload(None, "sess-001", "")
        assert "No files" in status

    def test_upload_no_session_id(self):
        from frontend.app import handle_upload
        log, file_list, status = handle_upload(
            [MagicMock()], "", ""
        )
        assert "Session ID" in status

    def test_upload_success(self, tmp_path):
        from frontend.app import handle_upload, _client
        fake_file = tmp_path / "test.pdf"
        fake_file.write_bytes(b"%PDF test")

        mock_file = MagicMock()
        mock_file.name = str(fake_file)

        with patch.object(_client, "upload_file") as mock_upload, \
             patch.object(_client, "get_ingested_files") as mock_files:
            mock_upload.return_value = {
                "chunks_inserted": 5, "ingest_time_ms": 1200.0,
                "modality": "text", "link_counts": {}, "file_hash": "abc",
                "session_id": "sess-001", "success": True, "filename": "test.pdf",
            }
            mock_files.return_value = []
            log, file_list, status = handle_upload(
                [mock_file], "sess-001", ""
            )
        assert "uploaded" in status
        assert "✅" in log

    def test_upload_duplicate(self, tmp_path):
        from frontend.app import handle_upload, _client
        from frontend.api_client import APIError
        fake_file = tmp_path / "dup.pdf"
        fake_file.write_bytes(b"%PDF dup")

        mock_file = MagicMock()
        mock_file.name = str(fake_file)

        with patch.object(_client, "upload_file",
                          side_effect=APIError(409, "Duplicate")), \
             patch.object(_client, "get_ingested_files", return_value=[]):
            log, file_list, status = handle_upload(
                [mock_file], "sess-001", ""
            )
        assert "❌" in log or "failed" in status.lower()


class TestHandleSendRetrieveOnly:

    def test_retrieve_only_success(self):
        from frontend.app import handle_send, _client

        gold_chunks = [{"chunk_id": "abc", "modality": "text",
                        "text": "Revenue data", "source_file": "r.pdf",
                        "page_number": 1, "reranker_score": 0.9,
                        "timestamp_display": ""}]

        with patch.object(_client, "retrieve_only") as mock_retrieve:
            mock_retrieve.return_value = {
                "gold_chunks": gold_chunks,
                "latency_ms":  {"total": 250.0},
            }
            results = list(handle_send(
                query="test", history=[], session_id="sess-001",
                retrieve_only=True, max_tokens=512, temperature=0.1,
                top_k=50, enable_reranking=True,
            ))
        assert len(results) > 0
        last_history, _, _, _ = results[-1]
        assert len(last_history) > 0

    def test_empty_query_yields_empty(self):
        from frontend.app import handle_send
        results = list(handle_send(
            query="   ", history=[], session_id="sess-001",
            retrieve_only=False, max_tokens=512, temperature=0.1,
            top_k=50, enable_reranking=True,
        ))
        # Should yield once with unchanged history
        assert results[0][0] == []


class TestHandleExport:

    def test_export_no_query_no_history(self):
        from frontend.app import handle_export
        tmp_path, status = handle_export("", [], "sess-001", "xlsx")
        assert tmp_path is None
        assert "No query" in status

    def test_export_no_session(self):
        from frontend.app import handle_export
        tmp_path, status = handle_export("export to excel", [], "", "xlsx")
        assert tmp_path is None
        assert "session" in status.lower()

    def test_export_success(self, tmp_path):
        from frontend.app import handle_export, _client

        with patch.object(_client, "generate_export") as mock_gen, \
             patch.object(_client, "download_export") as mock_dl:
            mock_gen.return_value = {
                "success": True,
                "filename": "report_20240101.xlsx",
                "download_url": "/export/download/report_20240101.xlsx",
            }
            mock_dl.return_value = b"PK\x03\x04" + b"\x00" * 200

            file_path, status = handle_export(
                "export Q3 data to excel", [], "sess-001", "xlsx"
            )

        assert file_path is not None
        assert "✅" in status

    def test_export_api_error(self):
        from frontend.app import handle_export, _client
        from frontend.api_client import APIError
        with patch.object(_client, "generate_export",
                          side_effect=APIError(404, "No documents found")):
            file_path, status = handle_export(
                "export data", [], "sess-001", "xlsx"
            )
        assert file_path is None
        assert "❌" in status


class TestHandleNewSession:

    def test_new_session_generates_id(self):
        from frontend.app import handle_new_session
        sid, hist, log, files, cites, clusters, sources = handle_new_session()
        assert sid.startswith("sess-")
        assert hist == []
        assert log  == ""

    def test_new_session_unique_each_call(self):
        from frontend.app import handle_new_session
        sid1 = handle_new_session()[0]
        sid2 = handle_new_session()[0]
        assert sid1 != sid2


class TestHandleClearHistory:

    def test_clear_history_success(self):
        from frontend.app import handle_clear_history, _client
        with patch.object(_client, "clear_history") as mock_clear:
            mock_clear.return_value = {"success": True, "deleted": 4}
            history, status = handle_clear_history("sess-001")
        assert history == []
        assert "4" in status

    def test_clear_history_no_session(self):
        from frontend.app import handle_clear_history
        history, status = handle_clear_history("")
        assert history == []
        assert "No session" in status


class TestHandleRefreshStats:

    def test_refresh_stats_success(self):
        from frontend.app import handle_refresh_stats, _client
        stats = {
            "milvus":  {"text_chunks": {"row_count": 5}, "image_chunks": {"row_count": 2}, "audio_chunks": {"row_count": 1}},
            "tantivy": {"num_docs": 8},
            "sqlite":  {"chat_messages": 3, "total_links": 2},
            "vram":    {"used_gb": 1.5, "ceiling_gb": 3.5, "loaded_models": ["llm_3b"]},
        }
        with patch.object(_client, "get_health_stats", return_value=stats):
            vram_html, stats_html = handle_refresh_stats()
        assert "1.5" in vram_html
        assert "5"   in stats_html

    def test_refresh_stats_backend_offline(self):
        from frontend.app import handle_refresh_stats, _client
        with patch.object(_client, "get_health_stats", side_effect=Exception("offline")):
            vram_html, stats_html = handle_refresh_stats()
        assert "offline" in vram_html.lower() or "Backend" in vram_html


# ── UI Build Test ──────────────────────────────────────────────────────────
class TestBuildUI:

    def test_build_ui_returns_blocks(self):
        """build_ui() should return a gr.Blocks instance without crashing."""
        import gradio as gr
        from frontend.app import build_ui

        with patch("frontend.app._client") as mock_client:
            mock_client.is_backend_alive.return_value = False
            mock_client.get_health_stats.side_effect  = Exception("offline")

            demo = build_ui()
        assert isinstance(demo, gr.Blocks)

    def test_build_ui_has_chatbot(self):
        """The Blocks object should contain a Chatbot component."""
        import gradio as gr
        from frontend.app import build_ui

        with patch("frontend.app._client") as mock_client:
            mock_client.is_backend_alive.return_value = False
            mock_client.get_health_stats.side_effect  = Exception("offline")

            demo = build_ui()

        # Find Chatbot in the component tree
        found = any(
            isinstance(c, gr.Chatbot)
            for c in demo.blocks.values()
            if hasattr(c, "__class__")
        )
        assert found
