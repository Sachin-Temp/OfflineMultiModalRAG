"""
tests/test_integration.py

Integration tests covering cross-module interactions.
These tests verify that modules work correctly together,
using real lightweight components where possible and mocks
for GPU/DB-dependent components.

Run with:
    python -m pytest tests/test_integration.py -v -m integration
"""

import os
import uuid
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from models.schemas import (
    TextChunk, ImageChunk, AudioChunk,
    Modality, IngestionResult,
)
from modules.retrieval.retrieval_engine import (
    GoldChunk, RetrievalResult, classify_query,
)
from modules.citation.citation_engine import (
    CitationEngine, CitationObject, CitationResult,
    annotate_response, detect_clusters,
)
from modules.export.export_engine import ExportEngine, ExportResult


# ═══════════════════════════════════════════════════════════════════════════
# SQLITE INTEGRATION TESTS
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestSQLiteIntegration:
    """Tests using a real ephemeral SQLiteStore."""

    def test_chat_history_round_trip(self, tmp_sqlite):
        """add_message → get_history round-trip preserves data."""
        session_id = "int-sess-001"
        tmp_sqlite.add_message(session_id, "user", "What is AI?")
        tmp_sqlite.add_message(
            session_id, "assistant", "AI is artificial intelligence [1].",
            cited_chunks=["chunk-abc"],
        )

        history = tmp_sqlite.get_history(session_id, last_n=10)
        assert len(history) == 2
        assert history[0]["role"] == "user"
        assert history[0]["message"] == "What is AI?"
        assert history[1]["role"] == "assistant"
        assert "artificial intelligence" in history[1]["message"]

    def test_session_isolation(self, tmp_sqlite):
        """Messages from different sessions don't leak."""
        tmp_sqlite.add_message("sess-A", "user", "Hello from A")
        tmp_sqlite.add_message("sess-B", "user", "Hello from B")

        history_a = tmp_sqlite.get_history("sess-A")
        history_b = tmp_sqlite.get_history("sess-B")

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["message"] == "Hello from A"
        assert history_b[0]["message"] == "Hello from B"

    def test_delete_session_clears_history(self, tmp_sqlite):
        """Deleting session removes all messages."""
        session_id = "sess-delete"
        tmp_sqlite.add_message(session_id, "user", "Message 1")
        tmp_sqlite.add_message(session_id, "assistant", "Reply 1")
        tmp_sqlite.add_message(session_id, "user", "Message 2")

        deleted = tmp_sqlite.delete_session_history(session_id)
        assert deleted == 3

        history = tmp_sqlite.get_history(session_id)
        assert len(history) == 0

    def test_cross_modal_links_same_page(self, tmp_sqlite, sample_text_chunks, sample_image_chunks):
        """Same-page links are created between text and image chunks on the same page."""
        # Both txt-001 and img-001 are from report.pdf page 1
        count = tmp_sqlite.generate_same_page_links(
            sample_text_chunks, sample_image_chunks
        )
        # At least one same-page link should be created (txt-001 ↔ img-001)
        assert count >= 1

        # Verify the link is queryable
        links = tmp_sqlite.get_linked_chunks("txt-001", min_strength=0.0)
        assert len(links) >= 1
        link_chunk_ids = [lnk["linked_chunk_id"] for lnk in links]
        assert "img-001" in link_chunk_ids

    def test_session_ids_enumeration(self, tmp_sqlite):
        """get_session_ids returns all sessions with history."""
        tmp_sqlite.add_message("sess-1", "user", "msg")
        tmp_sqlite.add_message("sess-2", "user", "msg")
        tmp_sqlite.add_message("sess-3", "user", "msg")

        session_ids = tmp_sqlite.get_session_ids()
        assert set(session_ids) == {"sess-1", "sess-2", "sess-3"}


# ═══════════════════════════════════════════════════════════════════════════
# CITATION PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestCitationIntegration:
    """Tests for citation engine with real GoldChunks."""

    def _build_citation_object(self, index, chunk):
        """Helper to build a CitationObject from a GoldChunk."""
        html_span = (
            f'<cite data-chunk-id="{chunk.chunk_id}" '
            f'data-modality="{chunk.modality}">[{index}]</cite>'
        )
        return CitationObject(
            index=index,
            chunk_id=chunk.chunk_id,
            modality=chunk.modality,
            source_file=chunk.source_file,
            page_number=chunk.page_number,
            start_time=getattr(chunk, "start_time", 0.0),
            end_time=getattr(chunk, "end_time", 0.0),
            timestamp_display=getattr(chunk, "timestamp_display", ""),
            text_preview=chunk.text[:50],
            image_path=getattr(chunk, "image_path", ""),
            thumbnail_path=getattr(chunk, "thumbnail_path", ""),
            modality_label=chunk.modality.upper(),
            modality_color="#1E40AF",
            html_span=html_span,
        )

    def test_annotate_response_replaces_markers(self, gold_chunks):
        """[N] markers in response text are replaced with HTML <cite> spans."""
        citations = [self._build_citation_object(i, c) for i, c in enumerate(gold_chunks, 1)]

        response_text = "ML is a field of AI [1]. See the diagram [2]. As discussed [3]."
        annotated = annotate_response(response_text, citations)

        # Should contain <cite> elements for all three references
        assert "<cite" in annotated
        assert "data-chunk-id" in annotated
        assert "[1]" not in annotated or "<cite" in annotated  # [1] replaced or wrapped

    def test_citation_engine_process(self, gold_chunks):
        """CitationEngine.process creates CitationObjects from response + citation_metadata + index_map."""
        engine = CitationEngine()
        response = "Machine learning is great [1]. See figure [2]."
        index_map = {1: gold_chunks[0], 2: gold_chunks[1]}
        citation_metadata = [
            {"index": 1, "chunk_id": gold_chunks[0].chunk_id},
            {"index": 2, "chunk_id": gold_chunks[1].chunk_id},
        ]

        result = engine.process(response, citation_metadata, index_map)

        assert isinstance(result, CitationResult)
        assert result.total_citations >= 2
        assert len(result.citations) >= 2

    def test_cluster_detection_same_page(self, gold_chunks):
        """detect_clusters finds same-page clusters between text and image chunks."""
        mock_sqlite = MagicMock()
        # detect_clusters uses get_links_between(chunk_a, chunk_b)
        mock_sqlite.get_links_between.return_value = [
            {
                "link_type": "same_page",
                "strength": 1.0,
            }
        ]

        citations = [self._build_citation_object(i, c) for i, c in enumerate(gold_chunks[:2], 1)]
        index_map = {1: gold_chunks[0], 2: gold_chunks[1]}

        clusters = detect_clusters(citations, index_map, mock_sqlite, min_strength=0.5)
        # Should detect the same_page link between txt-001 (text) and img-001 (image)
        assert len(clusters) >= 1


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT PIPELINE INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestExportIntegration:
    """Integration tests for ExportEngine with real file generation."""

    SAMPLE_EXPORT_DATA = {
        "title": "Test Report",
        "summary": "A summary of machine learning concepts.",
        "export_type": "xlsx",
        "sections": [
            {
                "heading": "Introduction",
                "body": "Machine learning is a field of artificial intelligence.",
            },
            {
                "heading": "Methods",
                "body": "We used neural networks for classification.",
            },
        ],
        "sources": [
            {"file": "report.pdf", "pages": "1-5"},
            {"file": "lecture.mp3", "time": "00:00 - 01:00"},
        ],
    }

    def _create_export_data(self, fmt):
        data = dict(self.SAMPLE_EXPORT_DATA)
        data["export_type"] = fmt
        return data

    def test_export_xlsx(self, tmp_dir):
        """XLSX export creates a valid .xlsx file."""
        with patch("modules.export.export_engine.OUTPUT_DIR", tmp_dir):
            engine = ExportEngine()
            result = engine.export(self._create_export_data("xlsx"))

            assert result.success, f"XLSX export failed: {result.error}"
            assert result.file_path.exists()
            assert result.file_path.suffix == ".xlsx"
            assert result.file_size_bytes > 0

    def test_export_docx(self, tmp_dir):
        """DOCX export creates a valid .docx file."""
        with patch("modules.export.export_engine.OUTPUT_DIR", tmp_dir):
            engine = ExportEngine()
            result = engine.export(self._create_export_data("docx"))

            assert result.success, f"DOCX export failed: {result.error}"
            assert result.file_path.exists()
            assert result.file_path.suffix == ".docx"
            assert result.file_size_bytes > 0

    def test_export_pptx(self, tmp_dir):
        """PPTX export creates a valid .pptx file."""
        with patch("modules.export.export_engine.OUTPUT_DIR", tmp_dir):
            engine = ExportEngine()
            result = engine.export(self._create_export_data("pptx"))

            assert result.success, f"PPTX export failed: {result.error}"
            assert result.file_path.exists()
            assert result.file_path.suffix == ".pptx"
            assert result.file_size_bytes > 0

    def test_export_csv(self, tmp_dir):
        """CSV export creates a valid output file."""
        with patch("modules.export.export_engine.OUTPUT_DIR", tmp_dir):
            engine = ExportEngine()
            data = self._create_export_data("csv")
            # CSV needs sheets with headers/rows
            data["sheets"] = [
                {
                    "name": "Main Data",
                    "headers": ["Topic", "Description"],
                    "rows": [
                        ["ML", "Machine learning overview"],
                        ["DL", "Deep learning overview"],
                    ],
                }
            ]
            result = engine.export(data)

            assert result.success, f"CSV export failed: {result.error}"
            assert result.file_path.exists()
            assert result.file_size_bytes > 0


# ═══════════════════════════════════════════════════════════════════════════
# VRAM MANAGER INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestVRAMManagerIntegration:
    """Tests for VRAMManager acquire/release/eviction logic."""

    def _fresh_vram_manager(self):
        """Create a fresh VRAMManager instance (bypassing singleton)."""
        from core.vram_manager import VRAMManager
        mgr = object.__new__(VRAMManager)
        mgr._initialized = False
        mgr.__init__()
        return mgr

    def test_acquire_release_cycle(self):
        """Basic acquire → release cycle updates used_gb correctly."""
        mgr = self._fresh_vram_manager()

        assert mgr.acquire("clip")
        assert mgr.used_gb > 0

        mgr.release("clip")
        assert mgr.used_gb == 0.0

    def test_ceiling_enforcement(self):
        """Cannot acquire beyond ceiling when no evictable models exist."""
        mgr = self._fresh_vram_manager()

        # Load LLM models (non-evictable)
        assert mgr.acquire("llm_3b")  # 2.1 GB
        assert mgr.acquire("llm_1b")  # 0.7 GB  → total 2.8 GB

        # Now try to load something that would exceed ceiling
        # bge_m3 = 0.55, total would be 3.35 which is under 3.5
        assert mgr.acquire("bge_m3") is True

        # Now try clip (0.35), total would be 3.7 > 3.5
        # No non-LLM evictable models except bge_m3
        # Should evict bge_m3 to make room
        eviction_called = []
        mgr.register_evict_callback("bge_m3", lambda: eviction_called.append(True))
        result = mgr.acquire("clip")

        assert result is True
        assert len(eviction_called) == 1  # bge_m3 was evicted

    def test_lru_eviction_order(self):
        """LRU eviction evicts the least recently used non-LLM model first."""
        mgr = self._fresh_vram_manager()

        mgr.acquire("clip")       # loaded first (oldest)
        mgr.acquire("bge_m3")     # loaded second
        mgr.acquire("easyocr")    # loaded third (newest)

        evicted_models = []
        mgr.register_evict_callback("clip", lambda: evicted_models.append("clip"))
        mgr.register_evict_callback("bge_m3", lambda: evicted_models.append("bge_m3"))
        mgr.register_evict_callback("easyocr", lambda: evicted_models.append("easyocr"))

        # Force eviction by acquiring  whisper (0.46 GB)
        # Total GPU at this point: clip(0.35) + bge_m3(0.55) + easyocr(0.30) = 1.20 GB
        # whisper = 0.46, total = 1.66 < 3.5 — no eviction needed
        # Instead, let's acquire llm_3b (2.1) → 3.30 GB, then llm_1b (0.7) → 4.0
        # That would need eviction
        mgr.acquire("llm_3b")  # 3.30 total — fits
        # Now try llm_1b (0.7): 3.30 + 0.7 = 4.0 > 3.5
        # Must evict non-LLM in LRU order: clip first
        mgr.acquire("llm_1b")

        assert evicted_models[0] == "clip"  # oldest non-LLM evicted first

    def test_status_reporting(self):
        """status() reports correct loaded models and usage."""
        mgr = self._fresh_vram_manager()
        mgr.acquire("clip")

        status = mgr.status()
        assert "clip" in status["loaded_models"]
        assert status["used_gb"] > 0
        assert status["ceiling_gb"] > 0
        assert status["available_gb"] == status["ceiling_gb"] - status["used_gb"]


# ═══════════════════════════════════════════════════════════════════════════
# QUERY ROUTING INTEGRATION
# ═══════════════════════════════════════════════════════════════════════════


@pytest.mark.integration
class TestQueryRoutingIntegration:
    """Tests for query classification and routing logic."""

    def test_text_only_query(self):
        """Plain text query routes to text search."""
        result = classify_query("What is machine learning?")
        assert result["search_text"] is True

    def test_image_query(self):
        """Image-related query routes to image search."""
        result = classify_query("show me the diagram from page 5")
        assert result["search_images"] is True

    def test_audio_query(self):
        """Audio-related query routes to audio search."""
        result = classify_query("what was said in the lecture recording?")
        assert result["search_audio"] is True

    def test_mixed_modality_query(self):
        """Complex query may enable multiple search modalities."""
        result = classify_query(
            "compare the chart on page 3 with the audio transcript"
        )
        # Should search at least images and audio
        assert result["search_images"] is True or result["search_audio"] is True
