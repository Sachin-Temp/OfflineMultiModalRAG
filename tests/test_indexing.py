"""
tests/test_indexing.py
Full test suite for MilvusStore and TantivyIndex.

Tests cover:
- TantivyIndex: schema creation, add/commit, BM25F search, modality
  filtering, session filtering, stats, persistence, empty queries,
  incremental add, and multi-field search
- MilvusStore: unit tests for BGE-M3 helpers, dimension validation,
  URI selection logic, and structure tests
  (Skip all Milvus integration tests unless a Milvus backend is reachable)

Run with:
    pytest tests/test_indexing.py -v                    # all tests
    pytest tests/test_indexing.py -v -m "not milvus"    # tantivy only
    pytest tests/test_indexing.py -v -m "not bge_m3"    # skip model tests
"""

import json
import os
import shutil
import sys
import uuid
import pytest
from pathlib import Path
from typing import List
from unittest.mock import patch, MagicMock

from models.schemas import TextChunk, ImageChunk, AudioChunk, Modality


# ═══════════════════════════════════════════════════════════════════════════
# MARKERS
# ═══════════════════════════════════════════════════════════════════════════

# Register custom markers
pytestmark = [
    pytest.mark.filterwarnings("ignore::DeprecationWarning"),
]


# ═══════════════════════════════════════════════════════════════════════════
# TANTIVY TESTS
# ═══════════════════════════════════════════════════════════════════════════


class TestTantivySchema:
    """Test _build_schema() and index creation."""

    def test_schema_builds_successfully(self):
        """Schema builder should create all 5 fields without error."""
        from modules.indexing.tantivy_index import _build_schema
        schema = _build_schema()
        assert schema is not None

    def test_index_creates_in_temp_dir(self, tmp_path):
        """TantivyIndex should create an index in the given directory."""
        from modules.indexing.tantivy_index import TantivyIndex
        idx = TantivyIndex(index_path=str(tmp_path / "test_index"))
        idx.initialize()
        assert idx._index is not None
        # Index directory should contain some files
        idx_dir = tmp_path / "test_index"
        assert idx_dir.exists()
        assert len(list(idx_dir.iterdir())) > 0

    def test_initialize_idempotent(self, tmp_path):
        """Calling initialize() twice should not error — reuse existing index."""
        from modules.indexing.tantivy_index import TantivyIndex
        idx_path = str(tmp_path / "test_index")
        idx = TantivyIndex(index_path=idx_path)
        idx.initialize()
        # Second call should reuse
        idx2 = TantivyIndex(index_path=idx_path)
        idx2.initialize()
        assert idx2._index is not None


class TestTantivyAddChunks:
    """Test add_text_chunks, add_image_chunks, add_audio_chunks."""

    @pytest.fixture
    def tantivy_index(self, tmp_path):
        from modules.indexing.tantivy_index import TantivyIndex
        idx = TantivyIndex(index_path=str(tmp_path / "test_index"))
        idx.initialize()
        return idx

    @pytest.fixture
    def sample_text_chunks(self) -> List[TextChunk]:
        return [
            TextChunk(
                text="The quarterly revenue exceeded expectations with strong growth.",
                source_file="report.pdf",
                page_number=1,
                session_id="session-1",
            ),
            TextChunk(
                text="Machine learning models improve accuracy over time through training.",
                source_file="ml_paper.pdf",
                page_number=3,
                session_id="session-1",
            ),
            TextChunk(
                text="Budget allocation for the fiscal year was approved by the board.",
                source_file="minutes.docx",
                page_number=1,
                session_id="session-2",
            ),
        ]

    @pytest.fixture
    def sample_image_chunks(self) -> List[ImageChunk]:
        return [
            ImageChunk(
                image_path="/images/chart.png",
                source_file="report.pdf",
                page_number=2,
                ocr_text="Revenue chart showing quarterly growth",
                session_id="session-1",
            ),
            ImageChunk(
                image_path="/images/diagram.png",
                source_file="ml_paper.pdf",
                page_number=5,
                ocr_text=None,  # No OCR text
                llm_description="Architecture diagram of neural network",
                session_id="session-1",
            ),
        ]

    @pytest.fixture
    def sample_audio_chunks(self) -> List[AudioChunk]:
        return [
            AudioChunk(
                text="The speaker discussed revenue growth in the third quarter.",
                audio_file="meeting.wav",
                session_id="session-1",
                start_time=0.0,
                end_time=30.0,
            ),
            AudioChunk(
                text="Budget concerns were raised during the annual review.",
                audio_file="review.wav",
                session_id="session-2",
                start_time=0.0,
                end_time=45.0,
            ),
        ]

    def test_add_text_chunks(self, tantivy_index, sample_text_chunks):
        """Text chunks should be added and appear in stats."""
        tantivy_index.add_text_chunks(sample_text_chunks)
        tantivy_index.reload()
        stats = tantivy_index.get_stats()
        assert stats["num_docs"] == 3

    def test_add_image_chunks(self, tantivy_index, sample_image_chunks):
        """Image chunks should be indexed using OCR text and llm_description."""
        tantivy_index.add_image_chunks(sample_image_chunks)
        tantivy_index.reload()
        stats = tantivy_index.get_stats()
        assert stats["num_docs"] == 2  # Both indexed, even if no OCR text

    def test_add_audio_chunks(self, tantivy_index, sample_audio_chunks):
        """Audio transcript chunks should be indexed."""
        tantivy_index.add_audio_chunks(sample_audio_chunks)
        tantivy_index.reload()
        stats = tantivy_index.get_stats()
        assert stats["num_docs"] == 2

    def test_add_empty_text_chunks_skipped(self, tantivy_index):
        """Chunks with only whitespace text should be skipped."""
        chunks = [
            TextChunk(text="   ", source_file="empty.pdf", page_number=1),
            TextChunk(text="", source_file="empty2.pdf", page_number=1),
        ]
        tantivy_index.add_text_chunks(chunks)
        tantivy_index.reload()
        stats = tantivy_index.get_stats()
        assert stats["num_docs"] == 0

    def test_incremental_add(self, tantivy_index, sample_text_chunks):
        """Adding more chunks after initial add should increment count."""
        tantivy_index.add_text_chunks(sample_text_chunks[:1])
        tantivy_index.reload()
        assert tantivy_index.get_stats()["num_docs"] == 1

        tantivy_index.add_text_chunks(sample_text_chunks[1:])
        tantivy_index.reload()
        assert tantivy_index.get_stats()["num_docs"] == 3


class TestTantivySearch:
    """Test BM25F search including filtering."""

    @pytest.fixture
    def populated_index(self, tmp_path):
        """Create an index with a mix of text, image, and audio chunks."""
        from modules.indexing.tantivy_index import TantivyIndex
        idx = TantivyIndex(index_path=str(tmp_path / "search_index"))
        idx.initialize()

        # Add text chunks
        idx.add_text_chunks([
            TextChunk(
                text="The quarterly revenue exceeded expectations with strong growth.",
                source_file="report.pdf",
                page_number=1,
                session_id="session-1",
            ),
            TextChunk(
                text="Machine learning models improve accuracy over time.",
                source_file="ml_paper.pdf",
                page_number=3,
                session_id="session-1",
            ),
            TextChunk(
                text="Budget allocation for fiscal year was approved.",
                source_file="minutes.docx",
                page_number=1,
                session_id="session-2",
            ),
        ])

        # Add image chunks
        idx.add_image_chunks([
            ImageChunk(
                image_path="/images/chart.png",
                source_file="report.pdf",
                page_number=2,
                ocr_text="Revenue chart showing quarterly growth",
                session_id="session-1",
            ),
        ])

        # Add audio chunks
        idx.add_audio_chunks([
            AudioChunk(
                text="The speaker discussed revenue growth in the third quarter.",
                audio_file="meeting.wav",
                session_id="session-1",
                start_time=0.0,
                end_time=30.0,
            ),
        ])

        idx.reload()
        return idx

    def test_basic_search(self, populated_index):
        """Search for 'revenue' should find text, image, and audio chunks."""
        results = populated_index.search("revenue", top_k=10)
        assert len(results) >= 2
        # All results should have chunk_id and score
        for r in results:
            assert "chunk_id" in r
            assert "score" in r
            assert r["score"] > 0

    def test_search_returns_sorted_by_score(self, populated_index):
        """Results should be sorted by BM25F score descending."""
        results = populated_index.search("revenue growth", top_k=10)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_search_with_modality_filter(self, populated_index):
        """Modality filter should restrict results to that modality."""
        results = populated_index.search("revenue", top_k=10, modality="text")
        for r in results:
            assert r["modality"] == "text"

    def test_search_with_session_filter(self, populated_index):
        """Session filter should only return docs from that session."""
        results = populated_index.search("revenue", top_k=10, session_id="session-1")
        for r in results:
            assert r["session_id"] == "session-1"

    def test_search_no_results_for_unknown_term(self, populated_index):
        """Searching for a nonexistent word should return empty list."""
        results = populated_index.search("xyznonexistent123")
        assert results == []

    def test_search_empty_query(self, populated_index):
        """Empty query should return empty list."""
        results = populated_index.search("")
        assert results == []

    def test_search_whitespace_query(self, populated_index):
        """Whitespace-only query should return empty list."""
        results = populated_index.search("   ")
        assert results == []

    def test_top_k_limits_results(self, populated_index):
        """Top_k should cap the number of results."""
        results = populated_index.search("revenue", top_k=1)
        assert len(results) <= 1

    def test_search_image_ocr_text(self, populated_index):
        """Searching should find images via their OCR text."""
        results = populated_index.search("chart", top_k=10)
        image_results = [r for r in results if r["modality"] == "image"]
        assert len(image_results) >= 1

    def test_search_audio_transcript(self, populated_index):
        """Searching should find audio chunks via transcript text."""
        results = populated_index.search("speaker", top_k=10)
        audio_results = [r for r in results if r["modality"] == "audio"]
        assert len(audio_results) >= 1

    def test_search_finds_stemmed_words(self, populated_index):
        """English stemmer should match 'expected' to 'expectations'."""
        results = populated_index.search("expected")
        # Should find doc containing "expectations"
        assert len(results) >= 1

    def test_search_source_file(self, populated_index):
        """Should be able to find documents by source filename."""
        results = populated_index.search("report", top_k=10)
        source_files = [r["source_file"] for r in results]
        assert any("report" in sf for sf in source_files)


class TestTantivyPersistence:
    """Test that index persists across reopens."""

    def test_index_persists_after_reopen(self, tmp_path):
        """Data added and committed should survive index reopen."""
        from modules.indexing.tantivy_index import TantivyIndex
        idx_path = str(tmp_path / "persist_test")

        # Write
        idx = TantivyIndex(index_path=idx_path)
        idx.initialize()
        idx.add_text_chunks([
            TextChunk(
                text="Persistent data should survive restarts.",
                source_file="test.pdf",
                page_number=1,
                session_id="s1",
            ),
        ])

        # Reopen
        idx2 = TantivyIndex(index_path=idx_path)
        idx2.initialize()
        idx2.reload()

        results = idx2.search("persistent")
        assert len(results) >= 1
        assert "persistent" in results[0]["text"].lower()


class TestTantivyStats:
    """Test get_stats()."""

    def test_stats_empty_index(self, tmp_path):
        """Empty index should report 0 docs and ok status."""
        from modules.indexing.tantivy_index import TantivyIndex
        idx = TantivyIndex(index_path=str(tmp_path / "stats_test"))
        idx.initialize()
        stats = idx.get_stats()
        assert stats["num_docs"] == 0
        assert stats["status"] == "ok"

    def test_stats_tracks_doc_count(self, tmp_path):
        """Stats should reflect the number of added documents."""
        from modules.indexing.tantivy_index import TantivyIndex
        idx = TantivyIndex(index_path=str(tmp_path / "stats_count"))
        idx.initialize()

        idx.add_text_chunks([
            TextChunk(
                text="Document one.",
                source_file="one.pdf",
                page_number=1,
            ),
            TextChunk(
                text="Document two.",
                source_file="two.pdf",
                page_number=1,
            ),
        ])
        idx.reload()

        stats = idx.get_stats()
        assert stats["num_docs"] == 2


# ═══════════════════════════════════════════════════════════════════════════
# MILVUS STORE UNIT TESTS (no Milvus connection needed)
# ═══════════════════════════════════════════════════════════════════════════


class TestMilvusURISelection:
    """Test _get_milvus_uri() logic for platform detection."""

    def test_env_var_override_wins(self):
        """MILVUS_URI env var should override platform detection."""
        from modules.indexing.milvus_store import _get_milvus_uri
        with patch.dict(os.environ, {"MILVUS_URI": "http://custom:19530"}):
            uri = _get_milvus_uri()
            assert uri == "http://custom:19530"

    def test_windows_defaults_to_localhost(self):
        """On Windows, default URI should be localhost:19530."""
        from modules.indexing.milvus_store import _get_milvus_uri
        with patch.dict(os.environ, {}, clear=True):
            # Remove MILVUS_URI if present
            os.environ.pop("MILVUS_URI", None)
            os.environ.pop("MILVUS_HOST", None)
            if sys.platform == "win32":
                uri = _get_milvus_uri()
                assert "localhost" in uri or "19530" in uri


class TestMilvusDimensionValidation:
    """Test that MilvusStore rejects wrong-dim embeddings in search methods."""

    @pytest.fixture
    def store(self):
        """Create a MilvusStore with mocked client to test search validation."""
        from modules.indexing.milvus_store import MilvusStore
        store = MilvusStore()
        mock_client = MagicMock()
        store._client = mock_client
        store._initialized = True
        return store

    def test_search_text_wrong_dim_returns_empty(self, store):
        """search_text should reject non-1024 dim vectors."""
        results = store.search_text(
            query_embedding=[0.1] * 512,  # CLIP dim, not BGE-M3
            top_k=10,
        )
        assert results == []

    def test_search_images_wrong_dim_returns_empty(self, store):
        """search_images should reject non-512 dim vectors."""
        results = store.search_images(
            query_embedding=[0.1] * 1024,  # BGE-M3 dim, not CLIP
            top_k=10,
        )
        assert results == []

    def test_search_audio_wrong_dim_returns_empty(self, store):
        """search_audio should reject non-1024 dim vectors."""
        results = store.search_audio(
            query_embedding=[0.1] * 256,  # Wrong dim
            top_k=10,
        )
        assert results == []

    def test_search_text_correct_dim_calls_client(self, store):
        """search_text with 1024-dim vector should call the Milvus client."""
        store._client.search.return_value = [[]]
        results = store.search_text(
            query_embedding=[0.1] * 1024,
            top_k=10,
        )
        store._client.search.assert_called_once()

    def test_search_images_correct_dim_calls_client(self, store):
        """search_images with 512-dim vector should call the Milvus client."""
        store._client.search.return_value = [[]]
        results = store.search_images(
            query_embedding=[0.1] * 512,
            top_k=10,
        )
        store._client.search.assert_called_once()


class TestMilvusInsertValidation:
    """Test insert methods validate and skip bad chunks."""

    @pytest.fixture
    def store(self):
        """MilvusStore with mocked client."""
        from modules.indexing.milvus_store import MilvusStore
        store = MilvusStore()
        mock_client = MagicMock()
        store._client = mock_client
        store._initialized = True
        return store

    def test_insert_empty_list_returns_zero(self, store):
        """Insert with empty list should return 0 without calling client."""
        result = store.insert_text_chunks([])
        assert result == 0

    def test_insert_image_skips_none_embedding(self, store):
        """Image chunks with None embedding should be skipped."""
        chunks = [
            ImageChunk(
                image_path="/test.png",
                source_file="test.pdf",
                page_number=1,
                embedding=None,
            ),
        ]
        result = store.insert_image_chunks(chunks)
        assert result == 0

    def test_insert_image_skips_wrong_dim(self, store):
        """Image chunks with wrong dimension should be skipped."""
        chunks = [
            ImageChunk(
                image_path="/test.png",
                source_file="test.pdf",
                page_number=1,
                embedding=[0.1] * 1024,  # Wrong! Should be 512
            ),
        ]
        result = store.insert_image_chunks(chunks)
        assert result == 0

    def test_insert_image_correct_dim_inserts(self, store):
        """Image chunks with correct 512-dim embedding should be inserted."""
        chunks = [
            ImageChunk(
                image_path="/test.png",
                source_file="test.pdf",
                page_number=1,
                embedding=[0.1] * 512,
                session_id="test-session",
            ),
        ]
        result = store.insert_image_chunks(chunks)
        assert result == 1
        store._client.insert.assert_called_once()


class TestMilvusCollectionCreation:
    """Test collection creation logic with mocked client."""

    @pytest.fixture
    def store(self):
        from modules.indexing.milvus_store import MilvusStore
        store = MilvusStore()
        mock_client = MagicMock()
        mock_client.list_collections.return_value = []
        store._client = mock_client
        return store

    def test_initialize_creates_three_collections(self, store):
        """initialize() should create text, image, and audio collections."""
        store.initialize()
        assert store._client.create_collection.call_count == 3

    def test_initialize_skips_existing_collections(self, store):
        """initialize() should not recreate existing collections."""
        from config.settings import COLLECTION_TEXT, COLLECTION_IMAGE, COLLECTION_AUDIO
        store._client.list_collections.return_value = [
            COLLECTION_TEXT, COLLECTION_IMAGE, COLLECTION_AUDIO
        ]
        store.initialize()
        assert store._client.create_collection.call_count == 0


class TestMilvusDeleteMethods:
    """Test delete_by_session and delete_by_chunk_id."""

    @pytest.fixture
    def store(self):
        from modules.indexing.milvus_store import MilvusStore
        store = MilvusStore()
        mock_client = MagicMock()
        mock_client.delete.return_value = {"delete_count": 5}
        store._client = mock_client
        store._initialized = True
        return store

    def test_delete_by_session_calls_all_collections(self, store):
        """delete_by_session should call delete on all 3 collections."""
        counts = store.delete_by_session("test-session")
        assert store._client.delete.call_count == 3
        # Check all keys present
        assert "text" in counts
        assert "image" in counts
        assert "audio" in counts

    def test_delete_by_chunk_id_returns_true(self, store):
        """delete_by_chunk_id should return True on success."""
        result = store.delete_by_chunk_id("text_chunks", "fake-uuid")
        assert result is True

    def test_delete_by_chunk_id_returns_false_on_error(self, store):
        """delete_by_chunk_id should return False on error."""
        store._client.delete.side_effect = Exception("Milvus error")
        result = store.delete_by_chunk_id("text_chunks", "fake-uuid")
        assert result is False


class TestMilvusStats:
    """Test get_collection_stats."""

    def test_stats_returns_all_collections(self):
        """get_collection_stats should return entries for all 3 collections."""
        from modules.indexing.milvus_store import MilvusStore
        store = MilvusStore()
        mock_client = MagicMock()
        mock_client.get_collection_stats.return_value = {"row_count": 42}
        store._client = mock_client
        store._initialized = True

        stats = store.get_collection_stats()
        assert len(stats) == 3
        for name, info in stats.items():
            assert info["status"] == "ok"
            assert info["row_count"] == 42


class TestMilvusBGEM3Helpers:
    """Test BGE-M3 embedding helper functions (mocked, no GPU needed)."""

    def test_compute_bge_m3_batch_empty_returns_empty(self):
        """_compute_bge_m3_batch([]) should return []."""
        from modules.indexing.milvus_store import _compute_bge_m3_batch
        result = _compute_bge_m3_batch([])
        assert result == []


class TestMilvusClose:
    """Test close() method."""

    def test_close_resets_state(self):
        """close() should nullify client and reset initialized flag."""
        from modules.indexing.milvus_store import MilvusStore
        store = MilvusStore()
        mock_client = MagicMock()
        store._client = mock_client
        store._initialized = True

        store.close()
        assert store._client is None
        assert store._initialized is False
