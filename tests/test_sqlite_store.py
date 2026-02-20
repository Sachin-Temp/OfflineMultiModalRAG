"""
tests/test_sqlite_store.py
Full test suite for the SQLiteStore module.

Tests cover:
- Database initialization (table creation, idempotency)
- Chat history (add, retrieve, delete, session listing)
- SAME_PAGE link generation
- TEMPORAL_PROXIMITY link generation
- SEMANTIC link generation (with mock CLIP embeddings)
- CO_RETRIEVED link update and strength computation
- Link queries (get_linked_chunks, get_links_between)
- Canonical ordering of link pairs
- File registration and deduplication
- File status updates
- Statistics endpoint
- Thread safety (concurrent writes)
- Edge cases (empty inputs, duplicate links, strength updates)

Run with:
    pytest tests/test_sqlite_store.py -v
    pytest tests/test_sqlite_store.py -v -m "not slow"
"""

import uuid
import time
import threading
import pytest
import numpy as np
from pathlib import Path
from typing import List


# ── Fixtures ───────────────────────────────────────────────────────────────
@pytest.fixture
def db(tmp_path):
    """Create a fresh SQLiteStore with a temp database for each test."""
    from modules.indexing.sqlite_store import SQLiteStore
    store = SQLiteStore(db_path=tmp_path / "test_rag.db")
    store.initialize()
    return store


def _make_text_chunk(
    page: int = 1,
    source: str = "report.pdf",
    session: str = "sess-001",
    text: str = "Sample text content for testing purposes.",
):
    from models.schemas import TextChunk
    return TextChunk(
        text=text,
        source_file=source,
        page_number=page,
        session_id=session,
    )


def _make_image_chunk(
    page: int = 1,
    source: str = "report.pdf",
    session: str = "sess-001",
    embedding: List[float] = None,
):
    from models.schemas import ImageChunk
    chunk = ImageChunk(
        image_path="data/media/test.png",
        source_file=source,
        page_number=page,
        session_id=session,
        ocr_text="Chart showing revenue data",
    )
    if embedding is not None:
        chunk.embedding = embedding
    else:
        # Random normalized 512-dim CLIP embedding
        v = np.random.randn(512).astype(np.float32)
        chunk.embedding = (v / np.linalg.norm(v)).tolist()
    return chunk


def _make_audio_chunk(
    session: str = "sess-001",
    text: str = "Budget approved in Q3 meeting discussion.",
):
    from models.schemas import AudioChunk
    return AudioChunk(
        text=text,
        audio_file="meeting.mp3",
        start_time=0.0,
        end_time=30.0,
        session_id=session,
        timestamp_display="00:00 - 00:30",
    )


# ── Initialization Tests ───────────────────────────────────────────────────
class TestInitialization:

    def test_tables_created(self, db, tmp_path):
        import sqlite3
        conn = sqlite3.connect(str(tmp_path / "test_rag.db"))
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        conn.close()
        assert "chat_history"      in tables
        assert "cross_modal_links" in tables
        assert "ingested_files"    in tables

    def test_idempotent_initialization(self, db):
        """Calling initialize() twice must not raise or duplicate tables."""
        db.initialize()
        db.initialize()
        stats = db.get_stats()
        assert stats["status"] == "ok"

    def test_starts_empty(self, db):
        stats = db.get_stats()
        assert stats["chat_messages"]  == 0
        assert stats["total_links"]    == 0
        assert stats["ingested_files"] == 0


# ── Chat History Tests ─────────────────────────────────────────────────────
class TestChatHistory:

    def test_add_user_message(self, db):
        row_id = db.add_message("sess-001", "user", "Hello, show me Q3 data")
        assert row_id is not None
        assert row_id > 0

    def test_add_assistant_message_with_citations(self, db):
        cited = ["chunk-001", "chunk-002"]
        row_id = db.add_message(
            "sess-001", "assistant",
            "Q3 revenue was $4.2M [1][2]",
            cited_chunks=cited,
        )
        assert row_id > 0

    def test_invalid_role_raises(self, db):
        with pytest.raises(ValueError):
            db.add_message("sess-001", "system", "invalid role")

    def test_get_history_returns_messages(self, db):
        db.add_message("sess-001", "user", "First question")
        db.add_message("sess-001", "assistant", "First answer")
        db.add_message("sess-001", "user", "Second question")

        history = db.get_history("sess-001")
        assert len(history) == 3

    def test_get_history_chronological_order(self, db):
        db.add_message("sess-001", "user", "Message 1")
        time.sleep(0.01)
        db.add_message("sess-001", "user", "Message 2")
        time.sleep(0.01)
        db.add_message("sess-001", "user", "Message 3")

        history = db.get_history("sess-001")
        messages = [h["message"] for h in history]
        assert messages.index("Message 1") < messages.index("Message 2")
        assert messages.index("Message 2") < messages.index("Message 3")

    def test_get_history_last_n_limit(self, db):
        for i in range(10):
            db.add_message("sess-001", "user", f"Message {i}")

        history = db.get_history("sess-001", last_n=5)
        assert len(history) == 5

    def test_get_history_session_isolation(self, db):
        db.add_message("sess-A", "user", "Session A message")
        db.add_message("sess-B", "user", "Session B message")

        history_a = db.get_history("sess-A")
        history_b = db.get_history("sess-B")

        assert len(history_a) == 1
        assert len(history_b) == 1
        assert history_a[0]["message"] == "Session A message"
        assert history_b[0]["message"] == "Session B message"

    def test_cited_chunks_round_trip(self, db):
        cited = ["abc-123", "def-456", "ghi-789"]
        db.add_message("sess-001", "assistant", "Answer", cited_chunks=cited)
        history = db.get_history("sess-001")
        assert history[0]["cited_chunks"] == cited

    def test_get_history_empty_session(self, db):
        history = db.get_history("nonexistent-session")
        assert history == []

    def test_delete_session_history(self, db):
        db.add_message("sess-001", "user", "Q1")
        db.add_message("sess-001", "assistant", "A1")
        db.add_message("sess-002", "user", "Q2")

        deleted = db.delete_session_history("sess-001")
        assert deleted == 2

        assert db.get_history("sess-001") == []
        assert len(db.get_history("sess-002")) == 1

    def test_get_session_ids(self, db):
        db.add_message("sess-A", "user", "Hello")
        db.add_message("sess-B", "user", "Hello")
        db.add_message("sess-C", "user", "Hello")

        session_ids = db.get_session_ids()
        assert "sess-A" in session_ids
        assert "sess-B" in session_ids
        assert "sess-C" in session_ids


# ── SAME_PAGE Link Tests ───────────────────────────────────────────────────
class TestSamePageLinks:

    def test_links_created_for_matching_pages(self, db):
        text_chunks  = [_make_text_chunk(page=1), _make_text_chunk(page=2)]
        image_chunks = [_make_image_chunk(page=1), _make_image_chunk(page=3)]

        count = db.generate_same_page_links(text_chunks, image_chunks)
        # Only page 1 matches — 1 text × 1 image = 1 link
        assert count == 1

    def test_link_strength_is_one(self, db):
        text_chunk  = _make_text_chunk(page=1)
        image_chunk = _make_image_chunk(page=1)

        db.generate_same_page_links([text_chunk], [image_chunk])
        links = db.get_links_between(text_chunk.chunk_id, image_chunk.chunk_id)

        assert len(links) == 1
        assert links[0]["link_type"] == "same_page"
        assert abs(links[0]["strength"] - 1.0) < 0.001

    def test_multiple_images_per_page(self, db):
        text_chunk = _make_text_chunk(page=5)
        images = [_make_image_chunk(page=5) for _ in range(3)]

        count = db.generate_same_page_links([text_chunk], images)
        assert count == 3

    def test_no_links_different_pages(self, db):
        text_chunks  = [_make_text_chunk(page=1)]
        image_chunks = [_make_image_chunk(page=2)]

        count = db.generate_same_page_links(text_chunks, image_chunks)
        assert count == 0

    def test_no_links_different_source_files(self, db):
        text_chunk  = _make_text_chunk(page=1, source="doc_a.pdf")
        image_chunk = _make_image_chunk(page=1, source="doc_b.pdf")

        count = db.generate_same_page_links([text_chunk], [image_chunk])
        assert count == 0

    def test_empty_inputs_return_zero(self, db):
        assert db.generate_same_page_links([], []) == 0
        assert db.generate_same_page_links([_make_text_chunk()], []) == 0
        assert db.generate_same_page_links([], [_make_image_chunk()]) == 0

    def test_duplicate_links_not_created(self, db):
        text_chunk  = _make_text_chunk(page=1)
        image_chunk = _make_image_chunk(page=1)

        db.generate_same_page_links([text_chunk], [image_chunk])
        db.generate_same_page_links([text_chunk], [image_chunk])

        links = db.get_links_between(text_chunk.chunk_id, image_chunk.chunk_id)
        # Should still be only 1 link (upsert, not duplicate)
        same_page_links = [l for l in links if l["link_type"] == "same_page"]
        assert len(same_page_links) == 1


# ── TEMPORAL_PROXIMITY Link Tests ──────────────────────────────────────────
class TestTemporalLinks:

    def test_links_created_for_same_session(self, db):
        audio_chunks = [_make_audio_chunk(session="sess-001")]
        text_chunks  = [_make_text_chunk(session="sess-001")]

        count = db.generate_temporal_links(
            audio_chunks, text_chunks, "sess-001"
        )
        assert count == 1

    def test_link_strength_is_0_7(self, db):
        audio = _make_audio_chunk(session="sess-001")
        text  = _make_text_chunk(session="sess-001")

        db.generate_temporal_links([audio], [text], "sess-001")
        links = db.get_links_between(audio.chunk_id, text.chunk_id)

        assert len(links) == 1
        assert links[0]["link_type"] == "temporal_proximity"
        assert abs(links[0]["strength"] - 0.7) < 0.001

    def test_no_links_different_sessions(self, db):
        audio = _make_audio_chunk(session="sess-A")
        text  = _make_text_chunk(session="sess-B")

        count = db.generate_temporal_links([audio], [text], "sess-A")
        assert count == 0

    def test_multiple_audio_text_pairs(self, db):
        audio_chunks = [_make_audio_chunk() for _ in range(2)]
        text_chunks  = [_make_text_chunk() for _ in range(3)]

        count = db.generate_temporal_links(
            audio_chunks, text_chunks, "sess-001"
        )
        assert count == 6   # 2 audio × 3 text


# ── SEMANTIC Link Tests ────────────────────────────────────────────────────
class TestSemanticLinks:

    def test_semantic_link_created_high_similarity(self, db):
        """
        If text and image embeddings are nearly identical (cosine sim ≈ 1.0),
        a semantic link should be created (1.0 > threshold of 0.75).
        """
        # Use same vector for text and image → perfect similarity
        shared_vec = np.random.randn(512).astype(np.float32)
        shared_vec /= np.linalg.norm(shared_vec)
        shared_list = shared_vec.tolist()

        text_chunk  = _make_text_chunk()
        image_chunk = _make_image_chunk(embedding=shared_list)

        # Mock the CLIP text encoder to return the same vector
        import modules.indexing.sqlite_store as store_module
        original_fn = store_module._encode_texts_with_clip

        def mock_encode(texts):
            return [shared_list for _ in texts]

        store_module._encode_texts_with_clip = mock_encode
        try:
            count = db.generate_semantic_links([text_chunk], [image_chunk])
            assert count == 1

            links = db.get_links_between(text_chunk.chunk_id, image_chunk.chunk_id)
            sem_links = [l for l in links if l["link_type"] == "semantic"]
            assert len(sem_links) == 1
            assert sem_links[0]["strength"] >= 0.75
        finally:
            store_module._encode_texts_with_clip = original_fn

    def test_semantic_link_not_created_low_similarity(self, db):
        """
        Orthogonal vectors have cosine similarity = 0, so no link should be created.
        """
        # Create two orthogonal 512-dim vectors
        vec_a = np.zeros(512, dtype=np.float32)
        vec_a[0] = 1.0
        vec_b = np.zeros(512, dtype=np.float32)
        vec_b[1] = 1.0   # orthogonal to vec_a

        text_chunk  = _make_text_chunk()
        image_chunk = _make_image_chunk(embedding=vec_b.tolist())

        import modules.indexing.sqlite_store as store_module
        original_fn = store_module._encode_texts_with_clip

        def mock_encode(texts):
            return [vec_a.tolist() for _ in texts]

        store_module._encode_texts_with_clip = mock_encode
        try:
            count = db.generate_semantic_links([text_chunk], [image_chunk])
            assert count == 0
        finally:
            store_module._encode_texts_with_clip = original_fn

    def test_semantic_links_skip_missing_clip_embedding(self, db):
        from models.schemas import ImageChunk
        text_chunk = _make_text_chunk()
        image_chunk = ImageChunk(
            image_path="data/media/img.png",
            source_file="test.pdf",
            page_number=1,
            session_id="sess-001",
        )
        image_chunk.embedding = None   # no CLIP embedding

        import modules.indexing.sqlite_store as store_module
        original_fn = store_module._encode_texts_with_clip
        store_module._encode_texts_with_clip = lambda t: [
            np.random.randn(512).tolist() for _ in t
        ]
        try:
            count = db.generate_semantic_links([text_chunk], [image_chunk])
            assert count == 0   # skipped because no image embedding
        finally:
            store_module._encode_texts_with_clip = original_fn


# ── CO_RETRIEVED Link Tests ────────────────────────────────────────────────
class TestCoRetrievedLinks:

    def test_co_retrieved_link_created(self, db):
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        modalities = {id_a: "text", id_b: "image"}

        count = db.update_co_retrieved_links([id_a, id_b], modalities)
        assert count == 1

        links = db.get_links_between(id_a, id_b)
        co_links = [l for l in links if l["link_type"] == "co_retrieved"]
        assert len(co_links) == 1

    def test_co_retrieved_initial_strength_is_one(self, db):
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        db.update_co_retrieved_links([id_a, id_b], {id_a: "text", id_b: "audio"})

        links = db.get_links_between(id_a, id_b)
        co_link = next(l for l in links if l["link_type"] == "co_retrieved")
        assert abs(co_link["strength"] - 1.0) < 0.001

    def test_co_retrieved_strength_stays_stable_on_repeat(self, db):
        """
        After 1 co-retrieval: strength = 1/1 = 1.0
        After 2nd co-retrieval: strength = 2/2 = 1.0 (still 1.0)
        """
        id_a = str(uuid.uuid4())
        id_b = str(uuid.uuid4())
        id_c = str(uuid.uuid4())
        mods = {id_a: "text", id_b: "image", id_c: "audio"}

        # First retrieval: A + B together → strength = 1/1 = 1.0
        db.update_co_retrieved_links([id_a, id_b], mods)
        links1 = db.get_links_between(id_a, id_b)
        co1 = next(l for l in links1 if l["link_type"] == "co_retrieved")
        assert abs(co1["strength"] - 1.0) < 0.001

        # Second retrieval: A + B together again → co=2, tot=2 → strength=1.0
        db.update_co_retrieved_links([id_a, id_b], mods)
        links2 = db.get_links_between(id_a, id_b)
        co2 = next(l for l in links2 if l["link_type"] == "co_retrieved")
        assert abs(co2["strength"] - 1.0) < 0.001

    def test_co_retrieved_three_chunks(self, db):
        """Three chunks together create 3 pairs."""
        ids = [str(uuid.uuid4()) for _ in range(3)]
        mods = {i: "text" for i in ids}
        count = db.update_co_retrieved_links(ids, mods)
        assert count == 3   # C(3,2) = 3 pairs

    def test_co_retrieved_single_chunk_no_pairs(self, db):
        id_a = str(uuid.uuid4())
        count = db.update_co_retrieved_links([id_a], {id_a: "text"})
        assert count == 0

    def test_co_retrieved_empty_list(self, db):
        count = db.update_co_retrieved_links([], {})
        assert count == 0


# ── Link Query Tests ───────────────────────────────────────────────────────
class TestLinkQueries:

    def test_get_linked_chunks_returns_linked(self, db):
        text_chunk  = _make_text_chunk(page=1)
        image_chunk = _make_image_chunk(page=1)

        db.generate_same_page_links([text_chunk], [image_chunk])

        linked = db.get_linked_chunks(
            text_chunk.chunk_id,
            min_strength=0.5,
        )
        assert len(linked) >= 1
        linked_ids = [l["linked_chunk_id"] for l in linked]
        assert image_chunk.chunk_id in linked_ids

    def test_get_linked_chunks_min_strength_filter(self, db):
        """Links below min_strength should be excluded."""
        text_chunk  = _make_text_chunk(page=1)
        image_chunk = _make_image_chunk(page=1)

        db.generate_same_page_links([text_chunk], [image_chunk])
        # same_page strength = 1.0

        # High threshold — should still return the link
        linked_high = db.get_linked_chunks(
            text_chunk.chunk_id, min_strength=0.99
        )
        assert len(linked_high) >= 1

        # Very high threshold — should exclude
        linked_very_high = db.get_linked_chunks(
            text_chunk.chunk_id, min_strength=1.01
        )
        assert len(linked_very_high) == 0

    def test_get_linked_chunks_link_type_filter(self, db):
        text  = _make_text_chunk(page=1)
        image = _make_image_chunk(page=1)
        db.generate_same_page_links([text], [image])

        same_page_only = db.get_linked_chunks(
            text.chunk_id,
            link_types=["same_page"],
        )
        for l in same_page_only:
            assert l["link_type"] == "same_page"

    def test_get_linked_chunks_bidirectional(self, db):
        """Links should be findable from either end."""
        text  = _make_text_chunk(page=1)
        image = _make_image_chunk(page=1)
        db.generate_same_page_links([text], [image])

        from_text  = db.get_linked_chunks(text.chunk_id)
        from_image = db.get_linked_chunks(image.chunk_id)

        text_linked_ids  = [l["linked_chunk_id"] for l in from_text]
        image_linked_ids = [l["linked_chunk_id"] for l in from_image]

        assert image.chunk_id in text_linked_ids
        assert text.chunk_id  in image_linked_ids

    def test_get_linked_chunks_no_links(self, db):
        orphan_id = str(uuid.uuid4())
        linked = db.get_linked_chunks(orphan_id)
        assert linked == []

    def test_get_links_between_no_link(self, db):
        a = str(uuid.uuid4())
        b = str(uuid.uuid4())
        links = db.get_links_between(a, b)
        assert links == []

    def test_get_links_between_multiple_types(self, db):
        """Two chunks can have multiple link types."""
        text  = _make_text_chunk(page=1)
        image = _make_image_chunk(page=1)

        # Create same_page link
        db.generate_same_page_links([text], [image])

        # Also create co_retrieved link
        db.update_co_retrieved_links(
            [text.chunk_id, image.chunk_id],
            {text.chunk_id: "text", image.chunk_id: "image"},
        )

        links = db.get_links_between(text.chunk_id, image.chunk_id)
        link_types = {l["link_type"] for l in links}
        assert "same_page"    in link_types
        assert "co_retrieved" in link_types


# ── Canonical Ordering Tests ───────────────────────────────────────────────
class TestCanonicalOrdering:

    def test_same_link_regardless_of_order(self, db):
        """
        Creating a link A→B and then B→A should result in only one row.
        """
        id_a = "aaa-" + str(uuid.uuid4())
        id_b = "zzz-" + str(uuid.uuid4())
        mods = {id_a: "text", id_b: "image"}

        db.update_co_retrieved_links([id_a, id_b], mods)
        db.update_co_retrieved_links([id_b, id_a], mods)

        links = db.get_links_between(id_a, id_b)
        co_links = [l for l in links if l["link_type"] == "co_retrieved"]
        assert len(co_links) == 1   # not 2


# ── File Tracking Tests ────────────────────────────────────────────────────
class TestFileTracking:

    @pytest.fixture
    def sample_file(self, tmp_path):
        f = tmp_path / "report.pdf"
        f.write_bytes(b"%PDF-1.4 sample content for hashing test 12345")
        return f

    def test_register_file(self, db, sample_file):
        file_hash = db.register_file(
            sample_file, "text", "sess-001", chunk_count=10
        )
        assert file_hash is not None
        assert len(file_hash) == 64   # SHA256 hex = 64 chars

    def test_is_duplicate_true(self, db, sample_file):
        db.register_file(sample_file, "text", "sess-001", chunk_count=5)
        assert db.is_duplicate(sample_file) is True

    def test_is_duplicate_false_new_file(self, db, tmp_path):
        new_file = tmp_path / "new_report.pdf"
        new_file.write_bytes(b"completely different content xyz 9999")
        assert db.is_duplicate(new_file) is False

    def test_different_content_not_duplicate(self, db, tmp_path):
        file_a = tmp_path / "a.pdf"
        file_b = tmp_path / "b.pdf"
        file_a.write_bytes(b"content of file A unique 111")
        file_b.write_bytes(b"content of file B unique 222")

        db.register_file(file_a, "text", "sess-001", chunk_count=3)
        assert db.is_duplicate(file_b) is False

    def test_get_ingested_files(self, db, sample_file):
        db.register_file(sample_file, "text", "sess-001", chunk_count=5)
        files = db.get_ingested_files()
        assert len(files) == 1
        assert files[0]["filename"] == sample_file.name
        assert files[0]["chunk_count"] == 5

    def test_get_ingested_files_session_filter(self, db, tmp_path):
        f1 = tmp_path / "f1.pdf"
        f2 = tmp_path / "f2.pdf"
        f1.write_bytes(b"file one content aaa")
        f2.write_bytes(b"file two content bbb")

        db.register_file(f1, "text",  "sess-A", chunk_count=3)
        db.register_file(f2, "audio", "sess-B", chunk_count=7)

        sess_a_files = db.get_ingested_files("sess-A")
        assert len(sess_a_files) == 1
        assert sess_a_files[0]["filename"] == "f1.pdf"

    def test_update_file_status(self, db, sample_file):
        file_hash = db.register_file(
            sample_file, "text", "sess-001",
            chunk_count=0, status="pending"
        )
        db.update_file_status(file_hash, "complete", chunk_count=15)

        files = db.get_ingested_files()
        assert files[0]["status"] == "complete"
        assert files[0]["chunk_count"] == 15


# ── Statistics Tests ───────────────────────────────────────────────────────
class TestStatistics:

    def test_stats_reflect_data(self, db):
        db.add_message("sess-001", "user", "Hello")
        db.add_message("sess-001", "assistant", "Hi")

        text  = _make_text_chunk(page=1)
        image = _make_image_chunk(page=1)
        db.generate_same_page_links([text], [image])

        stats = db.get_stats()
        assert stats["chat_messages"] == 2
        assert stats["total_links"]   == 1
        assert "same_page" in stats["links_by_type"]

    def test_stats_link_type_breakdown(self, db):
        text  = _make_text_chunk(page=1)
        image = _make_image_chunk(page=1)
        audio = _make_audio_chunk()

        db.generate_same_page_links([text], [image])
        db.generate_temporal_links([audio], [text], "sess-001")

        stats = db.get_stats()
        assert stats["links_by_type"].get("same_page", 0)          == 1
        assert stats["links_by_type"].get("temporal_proximity", 0) == 1


# ── Thread Safety Tests ────────────────────────────────────────────────────
class TestThreadSafety:

    @pytest.mark.slow
    def test_concurrent_message_inserts(self, db):
        """Multiple threads writing chat messages concurrently must not crash."""
        errors = []

        def insert_messages(thread_id):
            try:
                for i in range(10):
                    db.add_message(
                        f"sess-{thread_id}",
                        "user",
                        f"Thread {thread_id} message {i}",
                    )
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=insert_messages, args=(i,))
                   for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == [], f"Thread errors: {errors}"

        # All 50 messages should be present
        total = sum(
            len(db.get_history(f"sess-{i}", last_n=100))
            for i in range(5)
        )
        assert total == 50

    @pytest.mark.slow
    def test_concurrent_link_inserts(self, db):
        """Concurrent link creation must not produce duplicates."""
        text  = _make_text_chunk(page=1)
        image = _make_image_chunk(page=1)
        errors = []

        def create_links():
            try:
                db.generate_same_page_links([text], [image])
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=create_links) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert errors == []

        links = db.get_links_between(text.chunk_id, image.chunk_id)
        same_page = [l for l in links if l["link_type"] == "same_page"]
        assert len(same_page) == 1   # exactly one, not 5
