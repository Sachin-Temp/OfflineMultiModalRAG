"""
modules/indexing/sqlite_store.py

SQLite persistence layer for the Multimodal RAG System.

Three tables:
    chat_history       — conversation memory per session
    cross_modal_links  — chunk-to-chunk cross-modal connections
    ingested_files     — file ingestion tracking + deduplication

Cross-modal link types:
    SAME_PAGE          — text chunk ↔ image chunk on same PDF page (strength=1.0)
    TEMPORAL_PROXIMITY — audio chunk ↔ text chunk in same session (strength=0.7)
    CO_RETRIEVED       — any two chunks returned together in same query (dynamic)
    SEMANTIC           — text ↔ image with CLIP cosine similarity > 0.75

This module is the only place that reads/writes SQLite.
All other modules import SQLiteStore and call its methods.
Thread safety: uses threading.Lock for write operations.
"""

import hashlib
import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

import numpy as np
from loguru import logger

from config.settings import (
    SQLITE_DB_PATH,
    CHAT_HISTORY_TURNS,
    CLIP_SEMANTIC_LINK_THRESHOLD,
    CROSS_MODAL_LINK_STRENGTH_THRESHOLD,
    CLIP_DIM,
)
from models.schemas import (
    TextChunk,
    ImageChunk,
    AudioChunk,
    Modality,
)


# ── Link Type Constants ────────────────────────────────────────────────────
LINK_SAME_PAGE          = "same_page"
LINK_TEMPORAL_PROXIMITY = "temporal_proximity"
LINK_CO_RETRIEVED       = "co_retrieved"
LINK_SEMANTIC           = "semantic"

# ── Default Strengths ──────────────────────────────────────────────────────
STRENGTH_SAME_PAGE          = 1.0
STRENGTH_TEMPORAL_PROXIMITY = 0.7


# ── Schema SQL ─────────────────────────────────────────────────────────────
_SCHEMA_SQL = """
-- Conversation memory: one row per message turn
CREATE TABLE IF NOT EXISTS chat_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT    NOT NULL,
    role         TEXT    NOT NULL CHECK(role IN ('user', 'assistant')),
    message      TEXT    NOT NULL,
    cited_chunks TEXT,                          -- JSON array of chunk_ids used
    timestamp    DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_chat_session
    ON chat_history(session_id, timestamp);

-- Cross-modal links: connections between chunks of different modalities
CREATE TABLE IF NOT EXISTS cross_modal_links (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    chunk_id_a   TEXT    NOT NULL,
    chunk_id_b   TEXT    NOT NULL,
    modality_a   TEXT    NOT NULL,              -- text | image | audio
    modality_b   TEXT    NOT NULL,
    link_type    TEXT    NOT NULL,              -- same_page | temporal_proximity | co_retrieved | semantic
    strength     REAL    NOT NULL DEFAULT 0.0, -- 0.0 to 1.0
    source_file  TEXT,
    co_retrievals INTEGER DEFAULT 0,            -- counter for co_retrieved links
    total_retrievals INTEGER DEFAULT 0,         -- denominator for co_retrieved strength
    created_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    updated_at   DATETIME DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(chunk_id_a, chunk_id_b, link_type)   -- prevent duplicate links
);
CREATE INDEX IF NOT EXISTS idx_links_chunk_a
    ON cross_modal_links(chunk_id_a);
CREATE INDEX IF NOT EXISTS idx_links_chunk_b
    ON cross_modal_links(chunk_id_b);
CREATE INDEX IF NOT EXISTS idx_links_type
    ON cross_modal_links(link_type);

-- File ingestion tracking: one row per ingested file
CREATE TABLE IF NOT EXISTS ingested_files (
    file_hash    TEXT    PRIMARY KEY,           -- SHA256 of file content
    filename     TEXT    NOT NULL,
    file_path    TEXT,
    modality     TEXT,
    session_id   TEXT,
    chunk_count  INTEGER DEFAULT 0,
    status       TEXT    DEFAULT 'pending',     -- pending | complete | failed
    ingested_at  DATETIME DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_files_session
    ON ingested_files(session_id);
CREATE INDEX IF NOT EXISTS idx_files_status
    ON ingested_files(status);
"""


# ── SQLiteStore ────────────────────────────────────────────────────────────
class SQLiteStore:
    """
    Central SQLite persistence layer.

    Thread-safe: all write operations use a threading.Lock.
    Read operations use separate connections for concurrency.

    Usage:
        store = SQLiteStore()
        store.initialize()   # creates tables if needed

        # Chat history
        store.add_message(session_id, role, message, cited_chunks)
        history = store.get_history(session_id, last_n=5)

        # Link generation (call after ingestion)
        store.generate_same_page_links(text_chunks, image_chunks)
        store.generate_temporal_links(audio_chunks, text_chunks, session_id)
        store.generate_semantic_links(text_chunks, image_chunks)

        # Link updates (call after retrieval)
        store.update_co_retrieved_links(chunk_ids_returned_together)

        # Link queries (used by retrieval engine Phase 6)
        linked = store.get_linked_chunks(chunk_id, min_strength=0.6)

        # File tracking
        store.register_file(file_path, modality, session_id, chunk_count)
        is_dup = store.is_duplicate(file_path)
    """

    def __init__(self, db_path: Optional[Path] = None):
        self._db_path = Path(db_path or SQLITE_DB_PATH).resolve()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()

    # ── Connection Management ──────────────────────────────────────────────
    def _get_connection(self) -> sqlite3.Connection:
        """
        Get a SQLite connection with WAL mode enabled.
        WAL (Write-Ahead Logging) allows concurrent reads during writes.
        Row factory set to sqlite3.Row for dict-like access.
        """
        conn = sqlite3.connect(
            str(self._db_path),
            timeout=30,
            check_same_thread=False,
        )
        conn.row_factory = sqlite3.Row
        # WAL mode: readers don't block writers and vice versa
        conn.execute("PRAGMA journal_mode=WAL")
        # Foreign key enforcement
        conn.execute("PRAGMA foreign_keys=ON")
        # Synchronous=NORMAL: good balance of safety and speed
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def initialize(self):
        """
        Create all tables and indexes if they don't exist.
        Idempotent — safe to call multiple times.
        Uses CREATE TABLE IF NOT EXISTS throughout.
        """
        with self._write_lock:
            conn = self._get_connection()
            try:
                conn.executescript(_SCHEMA_SQL)
                conn.commit()
                logger.success(
                    f"SQLiteStore initialized: {self._db_path}"
                )
            except Exception as e:
                logger.error(f"SQLiteStore initialization failed: {e}")
                raise
            finally:
                conn.close()

    # ── Chat History ───────────────────────────────────────────────────────
    def add_message(
        self,
        session_id: str,
        role: str,
        message: str,
        cited_chunks: Optional[List[str]] = None,
    ) -> int:
        """
        Add a single message turn to chat history.

        Args:
            session_id:    Session identifier
            role:          'user' or 'assistant'
            message:       Message text content
            cited_chunks:  List of chunk_ids cited in this message (for assistant turns)

        Returns:
            Row ID of the inserted message.
        """
        if role not in ("user", "assistant"):
            raise ValueError(f"role must be 'user' or 'assistant', got '{role}'")

        cited_json = json.dumps(cited_chunks or [])

        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    """
                    INSERT INTO chat_history
                        (session_id, role, message, cited_chunks)
                    VALUES (?, ?, ?, ?)
                    """,
                    (session_id, role, message, cited_json),
                )
                conn.commit()
                row_id = cursor.lastrowid
                logger.debug(
                    f"Chat message added: session={session_id} "
                    f"role={role} id={row_id}"
                )
                return row_id
            finally:
                conn.close()

    def get_history(
        self,
        session_id: str,
        last_n: int = CHAT_HISTORY_TURNS,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the last N message turns for a session.
        Returns in chronological order (oldest first).

        Args:
            session_id: Session identifier
            last_n:     Maximum number of turns to return (default: 5)

        Returns:
            List of message dicts:
            [{"role": str, "message": str, "cited_chunks": list, "timestamp": str}]
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT role, message, cited_chunks, timestamp
                FROM chat_history
                WHERE session_id = ?
                ORDER BY timestamp DESC
                LIMIT ?
                """,
                (session_id, last_n),
            ).fetchall()

            # Reverse to get chronological order (oldest first)
            messages = []
            for row in reversed(rows):
                try:
                    cited = json.loads(row["cited_chunks"] or "[]")
                except json.JSONDecodeError:
                    cited = []
                messages.append({
                    "role":          row["role"],
                    "message":       row["message"],
                    "cited_chunks":  cited,
                    "timestamp":     row["timestamp"],
                })
            return messages
        finally:
            conn.close()

    def get_session_ids(self) -> List[str]:
        """Return all unique session IDs that have chat history."""
        conn = self._get_connection()
        try:
            rows = conn.execute(
                "SELECT session_id FROM chat_history "
                "GROUP BY session_id "
                "ORDER BY MAX(timestamp) DESC"
            ).fetchall()
            return [row["session_id"] for row in rows]
        finally:
            conn.close()

    def delete_session_history(self, session_id: str) -> int:
        """
        Delete all chat history for a session.
        Returns number of rows deleted.
        """
        with self._write_lock:
            conn = self._get_connection()
            try:
                cursor = conn.execute(
                    "DELETE FROM chat_history WHERE session_id = ?",
                    (session_id,),
                )
                conn.commit()
                deleted = cursor.rowcount
                logger.info(
                    f"Deleted {deleted} chat messages for session {session_id}"
                )
                return deleted
            finally:
                conn.close()

    # ── Cross-Modal Link Generation ────────────────────────────────────────
    def _upsert_link(
        self,
        chunk_id_a: str,
        chunk_id_b: str,
        modality_a: str,
        modality_b: str,
        link_type: str,
        strength: float,
        source_file: Optional[str] = None,
        conn: Optional[sqlite3.Connection] = None,
    ):
        """
        Insert a new link or update strength if link already exists.
        Uses UNIQUE(chunk_id_a, chunk_id_b, link_type) constraint.

        Canonical ordering: chunk_id_a < chunk_id_b (lexicographic)
        to avoid duplicate links in both directions.

        Args:
            conn: Optional existing connection (for batch operations).
                  If None, creates its own connection.
        """
        # Canonical ordering — always store smaller ID first
        if chunk_id_a > chunk_id_b:
            chunk_id_a, chunk_id_b = chunk_id_b, chunk_id_a
            modality_a, modality_b = modality_b, modality_a

        now = datetime.now(timezone.utc).isoformat()

        sql = """
            INSERT INTO cross_modal_links
                (chunk_id_a, chunk_id_b, modality_a, modality_b,
                 link_type, strength, source_file, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(chunk_id_a, chunk_id_b, link_type)
            DO UPDATE SET
                strength   = MAX(strength, excluded.strength),
                updated_at = excluded.updated_at
        """
        params = (
            chunk_id_a, chunk_id_b, modality_a, modality_b,
            link_type, round(strength, 6), source_file, now,
        )

        close_after = conn is None
        if conn is None:
            conn = self._get_connection()
        try:
            conn.execute(sql, params)
            if close_after:
                conn.commit()
        finally:
            if close_after:
                conn.close()

    def generate_same_page_links(
        self,
        text_chunks: List[TextChunk],
        image_chunks: List[ImageChunk],
    ) -> int:
        """
        Generate SAME_PAGE links between text and image chunks that share
        the same source_file and page_number.

        Called after ingesting a PDF that contains both text and embedded images.
        Strength = 1.0 (co-located on same page = strongest possible link).

        Args:
            text_chunks:  TextChunk objects from Phase 1 PDF ingestion
            image_chunks: ImageChunk objects from Phase 2 PDF image extraction

        Returns:
            Number of links created.
        """
        if not text_chunks or not image_chunks:
            return 0

        # Build lookup: (source_file, page_number) → list of image chunks
        image_by_page: Dict[Tuple[str, int], List[ImageChunk]] = {}
        for img_chunk in image_chunks:
            key = (img_chunk.source_file, img_chunk.page_number)
            image_by_page.setdefault(key, []).append(img_chunk)

        links_created = 0
        with self._write_lock:
            conn = self._get_connection()
            try:
                for text_chunk in text_chunks:
                    key = (text_chunk.source_file, text_chunk.page_number)
                    matching_images = image_by_page.get(key, [])

                    for img_chunk in matching_images:
                        self._upsert_link(
                            chunk_id_a=text_chunk.chunk_id,
                            chunk_id_b=img_chunk.chunk_id,
                            modality_a=text_chunk.modality.value,
                            modality_b=img_chunk.modality.value,
                            link_type=LINK_SAME_PAGE,
                            strength=STRENGTH_SAME_PAGE,
                            source_file=text_chunk.source_file,
                            conn=conn,
                        )
                        links_created += 1

                conn.commit()
            except Exception as e:
                logger.error(f"generate_same_page_links failed: {e}")
                conn.rollback()
            finally:
                conn.close()

        logger.info(
            f"SAME_PAGE links: created {links_created} links "
            f"({len(text_chunks)} text × {len(image_chunks)} image chunks)"
        )
        return links_created

    def generate_temporal_links(
        self,
        audio_chunks: List[AudioChunk],
        text_chunks: List[TextChunk],
        session_id: str,
    ) -> int:
        """
        Generate TEMPORAL_PROXIMITY links between audio chunks and text chunks
        that belong to the same session.

        Rationale: if a user uploads a PDF and an audio recording in the same
        session, it's likely the audio discusses the document. We link all
        audio chunks to all text chunks in the session with strength=0.7.

        Called after ingesting an audio file when there are already text chunks
        in the same session.

        Args:
            audio_chunks: AudioChunk objects just ingested
            text_chunks:  TextChunk objects already in this session
            session_id:   The shared session identifier

        Returns:
            Number of links created.
        """
        if not audio_chunks or not text_chunks:
            return 0

        links_created = 0
        with self._write_lock:
            conn = self._get_connection()
            try:
                for audio_chunk in audio_chunks:
                    for text_chunk in text_chunks:
                        # Only link chunks from the same session
                        if (audio_chunk.session_id != session_id or
                                text_chunk.session_id != session_id):
                            continue

                        self._upsert_link(
                            chunk_id_a=audio_chunk.chunk_id,
                            chunk_id_b=text_chunk.chunk_id,
                            modality_a=audio_chunk.modality.value,
                            modality_b=text_chunk.modality.value,
                            link_type=LINK_TEMPORAL_PROXIMITY,
                            strength=STRENGTH_TEMPORAL_PROXIMITY,
                            source_file=None,
                            conn=conn,
                        )
                        links_created += 1

                conn.commit()
            except Exception as e:
                logger.error(f"generate_temporal_links failed: {e}")
                conn.rollback()
            finally:
                conn.close()

        logger.info(
            f"TEMPORAL_PROXIMITY links: created {links_created} links "
            f"(session={session_id})"
        )
        return links_created

    def generate_semantic_links(
        self,
        text_chunks: List[TextChunk],
        image_chunks: List[ImageChunk],
    ) -> int:
        """
        Generate SEMANTIC links between text and image chunks using CLIP's
        native text-image alignment.

        How it works:
        1. For each text chunk, encode it using CLIP's TEXT encoder → 512-dim
        2. Each image chunk already has a CLIP IMAGE embedding (512-dim)
        3. Compute cosine similarity between every text-image pair
        4. Create SEMANTIC link if similarity > CLIP_SEMANTIC_LINK_THRESHOLD (0.75)

        CRITICAL: This ONLY works between text and image chunks because:
        - CLIP text encoder and CLIP image encoder share the same 512-dim space
        - BGE-M3 text embeddings (1024-dim) are in a DIFFERENT space
        - You CANNOT compare BGE-M3 text embeddings with CLIP image embeddings
        - Only CLIP-text vs CLIP-image comparisons are semantically meaningful

        This method loads CLIP temporarily for encoding text chunks.
        It releases CLIP after completion.

        Args:
            text_chunks:  TextChunk objects (will be encoded with CLIP text encoder)
            image_chunks: ImageChunk objects (must have CLIP embeddings set)

        Returns:
            Number of semantic links created.
        """
        if not text_chunks or not image_chunks:
            return 0

        # Filter image chunks that have CLIP embeddings
        image_chunks_with_emb = [
            c for c in image_chunks
            if c.embedding is not None and len(c.embedding) == CLIP_DIM
        ]
        if not image_chunks_with_emb:
            logger.warning(
                "generate_semantic_links: no image chunks have CLIP embeddings. "
                "Was Phase 2 (ImageIngestor) run successfully?"
            )
            return 0

        # Encode text chunks with CLIP text encoder
        text_clip_embeddings = _encode_texts_with_clip(
            [c.text for c in text_chunks]
        )

        if not any(e is not None for e in text_clip_embeddings):
            logger.warning(
                "generate_semantic_links: CLIP text encoding failed for all chunks"
            )
            return 0

        # Build numpy arrays for efficient batch cosine similarity
        img_matrix = np.array(
            [c.embedding for c in image_chunks_with_emb],
            dtype=np.float32,
        )  # shape: (n_images, 512)

        links_created = 0
        with self._write_lock:
            conn = self._get_connection()
            try:
                for text_idx, text_chunk in enumerate(text_chunks):
                    text_emb = text_clip_embeddings[text_idx]
                    if text_emb is None:
                        continue

                    text_vec = np.array(text_emb, dtype=np.float32)

                    # Cosine similarity: dot product of L2-normalized vectors
                    # Both CLIP text and image embeddings are L2-normalized
                    # So cosine_sim = dot_product
                    similarities = img_matrix @ text_vec   # shape: (n_images,)

                    for img_idx, similarity in enumerate(similarities):
                        sim_float = float(similarity)
                        if sim_float >= CLIP_SEMANTIC_LINK_THRESHOLD:
                            img_chunk = image_chunks_with_emb[img_idx]
                            self._upsert_link(
                                chunk_id_a=text_chunk.chunk_id,
                                chunk_id_b=img_chunk.chunk_id,
                                modality_a=text_chunk.modality.value,
                                modality_b=img_chunk.modality.value,
                                link_type=LINK_SEMANTIC,
                                strength=sim_float,
                                source_file=text_chunk.source_file,
                                conn=conn,
                            )
                            links_created += 1

                conn.commit()
            except Exception as e:
                logger.error(f"generate_semantic_links failed: {e}")
                conn.rollback()
            finally:
                conn.close()

        logger.info(
            f"SEMANTIC links: created {links_created} links "
            f"(threshold={CLIP_SEMANTIC_LINK_THRESHOLD})"
        )
        return links_created

    def update_co_retrieved_links(
        self,
        chunk_ids: List[str],
        chunk_modalities: Dict[str, str],
    ) -> int:
        """
        Update CO_RETRIEVED links for every pair of chunks returned
        in the same retrieval query.

        Called by the retrieval engine (Phase 6) after every query.
        Increments co_retrievals counter for each pair.
        Recomputes strength = co_retrievals / total_retrievals.

        This makes the system smarter over time — frequently co-retrieved
        chunks become strongly linked and get included in context more often.

        Args:
            chunk_ids:        List of chunk_ids returned in this query
            chunk_modalities: Dict mapping chunk_id → modality string

        Returns:
            Number of links updated or created.
        """
        if len(chunk_ids) < 2:
            return 0

        pairs_updated = 0
        now = datetime.now(timezone.utc).isoformat()

        with self._write_lock:
            conn = self._get_connection()
            try:
                # Generate all unique pairs
                for i in range(len(chunk_ids)):
                    for j in range(i + 1, len(chunk_ids)):
                        id_a = chunk_ids[i]
                        id_b = chunk_ids[j]

                        # Canonical ordering
                        if id_a > id_b:
                            id_a, id_b = id_b, id_a

                        mod_a = chunk_modalities.get(id_a, "text")
                        mod_b = chunk_modalities.get(id_b, "text")

                        # Fetch existing link if any
                        existing = conn.execute(
                            """
                            SELECT id, co_retrievals, total_retrievals
                            FROM cross_modal_links
                            WHERE chunk_id_a = ? AND chunk_id_b = ?
                              AND link_type = ?
                            """,
                            (id_a, id_b, LINK_CO_RETRIEVED),
                        ).fetchone()

                        if existing:
                            new_co   = existing["co_retrievals"] + 1
                            new_tot  = existing["total_retrievals"] + 1
                            strength = new_co / new_tot
                            conn.execute(
                                """
                                UPDATE cross_modal_links
                                SET co_retrievals    = ?,
                                    total_retrievals = ?,
                                    strength         = ?,
                                    updated_at       = ?
                                WHERE id = ?
                                """,
                                (new_co, new_tot, round(strength, 6),
                                 now, existing["id"]),
                            )
                        else:
                            # First co-retrieval: strength = 1/1 = 1.0
                            conn.execute(
                                """
                                INSERT INTO cross_modal_links
                                    (chunk_id_a, chunk_id_b, modality_a, modality_b,
                                     link_type, strength, co_retrievals,
                                     total_retrievals, updated_at)
                                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """,
                                (id_a, id_b, mod_a, mod_b,
                                 LINK_CO_RETRIEVED, 1.0, 1, 1, now),
                            )
                        pairs_updated += 1

                conn.commit()
            except Exception as e:
                logger.error(f"update_co_retrieved_links failed: {e}")
                conn.rollback()
            finally:
                conn.close()

        logger.debug(f"CO_RETRIEVED: updated {pairs_updated} pairs")
        return pairs_updated

    # ── Link Queries ───────────────────────────────────────────────────────
    def get_linked_chunks(
        self,
        chunk_id: str,
        min_strength: float = CROSS_MODAL_LINK_STRENGTH_THRESHOLD,
        link_types: Optional[List[str]] = None,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """
        Get all chunks linked to a given chunk_id above a minimum strength.

        Used by the retrieval engine (Phase 6) to enrich the Top-5 gold chunks
        with up to 2 additional strongly-linked chunks.

        Args:
            chunk_id:    The chunk to find links for
            min_strength: Minimum link strength to include (default: 0.6)
            link_types:  If provided, filter to these link types only
            limit:       Maximum number of linked chunks to return

        Returns:
            List of link dicts ordered by strength descending:
            [{
                "linked_chunk_id": str,
                "linked_modality": str,
                "link_type": str,
                "strength": float,
                "source_file": str,
            }]
        """
        conn = self._get_connection()
        try:
            # Query both directions (chunk_id_a or chunk_id_b)
            if link_types:
                placeholders = ",".join("?" * len(link_types))
                sql = f"""
                    SELECT
                        CASE WHEN chunk_id_a = ? THEN chunk_id_b ELSE chunk_id_a END
                            AS linked_chunk_id,
                        CASE WHEN chunk_id_a = ? THEN modality_b ELSE modality_a END
                            AS linked_modality,
                        link_type,
                        strength,
                        source_file
                    FROM cross_modal_links
                    WHERE (chunk_id_a = ? OR chunk_id_b = ?)
                      AND strength >= ?
                      AND link_type IN ({placeholders})
                    ORDER BY strength DESC
                    LIMIT ?
                """
                params = (
                    chunk_id, chunk_id,
                    chunk_id, chunk_id,
                    min_strength,
                    *link_types,
                    limit,
                )
            else:
                sql = """
                    SELECT
                        CASE WHEN chunk_id_a = ? THEN chunk_id_b ELSE chunk_id_a END
                            AS linked_chunk_id,
                        CASE WHEN chunk_id_a = ? THEN modality_b ELSE modality_a END
                            AS linked_modality,
                        link_type,
                        strength,
                        source_file
                    FROM cross_modal_links
                    WHERE (chunk_id_a = ? OR chunk_id_b = ?)
                      AND strength >= ?
                    ORDER BY strength DESC
                    LIMIT ?
                """
                params = (
                    chunk_id, chunk_id,
                    chunk_id, chunk_id,
                    min_strength,
                    limit,
                )

            rows = conn.execute(sql, params).fetchall()
            return [dict(row) for row in rows]

        finally:
            conn.close()

    def get_links_between(
        self,
        chunk_id_a: str,
        chunk_id_b: str,
    ) -> List[Dict[str, Any]]:
        """
        Get all links between two specific chunks (all link types).
        Used by the citation engine (Phase 8) to explain connections.

        Returns:
            List of link dicts, or empty list if no links exist.
        """
        # Canonical ordering
        if chunk_id_a > chunk_id_b:
            chunk_id_a, chunk_id_b = chunk_id_b, chunk_id_a

        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT chunk_id_a, chunk_id_b, modality_a, modality_b,
                       link_type, strength, source_file, created_at
                FROM cross_modal_links
                WHERE chunk_id_a = ? AND chunk_id_b = ?
                ORDER BY strength DESC
                """,
                (chunk_id_a, chunk_id_b),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def get_all_links_for_session(
        self,
        session_id: str,
        min_strength: float = 0.0,
    ) -> List[Dict[str, Any]]:
        """
        Get all cross-modal links for chunks in a session.
        Used by the frontend to render the link visualization graph.

        Args:
            session_id:   Session to get links for
            min_strength: Minimum strength filter

        Returns:
            List of link dicts with both chunk IDs and metadata.
        """
        conn = self._get_connection()
        try:
            rows = conn.execute(
                """
                SELECT chunk_id_a, chunk_id_b, modality_a, modality_b,
                       link_type, strength, source_file, created_at
                FROM cross_modal_links
                WHERE strength >= ?
                ORDER BY strength DESC
                """,
                (min_strength,),
            ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    # ── File Tracking ──────────────────────────────────────────────────────
    def _compute_file_hash(self, file_path: Path) -> str:
        """
        Compute SHA256 hash of file content.
        Used for deduplication — same content = same hash.
        Reads in 64KB chunks to handle large files efficiently.
        """
        sha256 = hashlib.sha256()
        with open(str(file_path), "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()

    def register_file(
        self,
        file_path: Path,
        modality: str,
        session_id: str,
        chunk_count: int,
        status: str = "complete",
    ) -> str:
        """
        Register a file as ingested in the database.
        Uses SHA256 hash as primary key for deduplication.

        Args:
            file_path:   Path to the ingested file
            modality:    'text', 'image', or 'audio'
            session_id:  Session this file belongs to
            chunk_count: Number of chunks produced
            status:      'pending', 'complete', or 'failed'

        Returns:
            SHA256 file hash (primary key in ingested_files table)
        """
        file_path = Path(file_path)
        file_hash = self._compute_file_hash(file_path)

        with self._write_lock:
            conn = self._get_connection()
            try:
                conn.execute(
                    """
                    INSERT OR REPLACE INTO ingested_files
                        (file_hash, filename, file_path, modality,
                         session_id, chunk_count, status, ingested_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        file_hash,
                        file_path.name,
                        str(file_path),
                        modality,
                        session_id,
                        chunk_count,
                        status,
                        datetime.now(timezone.utc).isoformat(),
                    ),
                )
                conn.commit()
                logger.info(
                    f"File registered: {file_path.name} "
                    f"(hash={file_hash[:8]}..., chunks={chunk_count})"
                )
                return file_hash
            finally:
                conn.close()

    def is_duplicate(self, file_path: Path) -> bool:
        """
        Check if a file has already been ingested (by content hash).
        Returns True if the same file content was previously ingested.

        Args:
            file_path: Path to check

        Returns:
            True if duplicate (already ingested), False if new.
        """
        try:
            file_hash = self._compute_file_hash(Path(file_path))
        except Exception as e:
            logger.warning(f"Could not hash file {file_path}: {e}")
            return False

        conn = self._get_connection()
        try:
            row = conn.execute(
                "SELECT file_hash, status FROM ingested_files WHERE file_hash = ?",
                (file_hash,),
            ).fetchone()
            if row and row["status"] == "complete":
                logger.info(
                    f"Duplicate file detected: {Path(file_path).name} "
                    f"(hash={file_hash[:8]}...)"
                )
                return True
            return False
        finally:
            conn.close()

    def get_ingested_files(
        self,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Get list of all ingested files, optionally filtered by session.

        Returns:
            List of file dicts with keys:
            file_hash, filename, modality, session_id, chunk_count,
            status, ingested_at
        """
        conn = self._get_connection()
        try:
            if session_id:
                rows = conn.execute(
                    """
                    SELECT file_hash, filename, file_path, modality,
                           session_id, chunk_count, status, ingested_at
                    FROM ingested_files
                    WHERE session_id = ?
                    ORDER BY ingested_at DESC
                    """,
                    (session_id,),
                ).fetchall()
            else:
                rows = conn.execute(
                    """
                    SELECT file_hash, filename, file_path, modality,
                           session_id, chunk_count, status, ingested_at
                    FROM ingested_files
                    ORDER BY ingested_at DESC
                    """
                ).fetchall()
            return [dict(row) for row in rows]
        finally:
            conn.close()

    def update_file_status(
        self,
        file_hash: str,
        status: str,
        chunk_count: Optional[int] = None,
    ):
        """
        Update the status of an ingested file record.
        Used to mark files as 'complete' or 'failed' after processing.
        """
        with self._write_lock:
            conn = self._get_connection()
            try:
                if chunk_count is not None:
                    conn.execute(
                        """
                        UPDATE ingested_files
                        SET status = ?, chunk_count = ?
                        WHERE file_hash = ?
                        """,
                        (status, chunk_count, file_hash),
                    )
                else:
                    conn.execute(
                        "UPDATE ingested_files SET status = ? WHERE file_hash = ?",
                        (status, file_hash),
                    )
                conn.commit()
            finally:
                conn.close()

    # ── Statistics & Health ────────────────────────────────────────────────
    def get_stats(self) -> Dict[str, Any]:
        """
        Return database statistics for the /health endpoint (Phase 9).

        Returns:
            Dict with counts for all tables and link type breakdown.
        """
        conn = self._get_connection()
        try:
            chat_count = conn.execute(
                "SELECT COUNT(*) FROM chat_history"
            ).fetchone()[0]

            link_count = conn.execute(
                "SELECT COUNT(*) FROM cross_modal_links"
            ).fetchone()[0]

            file_count = conn.execute(
                "SELECT COUNT(*) FROM ingested_files WHERE status = 'complete'"
            ).fetchone()[0]

            link_by_type = {}
            rows = conn.execute(
                """
                SELECT link_type, COUNT(*) as cnt
                FROM cross_modal_links
                GROUP BY link_type
                """
            ).fetchall()
            for row in rows:
                link_by_type[row["link_type"]] = row["cnt"]

            return {
                "chat_messages":  chat_count,
                "total_links":    link_count,
                "links_by_type":  link_by_type,
                "ingested_files": file_count,
                "db_path":        str(self._db_path),
                "status":         "ok",
            }
        except Exception as e:
            return {"status": f"error: {e}"}
        finally:
            conn.close()


# ── CLIP Text Encoding Helper ──────────────────────────────────────────────
def _encode_texts_with_clip(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Encode a list of text strings using CLIP's text encoder.
    Returns 512-dim embeddings in CLIP's native space.

    Used ONLY by generate_semantic_links() to find text-image semantic matches.
    Uses VRAMManager to acquire/release CLIP safely.

    IMPORTANT: These embeddings are in CLIP's 512-dim space and are
    directly comparable with ImageChunk.embedding values (also CLIP 512-dim).
    Do NOT use these for text-text similarity — use BGE-M3 for that.

    Returns:
        List of 512-dim float lists (same length as input texts).
        None entries for texts that failed to encode.
    """
    from core.vram_manager import vram_manager

    if not texts:
        return []

    results = [None] * len(texts)

    try:
        import clip
        import torch

        # Acquire VRAM for CLIP
        acquired = vram_manager.acquire("clip")
        if not acquired:
            logger.warning(
                "_encode_texts_with_clip: could not acquire VRAM for CLIP. "
                "Semantic links will not be generated."
            )
            return results

        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device=device)
        model.eval()

        batch_size = 64
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i: i + batch_size]
                try:
                    tokens = clip.tokenize(batch, truncate=True).to(device)
                    embeddings = model.encode_text(tokens)
                    embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
                    emb_list = embeddings.cpu().numpy().tolist()
                    for j, emb in enumerate(emb_list):
                        results[i + j] = emb
                except Exception as e:
                    logger.warning(f"CLIP text batch {i} failed: {e}")

        # Release CLIP
        del model
        torch.cuda.empty_cache()
        vram_manager.release("clip")

    except ImportError:
        logger.warning("CLIP not available — semantic links will not be generated")
    except Exception as e:
        logger.error(f"_encode_texts_with_clip failed: {e}")
        try:
            vram_manager.release("clip")
        except Exception:
            pass

    return results
