"""
modules/indexing/milvus_store.py

Manages three Milvus Lite vector collections — one per modality.

Collections:
    text_chunks:  BGE-M3 1024-dim embeddings, IVF_FLAT, COSINE
    image_chunks: CLIP 512-dim embeddings,    IVF_FLAT, COSINE
    audio_chunks: BGE-M3 1024-dim embeddings, IVF_FLAT, COSINE

Responsibilities:
    1. Create collections if they don't exist (idempotent)
    2. Compute BGE-M3 embeddings for TextChunk and AudioChunk objects
       (ImageChunk embeddings are already computed in Phase 2)
    3. Insert chunks into correct collection
    4. Search collections and return scored results
    5. Delete chunks by session_id or chunk_id
    6. Manage BGE-M3 model loading via VRAMManager

CRITICAL DIMENSION RULES:
    - text_chunks and audio_chunks: EXACTLY 1024-dim (BGE-M3)
    - image_chunks: EXACTLY 512-dim (CLIP ViT-B/32)
    - NEVER mix these spaces
    - NEVER project CLIP into 1024-dim or BGE-M3 into 512-dim

Connection strategy:
    On Linux/macOS: Uses Milvus Lite (file-based) via MILVUS_DB_PATH
    On Windows:     Connects to Milvus server at MILVUS_URI (default localhost:19530)
                    Run via Docker: docker-compose up -d
    Set env var MILVUS_URI to override the connection target on any platform.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple, Union

from loguru import logger

from config.settings import (
    MILVUS_DB_PATH,
    COLLECTION_TEXT,
    COLLECTION_IMAGE,
    COLLECTION_AUDIO,
    BGE_M3_DIM,
    CLIP_DIM,
    BGE_M3_MODEL,
    MILVUS_NLIST_TEXT,
    MILVUS_NLIST_IMAGE,
    MILVUS_NLIST_AUDIO,
    MILVUS_NPROBE,
    MILVUS_TOP_K,
)
from core.vram_manager import vram_manager
from models.schemas import (
    TextChunk,
    ImageChunk,
    AudioChunk,
    Modality,
)


# ── BGE-M3 Model Management ────────────────────────────────────────────────
_bge_m3_model = None


def _load_bge_m3():
    """
    Load BGE-M3 sentence transformer with VRAM management.
    Returns the loaded SentenceTransformer model.
    Acquires VRAM slot before loading.
    """
    global _bge_m3_model

    if _bge_m3_model is not None:
        logger.debug("BGE-M3 already loaded — reusing cached instance")
        return _bge_m3_model

    def _evict_bge_m3():
        global _bge_m3_model
        import torch
        if _bge_m3_model is not None:
            del _bge_m3_model
            _bge_m3_model = None
            torch.cuda.empty_cache()
            logger.info("BGE-M3 evicted from VRAM by VRAMManager")

    vram_manager.register_evict_callback("bge_m3", _evict_bge_m3)

    acquired = vram_manager.acquire("bge_m3")
    if not acquired:
        raise RuntimeError(
            "VRAMManager could not allocate VRAM for BGE-M3. "
            "Ensure LLM is not loaded before running indexing."
        )

    try:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading BGE-M3 on {device}...")
        model = SentenceTransformer(BGE_M3_MODEL, device=device)
        _bge_m3_model = model
        logger.success(f"BGE-M3 loaded on {device}")
        return model
    except Exception as e:
        vram_manager.release("bge_m3")
        raise RuntimeError(f"Failed to load BGE-M3: {e}") from e


def _unload_bge_m3():
    """
    Explicitly unload BGE-M3 and release VRAM.
    Call after finishing a batch of embedding computations.
    """
    global _bge_m3_model
    import torch
    if _bge_m3_model is not None:
        del _bge_m3_model
        _bge_m3_model = None
        torch.cuda.empty_cache()
        vram_manager.release("bge_m3")
        logger.info("BGE-M3 explicitly unloaded and VRAM released")


def _compute_bge_m3_embedding(text: str) -> Optional[List[float]]:
    """
    Compute a single BGE-M3 1024-dim embedding for a text string.
    Returns list of 1024 floats (L2-normalized) or None on failure.
    """
    try:
        model = _load_bge_m3()
        embedding = model.encode(
            text,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        embedding_list = embedding.tolist()

        assert len(embedding_list) == BGE_M3_DIM, (
            f"Expected {BGE_M3_DIM}-dim BGE-M3 embedding, "
            f"got {len(embedding_list)}"
        )
        return embedding_list
    except Exception as e:
        logger.error(f"BGE-M3 embedding failed: {e}")
        return None


def _compute_bge_m3_batch(texts: List[str]) -> List[Optional[List[float]]]:
    """
    Compute BGE-M3 embeddings for a batch of texts.
    More efficient than calling _compute_bge_m3_embedding() in a loop.
    Returns list of embeddings (same length as texts), None for failures.
    """
    if not texts:
        return []
    try:
        model = _load_bge_m3()
        embeddings = model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 10,
            batch_size=32,
            convert_to_numpy=True,
        )
        result = []
        for emb in embeddings:
            emb_list = emb.tolist()
            if len(emb_list) == BGE_M3_DIM:
                result.append(emb_list)
            else:
                logger.error(f"BGE-M3 wrong dimension: {len(emb_list)}")
                result.append(None)
        return result
    except Exception as e:
        logger.error(f"BGE-M3 batch embedding failed: {e}")
        return [None] * len(texts)


# ── Milvus Connection Helper ──────────────────────────────────────────────
def _get_milvus_uri() -> str:
    """
    Determine the Milvus connection URI based on platform and environment.

    Priority:
      1. MILVUS_URI env var (explicit override)
      2. On Linux/macOS: file-based Milvus Lite path (MILVUS_DB_PATH)
      3. On Windows: http://localhost:19530 (Docker Milvus required)
    """
    # Explicit override always wins
    env_uri = os.getenv("MILVUS_URI")
    if env_uri:
        return env_uri

    # On non-Windows: try Milvus Lite (file-based) by default
    if sys.platform != "win32":
        db_path = str(Path(MILVUS_DB_PATH).resolve())
        try:
            Path(db_path).parent.mkdir(parents=True, exist_ok=True)
            return db_path
        except Exception:
            pass

    # On Windows (or fallback): Milvus Lite not available on Python 3.13+, use Docker server
    return os.getenv("MILVUS_HOST", "http://localhost:19530")


# ── MilvusStore Class ──────────────────────────────────────────────────────
class MilvusStore:
    """
    Manages all Milvus vector collections for the RAG system.

    Collections:
        text_chunks:  BGE-M3 1024-dim, IVF_FLAT, COSINE
        image_chunks: CLIP 512-dim,    IVF_FLAT, COSINE
        audio_chunks: BGE-M3 1024-dim, IVF_FLAT, COSINE

    Usage:
        store = MilvusStore()
        store.initialize()   # creates collections if needed

        # Insert chunks
        store.insert_text_chunks(text_chunks)
        store.insert_image_chunks(image_chunks)
        store.insert_audio_chunks(audio_chunks)

        # Search
        results = store.search_text(query_embedding, top_k=50)
        results = store.search_images(clip_embedding, top_k=50)
        results = store.search_audio(query_embedding, top_k=50)

        # Cleanup
        store.release_models()
    """

    def __init__(self, uri: Optional[str] = None):
        """
        Initialize the MilvusStore.

        Args:
            uri: Optional explicit Milvus URI. If not provided, auto-detected
                 based on platform (Milvus Lite on Linux/Mac, Docker on Windows).
        """
        self._client = None
        self._initialized = False
        self._uri = uri or _get_milvus_uri()

    def _get_client(self):
        """
        Get or create the Milvus client.
        Uses auto-detected URI based on platform.
        """
        if self._client is not None:
            return self._client

        try:
            from pymilvus import MilvusClient
            self._client = MilvusClient(uri=self._uri)
            logger.info(f"Milvus client connected: {self._uri}")
            return self._client
        except Exception as e:
            raise RuntimeError(
                f"Failed to connect to Milvus at '{self._uri}': {e}\n"
                f"On Windows, ensure Milvus is running via Docker:\n"
                f"  docker-compose up -d"
            ) from e

    def initialize(self):
        """
        Create all three collections if they don't exist.
        Idempotent — safe to call multiple times.
        """
        client = self._get_client()

        self._create_collection_if_not_exists(
            name=COLLECTION_TEXT,
            dim=BGE_M3_DIM,
            description="Text chunks with BGE-M3 1024-dim embeddings",
        )
        self._create_collection_if_not_exists(
            name=COLLECTION_IMAGE,
            dim=CLIP_DIM,
            description="Image chunks with CLIP 512-dim embeddings",
        )
        self._create_collection_if_not_exists(
            name=COLLECTION_AUDIO,
            dim=BGE_M3_DIM,
            description="Audio transcript chunks with BGE-M3 1024-dim embeddings",
        )

        self._initialized = True
        logger.success(
            f"MilvusStore initialized: collections "
            f"[{COLLECTION_TEXT}, {COLLECTION_IMAGE}, {COLLECTION_AUDIO}]"
        )

    def _create_collection_if_not_exists(
        self,
        name: str,
        dim: int,
        description: str,
    ):
        """
        Create a Milvus collection with COSINE metric.
        Skips creation if collection already exists.

        Uses the simple MilvusClient API — extra scalar fields are stored
        via the dynamic field feature (schema-free inserts).
        """
        client = self._get_client()

        existing = client.list_collections()
        if name in existing:
            logger.debug(f"Collection '{name}' already exists — skipping creation")
            return

        logger.info(f"Creating collection '{name}' (dim={dim}, metric=COSINE)...")

        client.create_collection(
            collection_name=name,
            dimension=dim,
            metric_type="COSINE",
            auto_id=False,
            primary_field_name="id",
            vector_field_name="vector",
        )

        logger.success(f"Collection '{name}' created (dim={dim})")

    # ── INSERT METHODS ─────────────────────────────────────────────────────

    def insert_text_chunks(
        self,
        chunks: List[TextChunk],
        batch_size: int = 32,
    ) -> int:
        """
        Compute BGE-M3 embeddings for TextChunks and insert into text_chunks collection.

        Args:
            chunks:     List of TextChunk objects (embedding may be None)
            batch_size: How many chunks to embed and insert at once

        Returns:
            Number of chunks successfully inserted.
        """
        if not chunks:
            return 0

        self._ensure_initialized()
        client = self._get_client()
        inserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]

            # Compute BGE-M3 embeddings for chunks that need them
            texts_to_embed = [c.text for c in batch if c.embedding is None]
            embed_indices = [j for j, c in enumerate(batch) if c.embedding is None]

            if texts_to_embed:
                embeddings = _compute_bge_m3_batch(texts_to_embed)
                for idx, emb in zip(embed_indices, embeddings):
                    batch[idx].embedding = emb

            # Build insert data
            data = []
            for chunk in batch:
                if chunk.embedding is None:
                    logger.warning(
                        f"Skipping text chunk {chunk.chunk_id[:8]}: "
                        "embedding computation failed"
                    )
                    continue

                if len(chunk.embedding) != BGE_M3_DIM:
                    logger.error(
                        f"Text chunk {chunk.chunk_id[:8]} has wrong embedding dim: "
                        f"{len(chunk.embedding)} (expected {BGE_M3_DIM})"
                    )
                    continue

                text_truncated = chunk.text[:8000] if len(chunk.text) > 8000 else chunk.text
                metadata = chunk.to_dict()
                metadata_json = json.dumps(metadata, default=str)[:4000]

                data.append({
                    "id":             chunk.chunk_id,
                    "vector":         chunk.embedding,
                    "text":           text_truncated,
                    "source_file":    chunk.source_file,
                    "page_number":    chunk.page_number or 1,
                    "session_id":     chunk.session_id or "",
                    "modality":       chunk.modality.value,
                    "metadata_json":  metadata_json,
                })

            if data:
                try:
                    client.insert(
                        collection_name=COLLECTION_TEXT,
                        data=data,
                    )
                    batch_inserted = len(data)
                    inserted += batch_inserted
                    logger.debug(
                        f"Inserted {batch_inserted} text chunks "
                        f"(total: {inserted}/{len(chunks)})"
                    )
                except Exception as e:
                    logger.error(f"Milvus insert failed for text batch: {e}")

        logger.info(f"Text chunk insertion complete: {inserted}/{len(chunks)} inserted")
        return inserted

    def insert_image_chunks(
        self,
        chunks: List[ImageChunk],
        batch_size: int = 32,
    ) -> int:
        """
        Insert ImageChunks into image_chunks collection.
        CLIP embeddings are already computed in Phase 2 — no embedding here.

        Args:
            chunks:     List of ImageChunk objects (embedding already set by Phase 2)
            batch_size: Insert batch size

        Returns:
            Number of chunks successfully inserted.
        """
        if not chunks:
            return 0

        self._ensure_initialized()
        client = self._get_client()
        inserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            data = []

            for chunk in batch:
                if chunk.embedding is None:
                    logger.warning(
                        f"Skipping image chunk {chunk.chunk_id[:8]}: "
                        "CLIP embedding is None (was Phase 2 CLIP load successful?)"
                    )
                    continue

                if len(chunk.embedding) != CLIP_DIM:
                    logger.error(
                        f"Image chunk {chunk.chunk_id[:8]} wrong embedding dim: "
                        f"{len(chunk.embedding)} (expected {CLIP_DIM})"
                    )
                    continue

                metadata = chunk.to_dict()
                metadata_json = json.dumps(metadata, default=str)[:4000]
                ocr_text = (chunk.ocr_text or "")[:4000]

                data.append({
                    "id":              chunk.chunk_id,
                    "vector":          chunk.embedding,
                    "ocr_text":        ocr_text,
                    "source_file":     chunk.source_file,
                    "page_number":     chunk.page_number or 1,
                    "session_id":      chunk.session_id or "",
                    "image_path":      chunk.image_path or "",
                    "thumbnail_path":  chunk.thumbnail_path or "",
                    "modality":        chunk.modality.value,
                    "metadata_json":   metadata_json,
                })

            if data:
                try:
                    client.insert(
                        collection_name=COLLECTION_IMAGE,
                        data=data,
                    )
                    inserted += len(data)
                    logger.debug(f"Inserted {len(data)} image chunks")
                except Exception as e:
                    logger.error(f"Milvus insert failed for image batch: {e}")

        logger.info(f"Image chunk insertion complete: {inserted}/{len(chunks)} inserted")
        return inserted

    def insert_audio_chunks(
        self,
        chunks: List[AudioChunk],
        batch_size: int = 32,
    ) -> int:
        """
        Compute BGE-M3 embeddings for AudioChunks and insert into audio_chunks collection.
        AudioChunk.embedding is None after Phase 3 — computed here.

        Args:
            chunks:     List of AudioChunk objects (embedding is None from Phase 3)
            batch_size: How many chunks to embed and insert at once

        Returns:
            Number of chunks successfully inserted.
        """
        if not chunks:
            return 0

        self._ensure_initialized()
        client = self._get_client()
        inserted = 0

        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]

            # Compute BGE-M3 embeddings for chunks without them
            texts_to_embed = [c.text for c in batch if c.embedding is None]
            embed_indices = [j for j, c in enumerate(batch) if c.embedding is None]

            if texts_to_embed:
                embeddings = _compute_bge_m3_batch(texts_to_embed)
                for idx, emb in zip(embed_indices, embeddings):
                    batch[idx].embedding = emb

            data = []
            for chunk in batch:
                if chunk.embedding is None:
                    logger.warning(
                        f"Skipping audio chunk {chunk.chunk_id[:8]}: "
                        "embedding computation failed"
                    )
                    continue

                if len(chunk.embedding) != BGE_M3_DIM:
                    logger.error(
                        f"Audio chunk wrong dim: "
                        f"{len(chunk.embedding)} (expected {BGE_M3_DIM})"
                    )
                    continue

                text_truncated = chunk.text[:8000] if len(chunk.text) > 8000 else chunk.text
                metadata = chunk.to_dict()
                metadata_json = json.dumps(metadata, default=str)[:4000]

                data.append({
                    "id":             chunk.chunk_id,
                    "vector":         chunk.embedding,
                    "text":           text_truncated,
                    "source_file":    chunk.audio_file,
                    "page_number":    1,
                    "session_id":     chunk.session_id or "",
                    "modality":       chunk.modality.value,
                    "metadata_json":  metadata_json,
                })

            if data:
                try:
                    client.insert(
                        collection_name=COLLECTION_AUDIO,
                        data=data,
                    )
                    inserted += len(data)
                    logger.debug(f"Inserted {len(data)} audio chunks")
                except Exception as e:
                    logger.error(f"Milvus insert failed for audio batch: {e}")

        logger.info(f"Audio chunk insertion complete: {inserted}/{len(chunks)} inserted")
        return inserted

    # ── SEARCH METHODS ─────────────────────────────────────────────────────

    def _search(
        self,
        collection_name: str,
        query_vector: List[float],
        top_k: int,
        filter_expr: Optional[str] = None,
        output_fields: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Execute ANN vector search on a collection.

        Args:
            collection_name: Which collection to search
            query_vector:    Query embedding vector
            top_k:           Number of results to return
            filter_expr:     Optional Milvus filter expression
                             e.g. 'session_id == "abc123"'
            output_fields:   Fields to return in results

        Returns:
            List of result dicts with keys: id, distance, and output_fields
        """
        client = self._get_client()

        if output_fields is None:
            output_fields = ["metadata_json", "source_file",
                             "page_number", "session_id", "modality"]

        try:
            search_params = {
                "metric_type": "COSINE",
                "params": {"nprobe": MILVUS_NPROBE},
            }

            results = client.search(
                collection_name=collection_name,
                data=[query_vector],
                limit=top_k,
                search_params=search_params,
                filter=filter_expr if filter_expr else "",
                output_fields=output_fields,
            )

            hits = []
            for hit in results[0]:   # results[0] = hits for first query vector
                hit_dict = {
                    "chunk_id": hit["id"],
                    "score":    hit["distance"],  # COSINE similarity in [0, 1]
                }
                # Merge entity fields into result dict
                if "entity" in hit:
                    hit_dict.update(hit["entity"])
                hits.append(hit_dict)

            return hits

        except Exception as e:
            logger.error(f"Milvus search failed on '{collection_name}': {e}")
            return []

    def search_text(
        self,
        query_embedding: List[float],
        top_k: int = MILVUS_TOP_K,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search text_chunks collection using BGE-M3 query embedding.

        Args:
            query_embedding: 1024-dim BGE-M3 embedding of the query
            top_k:           Number of results to return
            session_id:      If provided, filter to this session only

        Returns:
            List of result dicts ordered by COSINE similarity (descending)
        """
        if len(query_embedding) != BGE_M3_DIM:
            logger.error(
                f"search_text: wrong query dim {len(query_embedding)} "
                f"(expected {BGE_M3_DIM})"
            )
            return []

        filter_expr = None
        if session_id:
            filter_expr = f'session_id == "{session_id}"'

        return self._search(
            collection_name=COLLECTION_TEXT,
            query_vector=query_embedding,
            top_k=top_k,
            filter_expr=filter_expr,
        )

    def search_images(
        self,
        query_embedding: List[float],
        top_k: int = MILVUS_TOP_K,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search image_chunks collection using CLIP query embedding.
        query_embedding must be 512-dim CLIP.

        Args:
            query_embedding: 512-dim CLIP embedding (text or image query)
            top_k:           Number of results to return
            session_id:      If provided, filter to this session only

        Returns:
            List of result dicts ordered by COSINE similarity (descending)
        """
        if len(query_embedding) != CLIP_DIM:
            logger.error(
                f"search_images: wrong query dim {len(query_embedding)} "
                f"(expected {CLIP_DIM})"
            )
            return []

        filter_expr = None
        if session_id:
            filter_expr = f'session_id == "{session_id}"'

        return self._search(
            collection_name=COLLECTION_IMAGE,
            query_vector=query_embedding,
            top_k=top_k,
            filter_expr=filter_expr,
            output_fields=[
                "metadata_json", "source_file",
                "page_number", "session_id", "image_path",
                "thumbnail_path", "modality", "ocr_text",
            ],
        )

    def search_audio(
        self,
        query_embedding: List[float],
        top_k: int = MILVUS_TOP_K,
        session_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Search audio_chunks collection using BGE-M3 query embedding.

        Args:
            query_embedding: 1024-dim BGE-M3 embedding of the query
            top_k:           Number of results to return
            session_id:      If provided, filter to this session only

        Returns:
            List of result dicts ordered by COSINE similarity (descending)
        """
        if len(query_embedding) != BGE_M3_DIM:
            logger.error(
                f"search_audio: wrong query dim {len(query_embedding)} "
                f"(expected {BGE_M3_DIM})"
            )
            return []

        filter_expr = None
        if session_id:
            filter_expr = f'session_id == "{session_id}"'

        return self._search(
            collection_name=COLLECTION_AUDIO,
            query_vector=query_embedding,
            top_k=top_k,
            filter_expr=filter_expr,
        )

    # ── DELETE METHODS ─────────────────────────────────────────────────────

    def delete_by_session(self, session_id: str) -> Dict[str, int]:
        """
        Delete all chunks belonging to a session from all collections.
        Used when a session is cleared or reset.

        Returns:
            Dict with counts: {"text": N, "image": N, "audio": N}
        """
        client = self._get_client()
        counts = {"text": 0, "image": 0, "audio": 0}
        filter_expr = f'session_id == "{session_id}"'

        for collection_name, key in [
            (COLLECTION_TEXT,  "text"),
            (COLLECTION_IMAGE, "image"),
            (COLLECTION_AUDIO, "audio"),
        ]:
            try:
                result = client.delete(
                    collection_name=collection_name,
                    filter=filter_expr,
                )
                # result may be a dict or int depending on pymilvus version
                if isinstance(result, dict):
                    counts[key] = result.get("delete_count", 0)
                else:
                    counts[key] = int(result) if result else 0
                logger.info(
                    f"Deleted {counts[key]} {key} chunks "
                    f"for session {session_id}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to delete {key} chunks "
                    f"for session {session_id}: {e}"
                )

        return counts

    def delete_by_chunk_id(
        self,
        collection_name: str,
        chunk_id: str,
    ) -> bool:
        """
        Delete a single chunk by ID from a specific collection.

        Args:
            collection_name: COLLECTION_TEXT, COLLECTION_IMAGE, or COLLECTION_AUDIO
            chunk_id:        The chunk UUID to delete

        Returns:
            True if deleted, False if failed
        """
        client = self._get_client()
        try:
            client.delete(
                collection_name=collection_name,
                ids=[chunk_id],
            )
            logger.info(f"Deleted chunk {chunk_id[:8]} from {collection_name}")
            return True
        except Exception as e:
            logger.error(f"Failed to delete chunk {chunk_id}: {e}")
            return False

    # ── UTILITY METHODS ────────────────────────────────────────────────────

    def get_collection_stats(self) -> Dict[str, Any]:
        """
        Return row counts and status for all three collections.
        Used by the /health endpoint in Phase 9.
        """
        client = self._get_client()
        stats = {}

        for name in [COLLECTION_TEXT, COLLECTION_IMAGE, COLLECTION_AUDIO]:
            try:
                info = client.get_collection_stats(collection_name=name)
                stats[name] = {
                    "row_count": info.get("row_count", 0),
                    "status": "ok",
                }
            except Exception as e:
                stats[name] = {"row_count": 0, "status": f"error: {e}"}

        return stats

    def compute_query_embedding(self, query_text: str) -> Optional[List[float]]:
        """
        Compute BGE-M3 embedding for a user query.
        Public accessor used by the retrieval engine (Phase 6).

        Args:
            query_text: Natural language query string

        Returns:
            1024-dim BGE-M3 embedding or None on failure
        """
        return _compute_bge_m3_embedding(query_text)

    def release_models(self):
        """
        Explicitly unload BGE-M3 and release VRAM.
        Call after finishing all embedding/indexing work,
        before loading the LLM.
        """
        _unload_bge_m3()

    def _ensure_initialized(self):
        """Auto-initialize if not already done."""
        if not self._initialized:
            self.initialize()

    def close(self):
        """Close the Milvus client connection."""
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
            self._client = None
            self._initialized = False
            logger.debug("Milvus client closed")
