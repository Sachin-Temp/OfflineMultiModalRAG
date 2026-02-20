"""
modules/indexing/tantivy_index.py

Manages the Tantivy BM25F sparse keyword index.

Why Tantivy instead of rank_bm25:
    rank_bm25 (BM25Okapi) requires ALL documents at init time.
    Any new upload triggers a full rebuild (30-60s blocking).
    Tantivy (Rust-based) supports true incremental add via writer.commit()
    — milliseconds per batch, persists to disk, survives restarts.

Single index covers ALL modalities:
    - Text chunks (chunk text)
    - Image chunks (OCR-extracted text)
    - Audio chunks (Whisper transcript text)
    Differentiated by the 'modality' field for filtered queries.

Index schema fields:
    chunk_id    (stored, not indexed) — UUID
    text        (stored, indexed with BM25F) — searchable content
    modality    (stored, indexed) — text | image | audio
    source_file (stored, indexed) — for filename-based queries
    session_id  (stored, indexed) — for session scoping
"""

from pathlib import Path
from typing import List, Optional, Dict, Any

from loguru import logger

from config.settings import TANTIVY_DIR
from models.schemas import TextChunk, ImageChunk, AudioChunk


# ── Schema & Index Initialization ─────────────────────────────────────────
def _build_schema():
    """
    Build the Tantivy schema for the RAG index.
    All fields are stored so we can retrieve them in search results.
    text and source_file are indexed with BM25F for keyword search.
    modality and session_id are indexed for filtered queries.
    """
    import tantivy
    builder = tantivy.SchemaBuilder()

    # chunk_id: stored only (not searchable — used as reference key)
    builder.add_text_field(
        "chunk_id",
        stored=True,
    )

    # text: the main searchable field — BM25F with English stemming
    builder.add_text_field(
        "text",
        stored=True,
        tokenizer_name="en_stem",
    )

    # source_file: searchable so users can query by filename
    builder.add_text_field(
        "source_file",
        stored=True,
        tokenizer_name="en_stem",
    )

    # modality: stored and indexed for boolean filter queries
    builder.add_text_field(
        "modality",
        stored=True,
        tokenizer_name="raw",    # exact match — no stemming for enum fields
    )

    # session_id: stored and indexed for session scoping
    builder.add_text_field(
        "session_id",
        stored=True,
        tokenizer_name="raw",    # exact match — session IDs are opaque strings
    )

    return builder.build()


class TantivyIndex:
    """
    Manages the Tantivy BM25F keyword index for all modalities.

    Design decisions:
    - Single index for all modalities (filtered by 'modality' field)
    - Incremental add via writer.commit() — no full rebuild on new docs
    - Persisted on disk at TANTIVY_DIR — survives restarts
    - 'en_stem' tokenizer for text/source_file (English stemming)
    - 'raw' tokenizer for modality/session_id (exact match)

    Usage:
        idx = TantivyIndex()
        idx.initialize()

        # Add chunks
        idx.add_text_chunks(text_chunks)
        idx.add_image_chunks(image_chunks)   # indexes OCR text
        idx.add_audio_chunks(audio_chunks)   # indexes transcript text

        # Search
        results = idx.search("revenue growth", top_k=50)
        results = idx.search("budget", top_k=50, session_id="abc")
        results = idx.search("chart", top_k=50, modality="image")
    """

    def __init__(self, index_path: Optional[str] = None):
        """
        Initialize the TantivyIndex.

        Args:
            index_path: Optional explicit path for the Tantivy index directory.
                        Defaults to TANTIVY_DIR from settings.
        """
        self._index = None
        self._schema = None
        self._index_path = index_path or str(TANTIVY_DIR.resolve())

    def initialize(self):
        """
        Create or open the Tantivy index at the configured path.
        Idempotent — if index already exists on disk, opens it.
        If it doesn't exist, creates a new one.
        """
        import tantivy

        Path(self._index_path).mkdir(parents=True, exist_ok=True)

        self._schema = _build_schema()

        try:
            # Try to open existing index first
            self._index = tantivy.Index(self._schema, path=self._index_path)
            logger.info(f"Tantivy index opened at: {self._index_path}")
        except Exception as e:
            logger.warning(f"Could not open existing index ({e}), creating new one")
            try:
                self._index = tantivy.Index(
                    self._schema,
                    path=self._index_path,
                    reuse=False,
                )
                logger.success(f"Tantivy index created at: {self._index_path}")
            except Exception as e2:
                raise RuntimeError(
                    f"Failed to create Tantivy index at {self._index_path}: {e2}"
                ) from e2

    def _ensure_initialized(self):
        """Auto-initialize if not already done."""
        if self._index is None:
            self.initialize()

    def _add_documents(self, documents: List[Dict[str, str]]):
        """
        Add a batch of documents to the index incrementally.
        Uses writer.commit() ONCE per batch — atomic, milliseconds, no rebuild.

        Args:
            documents: List of dicts with keys:
                chunk_id, text, source_file, modality, session_id
        """
        import tantivy

        if not documents:
            return

        # heap_size=50MB for the writer buffer
        writer = self._index.writer(heap_size=50_000_000)

        added = 0
        for doc_dict in documents:
            try:
                doc = tantivy.Document(
                    chunk_id=str(doc_dict.get("chunk_id", "")),
                    text=str(doc_dict.get("text", ""))[:8000],
                    source_file=str(doc_dict.get("source_file", "")),
                    modality=str(doc_dict.get("modality", "")),
                    session_id=str(doc_dict.get("session_id", "")),
                )
                writer.add_document(doc)
                added += 1
            except Exception as e:
                logger.warning(f"Failed to add document to Tantivy: {e}")
                continue

        try:
            writer.commit()   # atomic commit — appends segments, no rebuild
            logger.debug(f"Tantivy: committed {added} documents")
        except Exception as e:
            logger.error(f"Tantivy writer.commit() failed: {e}")

    def add_text_chunks(self, chunks: List[TextChunk]):
        """
        Add TextChunks to the Tantivy index.
        The chunk.text field is indexed for BM25F search.

        Args:
            chunks: List of TextChunk objects from Phase 1
        """
        self._ensure_initialized()

        documents = []
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            documents.append({
                "chunk_id":    chunk.chunk_id,
                "text":        chunk.text,
                "source_file": chunk.source_file,
                "modality":    chunk.modality.value,
                "session_id":  chunk.session_id or "",
            })

        self._add_documents(documents)
        logger.info(f"Tantivy: added {len(documents)} text chunks")

    def add_image_chunks(self, chunks: List[ImageChunk]):
        """
        Add ImageChunks to the Tantivy index using their OCR text.
        Images with no OCR text are still indexed with empty text
        (they'll only be found via CLIP semantic search in Milvus).

        Args:
            chunks: List of ImageChunk objects from Phase 2
        """
        self._ensure_initialized()

        documents = []
        for chunk in chunks:
            # Use OCR text for keyword indexing
            text_parts = []
            if chunk.ocr_text:
                text_parts.append(chunk.ocr_text)
            if chunk.llm_description:
                text_parts.append(chunk.llm_description)
            indexed_text = " ".join(text_parts).strip()

            # Index even if empty — chunk_id is needed for cross-modal linking
            documents.append({
                "chunk_id":    chunk.chunk_id,
                "text":        indexed_text,
                "source_file": chunk.source_file,
                "modality":    chunk.modality.value,
                "session_id":  chunk.session_id or "",
            })

        self._add_documents(documents)
        logger.info(
            f"Tantivy: added {len(documents)} image chunks "
            f"({sum(1 for c in chunks if c.ocr_text)} with OCR text)"
        )

    def add_audio_chunks(self, chunks: List[AudioChunk]):
        """
        Add AudioChunks to the Tantivy index using their transcript text.

        Args:
            chunks: List of AudioChunk objects from Phase 3
        """
        self._ensure_initialized()

        documents = []
        for chunk in chunks:
            if not chunk.text.strip():
                continue
            documents.append({
                "chunk_id":    chunk.chunk_id,
                "text":        chunk.text,
                "source_file": chunk.audio_file,
                "modality":    chunk.modality.value,
                "session_id":  chunk.session_id or "",
            })

        self._add_documents(documents)
        logger.info(f"Tantivy: added {len(documents)} audio chunks")

    def search(
        self,
        query: str,
        top_k: int = 50,
        session_id: Optional[str] = None,
        modality: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        BM25F keyword search across all indexed chunks.

        Args:
            query:      Natural language query string
            top_k:      Number of results to return
            session_id: If provided, filter to this session only
            modality:   If provided, filter to this modality only
                        ('text', 'image', or 'audio')

        Returns:
            List of result dicts ordered by BM25F score (descending):
            [{"chunk_id": str, "score": float, "text": str,
              "source_file": str, "modality": str, "session_id": str}, ...]
        """
        self._ensure_initialized()

        if not query.strip():
            return []

        try:
            searcher = self._index.searcher()

            # Build the query — search 'text' and 'source_file' fields
            try:
                base_query = self._index.parse_query(
                    query,
                    ["text", "source_file"],
                )
            except Exception as e:
                logger.warning(f"Tantivy query parse failed for '{query}': {e}")
                # Try simpler query with just text field
                try:
                    base_query = self._index.parse_query(query, ["text"])
                except Exception as e2:
                    logger.error(f"Tantivy query completely failed: {e2}")
                    return []

            # Execute search — fetch extra results to allow for post-filtering
            fetch_count = top_k * 3 if (session_id or modality) else top_k
            search_result = searcher.search(base_query, fetch_count)
            hits = search_result.hits

            results = []
            for score, doc_address in hits:
                try:
                    doc = searcher.doc(doc_address)

                    # Extract fields from document
                    chunk_id     = _get_field(doc, "chunk_id")
                    text         = _get_field(doc, "text")
                    source_file  = _get_field(doc, "source_file")
                    doc_modality = _get_field(doc, "modality")
                    doc_session  = _get_field(doc, "session_id")

                    # Apply session_id filter (post-search)
                    if session_id and doc_session != session_id:
                        continue

                    # Apply modality filter (post-search)
                    if modality and doc_modality != modality:
                        continue

                    results.append({
                        "chunk_id":    chunk_id,
                        "score":       float(score),
                        "text":        text,
                        "source_file": source_file,
                        "modality":    doc_modality,
                        "session_id":  doc_session,
                    })

                    if len(results) >= top_k:
                        break

                except Exception as e:
                    logger.debug(f"Failed to parse Tantivy hit: {e}")
                    continue

            logger.debug(
                f"Tantivy search '{query[:50]}': "
                f"{len(results)} results (top_k={top_k})"
            )
            return results

        except Exception as e:
            logger.error(f"Tantivy search failed: {e}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        """
        Return index statistics.
        Used by /health endpoint in Phase 9.
        """
        self._ensure_initialized()
        try:
            searcher = self._index.searcher()
            return {
                "num_docs":   searcher.num_docs,
                "index_path": str(self._index_path),
                "status":     "ok",
            }
        except Exception as e:
            return {"num_docs": 0, "status": f"error: {e}"}

    def reload(self):
        """
        Reload the index reader to see the latest committed documents.
        Call this if documents were added and you need to see them
        in search results immediately.
        """
        self._ensure_initialized()
        try:
            self._index.reload()
            logger.debug("Tantivy index reloaded")
        except Exception as e:
            logger.warning(f"Tantivy reload failed: {e}")


def _get_field(doc, field_name: str) -> str:
    """
    Safely extract a string field from a Tantivy Document object.
    Returns empty string if field is missing or has no values.
    """
    try:
        values = doc.get_all(field_name)
        if values:
            return str(values[0])
        return ""
    except Exception:
        return ""
