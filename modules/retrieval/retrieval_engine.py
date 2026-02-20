"""
modules/retrieval/retrieval_engine.py

Complete hybrid retrieval engine for the Multimodal RAG system.

Pipeline:
  1. Query understanding & routing (which modalities to search)
  2. Query encoding (BGE-M3 for text/audio, CLIP for images)
  3. Parallel Milvus ANN + Tantivy BM25F retrieval
  4. Score normalization (BM25F → [0,1])
  5. RRF fusion → top-100 candidates
  6. Modality diversification (minimum per-modality representation)
  7. Cross-encoder reranking (BGE-Reranker-v2-m3) → top-5 gold chunks
  8. Cross-modal link enrichment (up to 2 linked chunks added)
  9. CO_RETRIEVED link update in SQLite

Imports from:
  - Phase 4: MilvusStore, TantivyIndex
  - Phase 5: SQLiteStore
  - Phase 2: _compute_clip_text_embedding (for CLIP text encoding)
  - core: VRAMManager
  - config: all retrieval constants
"""

import json
import math
import re
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import (
    Any, Dict, List, Optional, Set, Tuple,
)

from loguru import logger

from config.settings import (
    BGE_M3_DIM,
    CLIP_DIM,
    BGE_RERANKER_MODEL,
    MILVUS_TOP_K,
    BM25_TOP_K,
    RRF_K,
    RERANKER_THRESHOLD,
    FINAL_TOP_K,
    CROSS_MODAL_LINK_STRENGTH_THRESHOLD,
)
from core.vram_manager import vram_manager
from modules.indexing.milvus_store import MilvusStore
from modules.indexing.tantivy_index import TantivyIndex
from modules.indexing.sqlite_store import SQLiteStore


# ── Reranker Model Management ──────────────────────────────────────────────
_reranker_model = None


def _load_reranker():
    """
    Load BGE-Reranker-v2-m3 cross-encoder.
    Runs on CPU — reranker shares CPU with Tantivy and SQLite operations.
    Returns the loaded CrossEncoder model.
    """
    global _reranker_model

    if _reranker_model is not None:
        logger.debug("Reranker already loaded — reusing")
        return _reranker_model

    def _evict_reranker():
        global _reranker_model
        import torch
        if _reranker_model is not None:
            del _reranker_model
            _reranker_model = None
            torch.cuda.empty_cache()
            logger.info("Reranker evicted from VRAM by VRAMManager")

    vram_manager.register_evict_callback("reranker", _evict_reranker)

    acquired = vram_manager.acquire("reranker")
    if not acquired:
        logger.warning(
            "VRAMManager could not allocate for reranker — "
            "will run reranker on CPU without VRAM tracking"
        )

    try:
        from sentence_transformers import CrossEncoder
        # Reranker runs on CPU to leave GPU for LLM
        device = "cpu"
        logger.info(f"Loading BGE-Reranker-v2-m3 on {device}...")
        model = CrossEncoder(
            BGE_RERANKER_MODEL,
            device=device,
            max_length=512,
        )
        _reranker_model = model
        logger.success("BGE-Reranker-v2-m3 loaded on CPU")
        return model
    except Exception as e:
        vram_manager.release("reranker")
        raise RuntimeError(f"Failed to load reranker: {e}") from e


def _unload_reranker():
    """Explicitly unload reranker and release VRAM."""
    global _reranker_model
    import torch
    if _reranker_model is not None:
        del _reranker_model
        _reranker_model = None
        torch.cuda.empty_cache()
        vram_manager.release("reranker")
        logger.info("Reranker explicitly unloaded")


# ── Data Structures ────────────────────────────────────────────────────────
@dataclass
class RawCandidate:
    """
    A single retrieval candidate from Milvus or Tantivy.
    Carries enough information for RRF fusion and reranking.
    """
    chunk_id:       str
    score:          float            # raw score from source system
    source:         str              # "milvus_text" | "milvus_image" | "milvus_audio" | "bm25"
    modality:       str              # "text" | "image" | "audio"
    text:           str              # text content for reranker
    source_file:    str = ""
    page_number:    int = 1
    session_id:     str = ""
    metadata_json:  str = ""         # full metadata from Milvus
    image_path:     str = ""         # for image chunks
    thumbnail_path: str = ""         # for image chunks
    start_time:     float = 0.0      # for audio chunks
    end_time:       float = 0.0      # for audio chunks


@dataclass
class GoldChunk:
    """
    A final retrieval result after reranking and enrichment.
    This is what the LLM receives as context (Phase 7).
    """
    chunk_id:         str
    modality:         str
    text:             str
    source_file:      str
    page_number:      int
    reranker_score:   float
    rrf_score:        float
    session_id:       str      = ""
    metadata_json:    str      = ""
    image_path:       str      = ""
    thumbnail_path:   str      = ""
    start_time:       float    = 0.0
    end_time:         float    = 0.0
    timestamp_display: str     = ""
    linked_chunk_ids: List[str] = field(default_factory=list)
    link_types:       List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "chunk_id":          self.chunk_id,
            "modality":          self.modality,
            "text":              self.text,
            "source_file":       self.source_file,
            "page_number":       self.page_number,
            "reranker_score":    round(self.reranker_score, 4),
            "rrf_score":         round(self.rrf_score, 6),
            "session_id":        self.session_id,
            "image_path":        self.image_path,
            "thumbnail_path":    self.thumbnail_path,
            "start_time":        self.start_time,
            "end_time":          self.end_time,
            "timestamp_display": self.timestamp_display,
            "linked_chunk_ids":  self.linked_chunk_ids,
            "link_types":        self.link_types,
        }


@dataclass
class RetrievalResult:
    """
    Complete result from one retrieval query.
    Passed to the LLM engine in Phase 7.
    """
    query:            str
    gold_chunks:      List[GoldChunk] = field(default_factory=list)
    linked_chunks:    List[GoldChunk] = field(default_factory=list)
    query_modality:   str = "text"       # dominant query modality
    routing:          Dict[str, Any] = field(default_factory=dict)
    latency_ms:       Dict[str, float] = field(default_factory=dict)
    total_candidates: int = 0

    @property
    def all_chunks(self) -> List[GoldChunk]:
        """Gold chunks + linked enrichment chunks combined."""
        return self.gold_chunks + self.linked_chunks

    @property
    def all_chunk_ids(self) -> List[str]:
        return [c.chunk_id for c in self.all_chunks]


# ── Step 1: Query Routing ──────────────────────────────────────────────────
# Visual keywords that indicate an image search should be prioritized
_VISUAL_KEYWORDS = {
    "chart", "graph", "diagram", "image", "picture", "photo",
    "screenshot", "figure", "table", "plot", "visualization",
    "drawing", "sketch", "illustration", "map", "infographic",
    "show me", "what does", "look like",
}

# Timestamp pattern: matches "14:32", "1:05:22", "at 14 minutes", "14-minute"
_TIMESTAMP_PATTERN = re.compile(
    r"\b(\d{1,2}:\d{2}(?::\d{2})?|\d{1,3}[\s\-]?(?:minute|second|min|sec)s?\b)",
    re.IGNORECASE,
)

# Filename pattern: matches "report.pdf", "meeting.mp3", etc.
_FILENAME_PATTERN = re.compile(
    r"\b\w[\w\s\-]*\.(pdf|docx|doc|mp3|mp4|wav|m4a|png|jpg|jpeg|webp)\b",
    re.IGNORECASE,
)


def classify_query(
    query: str,
    has_image_attachment: bool = False,
    has_audio_attachment: bool = False,
) -> Dict[str, Any]:
    """
    Analyse a query string and classify it for retrieval routing.

    Returns a routing dict with keys:
        search_text:   bool — search text_chunks collection
        search_images: bool — search image_chunks collection
        search_audio:  bool — search audio_chunks collection
        use_bm25:      bool — use Tantivy BM25F
        force_image:   bool — image search is primary
        force_audio:   bool — audio search is primary
        force_bm25:    bool — BM25 is primary (filename/ID query)
        has_timestamp: bool — query references a specific time
        detected_timestamp: str or None
        detected_filename:  str or None
        query_modality: str — dominant modality

    Strategy:
    - Always search all three Milvus collections + BM25 unless forced
    - "Force" flags set primary modality but don't exclude others
    - Image attachment → force_image=True
    - Audio attachment → force_audio=True
    - Timestamp in query → force_audio=True (timestamp nav query)
    - Filename in query → force_bm25=True (exact match needed)
    - Visual keywords → force_image weighted higher in routing
    """
    query_lower = query.lower()

    # Detect timestamp
    ts_match = _TIMESTAMP_PATTERN.search(query)
    detected_timestamp = ts_match.group(0) if ts_match else None
    has_timestamp = detected_timestamp is not None

    # Detect filename
    fn_match = _FILENAME_PATTERN.search(query)
    detected_filename = fn_match.group(0) if fn_match else None

    # Detect visual keywords
    has_visual = any(kw in query_lower for kw in _VISUAL_KEYWORDS)

    # Build routing
    routing: Dict[str, Any] = {
        "search_text":   True,
        "search_images": True,
        "search_audio":  True,
        "use_bm25":      True,
        "force_image":   False,
        "force_audio":   False,
        "force_bm25":    False,
        "has_timestamp":       has_timestamp,
        "detected_timestamp":  detected_timestamp,
        "detected_filename":   detected_filename,
        "query_modality":      "text",
    }

    # Apply force flags (priority: attachment > timestamp > filename > visual)
    if has_image_attachment:
        routing["force_image"]    = True
        routing["query_modality"] = "image"

    elif has_audio_attachment:
        routing["force_audio"]    = True
        routing["query_modality"] = "audio"

    elif has_timestamp:
        routing["force_audio"]    = True
        routing["query_modality"] = "audio"

    elif detected_filename:
        routing["force_bm25"]     = True
        routing["query_modality"] = "text"

    elif has_visual:
        routing["force_image"]    = True
        routing["query_modality"] = "image"

    logger.debug(
        f"Query routing: modality={routing['query_modality']} | "
        f"force_image={routing['force_image']} | "
        f"force_audio={routing['force_audio']} | "
        f"force_bm25={routing['force_bm25']} | "
        f"timestamp={detected_timestamp} | "
        f"filename={detected_filename}"
    )
    return routing


# ── Step 2: Query Encoding ─────────────────────────────────────────────────
def encode_query(
    query: str,
    milvus_store: MilvusStore,
    image_bytes: Optional[bytes] = None,
) -> Dict[str, Optional[List[float]]]:
    """
    Encode a query into all required embedding spaces.

    Returns dict with keys:
        bge_m3:  1024-dim BGE-M3 embedding (for text + audio search)
        clip:    512-dim CLIP embedding (for image search)
                 — from text query if no image attached
                 — from uploaded image if image_bytes provided

    Both embeddings are always computed so all three collections
    can be searched regardless of query type.
    """
    embeddings: Dict[str, Optional[List[float]]] = {
        "bge_m3": None,
        "clip":   None,
    }

    # Always compute BGE-M3 for text and audio collection search
    try:
        bge_embedding = milvus_store.compute_query_embedding(query)
        embeddings["bge_m3"] = bge_embedding
        if bge_embedding:
            logger.debug(f"BGE-M3 query encoded: {len(bge_embedding)}-dim")
    except Exception as e:
        logger.error(f"BGE-M3 query encoding failed: {e}")

    # Compute CLIP embedding for image collection search
    try:
        if image_bytes is not None:
            # Image uploaded as query — encode the image itself
            from PIL import Image
            import io
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            from modules.ingestion.image_ingestor import _compute_clip_embedding
            clip_embedding = _compute_clip_embedding(img)
            logger.debug("CLIP query encoded from uploaded image")
        else:
            # Text query — use CLIP text encoder for image search
            from modules.ingestion.image_ingestor import _compute_clip_text_embedding
            clip_embedding = _compute_clip_text_embedding(query)
            if clip_embedding:
                logger.debug(f"CLIP text query encoded: {len(clip_embedding)}-dim")

        embeddings["clip"] = clip_embedding
    except Exception as e:
        logger.error(f"CLIP query encoding failed: {e}")

    return embeddings


# ── Step 3: Parallel Retrieval ─────────────────────────────────────────────
def _parse_milvus_result(
    hit: Dict[str, Any],
    modality: str,
    source_label: str,
) -> Optional[RawCandidate]:
    """
    Convert a raw Milvus search hit dict into a RawCandidate.
    Extracts text content appropriate for the modality.
    Returns None if hit is malformed.
    """
    try:
        chunk_id = hit.get("chunk_id", "")
        if not chunk_id:
            return None

        score = float(hit.get("score", 0.0))

        # Extract text for reranker
        if modality == "image":
            text = hit.get("ocr_text", "") or ""
        else:
            text = hit.get("text", "") or ""

        # Parse additional fields
        image_path     = hit.get("image_path",     "") or ""
        thumbnail_path = hit.get("thumbnail_path", "") or ""

        # Parse audio timestamps from metadata_json if available
        start_time = 0.0
        end_time   = 0.0
        metadata_json = hit.get("metadata_json", "{}") or "{}"
        try:
            metadata = json.loads(metadata_json)
            start_time = float(metadata.get("start_time", 0.0))
            end_time   = float(metadata.get("end_time",   0.0))
        except Exception:
            pass

        return RawCandidate(
            chunk_id=chunk_id,
            score=score,
            source=source_label,
            modality=modality,
            text=text,
            source_file=hit.get("source_file", "") or "",
            page_number=int(hit.get("page_number", 1) or 1),
            session_id=hit.get("session_id", "") or "",
            metadata_json=metadata_json,
            image_path=image_path,
            thumbnail_path=thumbnail_path,
            start_time=start_time,
            end_time=end_time,
        )
    except Exception as e:
        logger.debug(f"Failed to parse Milvus hit: {e}")
        return None


def _parse_bm25_result(hit: Dict[str, Any]) -> Optional[RawCandidate]:
    """Convert a Tantivy BM25 hit dict into a RawCandidate."""
    try:
        chunk_id = hit.get("chunk_id", "")
        if not chunk_id:
            return None

        return RawCandidate(
            chunk_id=chunk_id,
            score=float(hit.get("score", 0.0)),
            source="bm25",
            modality=hit.get("modality", "text"),
            text=hit.get("text", ""),
            source_file=hit.get("source_file", ""),
            session_id=hit.get("session_id", ""),
        )
    except Exception as e:
        logger.debug(f"Failed to parse BM25 hit: {e}")
        return None


def run_parallel_retrieval(
    query: str,
    embeddings: Dict[str, Optional[List[float]]],
    routing: Dict[str, Any],
    milvus_store: MilvusStore,
    tantivy_index: TantivyIndex,
    session_id: Optional[str] = None,
    top_k: int = MILVUS_TOP_K,
) -> Tuple[List[RawCandidate], Dict[str, float]]:
    """
    Execute Milvus ANN search and Tantivy BM25F search in parallel
    using ThreadPoolExecutor.

    Both search systems are read-only so there is no locking conflict.
    ThreadPoolExecutor is used instead of asyncio because both Milvus
    and Tantivy clients are synchronous.

    Args:
        query:         Raw query string (for BM25)
        embeddings:    Dict with 'bge_m3' and 'clip' embeddings
        routing:       Query routing dict from classify_query()
        milvus_store:  Initialized MilvusStore instance
        tantivy_index: Initialized TantivyIndex instance
        session_id:    Optional session filter
        top_k:         Results per collection

    Returns:
        (list of RawCandidates, dict of latencies per source in ms)
    """
    all_candidates: List[RawCandidate] = []
    latencies: Dict[str, float] = {}

    def search_milvus_text():
        if not routing.get("search_text"):
            return []
        emb = embeddings.get("bge_m3")
        if emb is None or len(emb) != BGE_M3_DIM:
            return []
        t0 = time.time()
        hits = milvus_store.search_text(emb, top_k=top_k, session_id=session_id)
        latencies["milvus_text"] = (time.time() - t0) * 1000
        return [
            c for c in (
                _parse_milvus_result(h, "text", "milvus_text") for h in hits
            ) if c is not None
        ]

    def search_milvus_images():
        if not routing.get("search_images"):
            return []
        emb = embeddings.get("clip")
        if emb is None or len(emb) != CLIP_DIM:
            return []
        t0 = time.time()
        hits = milvus_store.search_images(emb, top_k=top_k, session_id=session_id)
        latencies["milvus_image"] = (time.time() - t0) * 1000
        return [
            c for c in (
                _parse_milvus_result(h, "image", "milvus_image") for h in hits
            ) if c is not None
        ]

    def search_milvus_audio():
        if not routing.get("search_audio"):
            return []
        emb = embeddings.get("bge_m3")
        if emb is None or len(emb) != BGE_M3_DIM:
            return []
        t0 = time.time()
        hits = milvus_store.search_audio(emb, top_k=top_k, session_id=session_id)
        latencies["milvus_audio"] = (time.time() - t0) * 1000
        return [
            c for c in (
                _parse_milvus_result(h, "audio", "milvus_audio") for h in hits
            ) if c is not None
        ]

    def search_bm25():
        if not routing.get("use_bm25"):
            return []
        t0 = time.time()
        hits = tantivy_index.search(
            query,
            top_k=BM25_TOP_K,
            session_id=session_id,
        )
        latencies["bm25"] = (time.time() - t0) * 1000
        return [
            c for c in (
                _parse_bm25_result(h) for h in hits
            ) if c is not None
        ]

    # Execute all four searches in parallel
    t_total = time.time()
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(search_milvus_text):   "milvus_text",
            executor.submit(search_milvus_images): "milvus_image",
            executor.submit(search_milvus_audio):  "milvus_audio",
            executor.submit(search_bm25):          "bm25",
        }
        for future in as_completed(futures):
            source = futures[future]
            try:
                results = future.result()
                all_candidates.extend(results)
                logger.debug(f"Retrieval {source}: {len(results)} candidates")
            except Exception as e:
                logger.error(f"Retrieval source {source} failed: {e}")

    latencies["total_retrieval"] = (time.time() - t_total) * 1000
    logger.info(
        f"Parallel retrieval complete: {len(all_candidates)} raw candidates "
        f"in {latencies['total_retrieval']:.0f}ms"
    )
    return all_candidates, latencies


# ── Step 4: Score Normalization ────────────────────────────────────────────
def normalize_bm25_scores(candidates: List[RawCandidate]) -> List[RawCandidate]:
    """
    Normalize Tantivy BM25F scores to [0, 1] range using min-max normalization.
    Milvus COSINE scores are already in [0, 1] and do not need normalization.

    Formula: score_norm = (score - min) / (max - min + ε)

    Args:
        candidates: Mixed list of candidates from all sources

    Returns:
        Same list with BM25 scores normalized to [0,1]
    """
    bm25_candidates = [c for c in candidates if c.source == "bm25"]
    if len(bm25_candidates) < 2:
        return candidates

    scores = [c.score for c in bm25_candidates]
    min_score = min(scores)
    max_score = max(scores)
    score_range = max_score - min_score + 1e-9   # ε prevents division by zero

    for c in candidates:
        if c.source == "bm25":
            c.score = (c.score - min_score) / score_range

    return candidates


# ── Step 5: RRF Fusion ─────────────────────────────────────────────────────
def rrf_score(rank: int, k: int = RRF_K) -> float:
    """
    Compute Reciprocal Rank Fusion score for a single rank position.
    Formula: 1 / (k + rank)
    k=60 is the standard constant from the original RRF paper.
    """
    return 1.0 / (k + rank)


def fuse_results_rrf(
    candidates: List[RawCandidate],
    top_n: int = 100,
) -> List[Tuple[str, float, RawCandidate]]:
    """
    Apply Reciprocal Rank Fusion across all retrieval sources.

    Strategy:
    1. Group candidates by source (milvus_text, milvus_image, milvus_audio, bm25)
    2. Within each source, rank candidates by their raw score descending
    3. For each chunk in each source, add rrf_score(rank) to its cumulative score
    4. Chunks appearing in multiple sources get multiple contributions (key benefit)
    5. Sort by cumulative RRF score descending, return top_n

    Args:
        candidates: All raw candidates from all sources
        top_n:      Number of fused results to return

    Returns:
        List of (chunk_id, rrf_score, best_candidate) tuples,
        ordered by RRF score descending.
        best_candidate is the RawCandidate with highest raw score
        (used to get metadata when a chunk appears in multiple sources).
    """
    if not candidates:
        return []

    # Group by source
    by_source: Dict[str, List[RawCandidate]] = defaultdict(list)
    for c in candidates:
        by_source[c.source].append(c)

    # Sort each source by score descending
    for source in by_source:
        by_source[source].sort(key=lambda x: x.score, reverse=True)

    # Compute RRF scores
    rrf_scores: Dict[str, float] = defaultdict(float)
    # Track best candidate per chunk_id (highest raw score across sources)
    best_candidate: Dict[str, RawCandidate] = {}

    for source, source_candidates in by_source.items():
        for rank, candidate in enumerate(source_candidates):
            cid = candidate.chunk_id
            rrf_scores[cid] += rrf_score(rank, k=RRF_K)
            # Keep best raw score candidate for metadata
            if cid not in best_candidate or candidate.score > best_candidate[cid].score:
                best_candidate[cid] = candidate

    # Sort by RRF score descending
    ranked = sorted(rrf_scores.items(), key=lambda x: -x[1])[:top_n]

    result = []
    for chunk_id, score_val in ranked:
        if chunk_id in best_candidate:
            result.append((chunk_id, score_val, best_candidate[chunk_id]))

    logger.info(
        f"RRF fusion: {len(candidates)} candidates → "
        f"{len(result)} unique chunks (top-{top_n})"
    )
    return result


# ── Step 6: Modality Diversification ──────────────────────────────────────
def diversify_modalities(
    fused: List[Tuple[str, float, RawCandidate]],
    min_text:  int = 20,
    min_image: int = 10,
    min_audio: int = 10,
    all_candidates: Optional[List[RawCandidate]] = None,
) -> List[Tuple[str, float, RawCandidate]]:
    """
    Enforce minimum per-modality representation in the reranker input.

    Problem this solves:
    Text embeddings from BGE-M3 are dense and tend to dominate ANN search.
    Without diversification, the reranker might see 100 text chunks and
    0 image/audio chunks, missing genuine cross-modal matches.

    Strategy:
    1. Count current modality distribution in fused top-100
    2. If any modality is under-represented, pull additional candidates
       from all_candidates to meet the minimum
    3. The added candidates are appended at the end (lower priority)
       so they only displace the weakest existing candidates

    Args:
        fused:         RRF-fused top-100 list
        min_text/image/audio: Minimum chunks per modality
        all_candidates: Full candidate pool to pull from if needed

    Returns:
        Diversified list (may be longer than original if padding was needed)
    """
    if not all_candidates:
        return fused

    # Count current distribution
    present_ids: Set[str] = {item[0] for item in fused}
    modality_counts: Dict[str, int] = defaultdict(int)
    for _, _, cand in fused:
        modality_counts[cand.modality] += 1

    # Build pool of candidates not already in fused, by modality
    extra_pool: Dict[str, List[RawCandidate]] = defaultdict(list)
    for c in all_candidates:
        if c.chunk_id not in present_ids:
            extra_pool[c.modality].append(c)

    # Sort extras by score descending
    for mod in extra_pool:
        extra_pool[mod].sort(key=lambda x: x.score, reverse=True)

    # Pad under-represented modalities
    result = list(fused)
    minimums = {"text": min_text, "image": min_image, "audio": min_audio}

    for modality, minimum in minimums.items():
        current = modality_counts.get(modality, 0)
        if current < minimum:
            needed = minimum - current
            extras = extra_pool.get(modality, [])[:needed]
            for extra in extras:
                # Assign a low RRF score so they rank below fused results
                result.append((extra.chunk_id, 0.001, extra))
                present_ids.add(extra.chunk_id)
                logger.debug(
                    f"Diversity padding: added {modality} chunk "
                    f"{extra.chunk_id[:8]} (had {current}/{minimum})"
                )

    if len(result) > len(fused):
        logger.info(
            f"Modality diversification: {len(fused)} → {len(result)} chunks "
            f"(added padding for under-represented modalities)"
        )

    return result


# ── Step 7: Cross-Encoder Reranking ───────────────────────────────────────
def _build_reranker_input_text(
    query: str,
    candidate: RawCandidate,
) -> str:
    """
    Build the text string to pair with the query for the cross-encoder.

    For text chunks: use chunk text directly
    For image chunks: use OCR text + any available description
    For audio chunks: use transcript text with timestamp prefix

    The reranker scores (query, text) pairs — the better the text
    represents the chunk's content, the more accurate the reranking.
    """
    if candidate.modality == "image":
        parts = []
        if candidate.text:        # OCR text
            parts.append(f"[OCR] {candidate.text}")
        # Try to get llm_description from metadata_json
        try:
            meta = json.loads(candidate.metadata_json or "{}")
            desc = meta.get("llm_description", "")
            if desc:
                parts.append(f"[Description] {desc}")
        except Exception:
            pass
        return " ".join(parts) if parts else "[image]"

    elif candidate.modality == "audio":
        ts = ""
        if candidate.start_time > 0 or candidate.end_time > 0:
            try:
                from modules.ingestion.audio_ingestor import _format_timestamp_range
                ts = f"[{_format_timestamp_range(candidate.start_time, candidate.end_time)}] "
            except Exception:
                pass
        return ts + (candidate.text or "[audio segment]")

    else:
        return candidate.text or "[empty]"


def _sigmoid(x: float) -> float:
    """Apply sigmoid to convert raw logits to [0,1] probabilities."""
    return 1.0 / (1.0 + math.exp(-x))


def rerank_candidates(
    query: str,
    candidates: List[Tuple[str, float, RawCandidate]],
    top_k: int = FINAL_TOP_K,
    score_threshold: float = RERANKER_THRESHOLD,
) -> List[Tuple[RawCandidate, float]]:
    """
    Rerank candidates using BGE-Reranker-v2-m3 cross-encoder.

    The cross-encoder jointly encodes (query, passage) pairs and
    outputs a relevance score. This is much more accurate than
    bi-encoder cosine similarity but too slow for large candidate sets
    — which is why we use it only on the top-100 after RRF fusion.

    Args:
        query:           User query string
        candidates:      RRF-fused candidates: (chunk_id, rrf_score, candidate)
        top_k:           Number of final results to return
        score_threshold: Drop chunks scoring below this threshold

    Returns:
        List of (RawCandidate, reranker_score) tuples,
        ordered by reranker score descending.
        Only includes chunks scoring above score_threshold.
    """
    if not candidates:
        return []

    t0 = time.time()

    try:
        reranker = _load_reranker()
    except Exception as e:
        logger.error(f"Reranker load failed: {e}. Falling back to RRF order.")
        # Fallback: return top_k by RRF score without reranking
        return [
            (cand, rrf_s)
            for _, rrf_s, cand in candidates[:top_k]
        ]

    # Build (query, passage) pairs for the reranker
    pairs: List[List[str]] = []
    candidate_list: List[Tuple[RawCandidate, float]] = []
    for _, rrf_s, cand in candidates:
        passage = _build_reranker_input_text(query, cand)
        pairs.append([query, passage])
        candidate_list.append((cand, rrf_s))

    # Score all pairs in one batch
    try:
        scores = reranker.predict(
            pairs,
            show_progress_bar=len(pairs) > 20,
            batch_size=16,
        )
    except Exception as e:
        logger.error(f"Reranker inference failed: {e}. Falling back to RRF.")
        return [
            (cand, rrf_s)
            for cand, rrf_s in candidate_list[:top_k]
        ]

    # Apply sigmoid to convert raw scores to [0,1] range
    scored = []
    for (cand, rrf_s), raw_score in zip(candidate_list, scores):
        norm_score = _sigmoid(float(raw_score))
        if norm_score >= score_threshold:
            scored.append((cand, norm_score))

    # Sort by reranker score descending
    scored.sort(key=lambda x: -x[1])
    result = scored[:top_k]

    elapsed_ms = (time.time() - t0) * 1000
    logger.info(
        f"Reranking: {len(candidates)} candidates → "
        f"{len(result)} gold chunks "
        f"(threshold={score_threshold}, {elapsed_ms:.0f}ms)"
    )
    return result


# ── Step 8: Cross-Modal Link Enrichment ────────────────────────────────────
def enrich_with_links(
    gold_results: List[Tuple[RawCandidate, float]],
    sqlite_store: SQLiteStore,
    max_linked: int = 2,
    min_strength: float = CROSS_MODAL_LINK_STRENGTH_THRESHOLD,
) -> Tuple[List[GoldChunk], List[GoldChunk]]:
    """
    Enrich gold chunks with strongly linked cross-modal chunks.

    For each gold chunk, query SQLite for linked chunks with
    strength > min_strength. Add up to max_linked linked chunks
    that are NOT already in the gold set.

    Args:
        gold_results:  List of (RawCandidate, reranker_score) from reranker
        sqlite_store:  SQLiteStore for link queries
        max_linked:    Maximum additional linked chunks to add
        min_strength:  Minimum link strength to consider

    Returns:
        (gold_chunks, linked_chunks) as lists of GoldChunk objects
    """
    gold_chunk_ids: Set[str] = {cand.chunk_id for cand, _ in gold_results}
    linked_infos: List[Dict[str, Any]] = []
    added_ids: Set[str] = set(gold_chunk_ids)

    # Convert gold results to GoldChunk objects
    gold_chunks: List[GoldChunk] = []

    for cand, reranker_score in gold_results:
        # Get links for this chunk
        links = sqlite_store.get_linked_chunks(
            cand.chunk_id,
            min_strength=min_strength,
            limit=5,
        )

        linked_ids = [l["linked_chunk_id"] for l in links]
        link_types = [l["link_type"]        for l in links]

        gold_chunk = GoldChunk(
            chunk_id=cand.chunk_id,
            modality=cand.modality,
            text=cand.text,
            source_file=cand.source_file,
            page_number=cand.page_number,
            reranker_score=reranker_score,
            rrf_score=0.0,    # assigned later from RRF score map
            session_id=cand.session_id,
            metadata_json=cand.metadata_json,
            image_path=cand.image_path,
            thumbnail_path=cand.thumbnail_path,
            start_time=cand.start_time,
            end_time=cand.end_time,
            linked_chunk_ids=linked_ids,
            link_types=link_types,
        )

        # Set timestamp_display for audio chunks
        if cand.modality == "audio" and (cand.start_time > 0 or cand.end_time > 0):
            try:
                from modules.ingestion.audio_ingestor import _format_timestamp_range
                gold_chunk.timestamp_display = _format_timestamp_range(
                    cand.start_time, cand.end_time
                )
            except Exception:
                pass

        gold_chunks.append(gold_chunk)

        # Collect strongly linked chunks not already in gold set
        for link in links:
            linked_id = link["linked_chunk_id"]
            if linked_id not in added_ids and len(linked_infos) < max_linked:
                added_ids.add(linked_id)
                linked_infos.append({
                    "chunk_id":  linked_id,
                    "modality":  link.get("linked_modality", "text"),
                    "link_type": link["link_type"],
                    "strength":  link["strength"],
                })

    # Create GoldChunk placeholders for linked chunks
    # Full content will be resolved by the LLM engine in Phase 7
    linked_gold_chunks: List[GoldChunk] = []
    for info in linked_infos[:max_linked]:
        linked_gold = GoldChunk(
            chunk_id=info["chunk_id"],
            modality=info["modality"],
            text=f"[Linked chunk: {info['link_type']} "
                 f"strength={info['strength']:.2f}]",
            source_file="",
            page_number=1,
            reranker_score=0.0,
            rrf_score=0.0,
            session_id="",
            link_types=[info["link_type"]],
        )
        linked_gold_chunks.append(linked_gold)

    return gold_chunks, linked_gold_chunks


# ── Public API ─────────────────────────────────────────────────────────────
class RetrievalEngine:
    """
    Complete multimodal hybrid retrieval engine.

    Orchestrates all retrieval steps:
    1. Query classification & routing
    2. Query encoding (BGE-M3 + CLIP)
    3. Parallel Milvus ANN + Tantivy BM25F
    4. Score normalization
    5. RRF fusion
    6. Modality diversification
    7. Cross-encoder reranking
    8. Cross-modal link enrichment
    9. CO_RETRIEVED link update

    Usage:
        engine = RetrievalEngine()
        engine.initialize()
        result = engine.retrieve(
            query="show me Q3 revenue chart",
            session_id="abc123",
        )
        for chunk in result.gold_chunks:
            print(chunk.modality, chunk.text[:100])
    """

    def __init__(
        self,
        milvus_store:  Optional[MilvusStore]  = None,
        tantivy_index: Optional[TantivyIndex] = None,
        sqlite_store:  Optional[SQLiteStore]  = None,
    ):
        """
        Accept optional pre-initialized store instances (for testing).
        If not provided, creates new instances.
        """
        self._milvus  = milvus_store  or MilvusStore()
        self._tantivy = tantivy_index or TantivyIndex()
        self._sqlite  = sqlite_store  or SQLiteStore()
        self._initialized = False

    def initialize(self):
        """
        Initialize all backing stores.
        Idempotent — safe to call multiple times.
        """
        self._milvus.initialize()
        self._tantivy.initialize()
        self._sqlite.initialize()
        self._initialized = True
        logger.success("RetrievalEngine initialized")

    def _ensure_initialized(self):
        if not self._initialized:
            self.initialize()

    def retrieve(
        self,
        query: str,
        session_id: Optional[str] = None,
        image_bytes: Optional[bytes] = None,
        audio_bytes: Optional[bytes] = None,
        top_k: int = MILVUS_TOP_K,
        final_k: int = FINAL_TOP_K,
        enable_reranking: bool = True,
        enable_link_enrichment: bool = True,
    ) -> RetrievalResult:
        """
        Execute the complete retrieval pipeline for a query.

        Args:
            query:                  Natural language query string
            session_id:             Optional session filter (scope to user's docs)
            image_bytes:            Optional uploaded image as bytes
            audio_bytes:            Optional uploaded audio as bytes
            top_k:                  Candidates per collection (default: 50)
            final_k:                Final gold chunks to return (default: 5)
            enable_reranking:       If False, skip cross-encoder (faster, less accurate)
            enable_link_enrichment: If False, skip cross-modal link enrichment

        Returns:
            RetrievalResult with gold_chunks, linked_chunks, and metadata
        """
        self._ensure_initialized()
        t_start = time.time()

        result = RetrievalResult(query=query)

        if not query.strip():
            logger.warning("Empty query received")
            return result

        # ── Step 1: Query Routing ──────────────────────────────────────
        routing = classify_query(
            query,
            has_image_attachment=(image_bytes is not None),
            has_audio_attachment=(audio_bytes is not None),
        )
        result.routing        = routing
        result.query_modality = routing["query_modality"]

        # ── Handle audio attachment: transcribe first ──────────────────
        if audio_bytes is not None:
            try:
                import tempfile
                import os
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                    f.write(audio_bytes)
                    tmp_path = f.name
                from modules.ingestion.audio_ingestor import _transcribe_audio
                whisper_result = _transcribe_audio(Path(tmp_path))
                if whisper_result:
                    transcript = whisper_result.get("text", "").strip()
                    if transcript:
                        query = f"{query} {transcript}".strip()
                        logger.info(
                            f"Audio attachment transcribed: {len(transcript)} chars"
                        )
                os.unlink(tmp_path)
            except Exception as e:
                logger.warning(f"Audio attachment transcription failed: {e}")

        # ── Step 2: Query Encoding ─────────────────────────────────────
        t_enc = time.time()
        embeddings = encode_query(query, self._milvus, image_bytes)
        result.latency_ms["encoding"] = (time.time() - t_enc) * 1000

        if embeddings["bge_m3"] is None and embeddings["clip"] is None:
            logger.error("All query embeddings failed — cannot retrieve")
            return result

        # ── Step 3: Parallel Retrieval ─────────────────────────────────
        all_candidates, retrieval_latencies = run_parallel_retrieval(
            query=query,
            embeddings=embeddings,
            routing=routing,
            milvus_store=self._milvus,
            tantivy_index=self._tantivy,
            session_id=session_id,
            top_k=top_k,
        )
        result.latency_ms.update(retrieval_latencies)
        result.total_candidates = len(all_candidates)

        if not all_candidates:
            logger.warning(
                "No candidates retrieved — empty index or failed searches"
            )
            return result

        # ── Step 4: Score Normalization ────────────────────────────────
        all_candidates = normalize_bm25_scores(all_candidates)

        # ── Step 5: RRF Fusion ─────────────────────────────────────────
        t_rrf = time.time()
        fused = fuse_results_rrf(all_candidates, top_n=100)
        result.latency_ms["rrf"] = (time.time() - t_rrf) * 1000

        if not fused:
            logger.warning("RRF fusion produced no results")
            return result

        # ── Step 6: Modality Diversification ──────────────────────────
        fused = diversify_modalities(
            fused,
            all_candidates=all_candidates,
        )

        # ── Step 7: Cross-Encoder Reranking ────────────────────────────
        if enable_reranking:
            t_rerank = time.time()
            reranked = rerank_candidates(
                query=query,
                candidates=fused,
                top_k=final_k,
                score_threshold=RERANKER_THRESHOLD,
            )
            result.latency_ms["reranking"] = (time.time() - t_rerank) * 1000
        else:
            # No reranking: take top_k by RRF score directly
            reranked = [
                (cand, rrf_s)
                for _, rrf_s, cand in fused[:final_k]
            ]

        if not reranked:
            logger.warning("No chunks passed reranker threshold")
            return result

        # Build RRF score map for assignment
        rrf_score_map = {chunk_id: s for chunk_id, s, _ in fused}

        # ── Step 8: Link Enrichment ────────────────────────────────────
        if enable_link_enrichment:
            t_link = time.time()
            gold_chunks, linked_chunks = enrich_with_links(
                gold_results=reranked,
                sqlite_store=self._sqlite,
            )
            result.latency_ms["link_enrichment"] = (
                (time.time() - t_link) * 1000
            )
        else:
            # No link enrichment: convert directly to GoldChunk
            gold_chunks = []
            linked_chunks = []
            for cand, reranker_score_val in reranked:
                gold_chunks.append(GoldChunk(
                    chunk_id=cand.chunk_id,
                    modality=cand.modality,
                    text=cand.text,
                    source_file=cand.source_file,
                    page_number=cand.page_number,
                    reranker_score=reranker_score_val,
                    rrf_score=rrf_score_map.get(cand.chunk_id, 0.0),
                    session_id=cand.session_id,
                    metadata_json=cand.metadata_json,
                    image_path=cand.image_path,
                    thumbnail_path=cand.thumbnail_path,
                    start_time=cand.start_time,
                    end_time=cand.end_time,
                ))

        # Assign RRF scores to gold chunks
        for gc in gold_chunks:
            gc.rrf_score = rrf_score_map.get(gc.chunk_id, 0.0)

        result.gold_chunks   = gold_chunks
        result.linked_chunks = linked_chunks

        # ── Step 9: CO_RETRIEVED Link Update ──────────────────────────
        all_returned_ids = result.all_chunk_ids
        if len(all_returned_ids) >= 2:
            modality_map = {
                gc.chunk_id: gc.modality
                for gc in result.all_chunks
            }
            try:
                self._sqlite.update_co_retrieved_links(
                    all_returned_ids, modality_map
                )
            except Exception as e:
                logger.warning(
                    f"CO_RETRIEVED link update failed (non-fatal): {e}"
                )

        # ── Final Latency ──────────────────────────────────────────────
        result.latency_ms["total"] = (time.time() - t_start) * 1000

        logger.info(
            f"Retrieval complete: query='{query[:60]}...' | "
            f"gold={len(result.gold_chunks)} | "
            f"linked={len(result.linked_chunks)} | "
            f"total={result.latency_ms['total']:.0f}ms"
        )
        return result

    def release_models(self):
        """
        Unload reranker from memory and release VRAM.
        Call before loading LLM.
        """
        _unload_reranker()
        self._milvus.release_models()
        logger.info("RetrievalEngine: models released")
