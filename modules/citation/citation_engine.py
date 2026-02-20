"""
modules/citation/citation_engine.py

Citation Engine for the Multimodal RAG system.

Takes LLM response text with [N] citation markers + the index_map
from Phase 7 and produces richly annotated CitationResult objects.

Responsibilities:
  1. Parse [N] markers from LLM response
  2. Build CitationObject per unique citation
  3. Detect cross-modal clusters (linked chunks cited together)
  4. Annotate response text with HTML <cite> spans
  5. Build source summary (unique files cited)
  6. Produce final CitationResult

The CitationResult is consumed by:
  - Phase 9 FastAPI endpoints (as JSON response payload)
  - Phase 10 Gradio frontend (renders citation badges)
  - Phase 8 Export engine (includes citations in exported docs)
"""

import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from loguru import logger

from modules.indexing.sqlite_store import SQLiteStore
from modules.retrieval.retrieval_engine import GoldChunk


# ── Link Type Display Labels ───────────────────────────────────────────────
LINK_TYPE_LABELS = {
    "same_page":          "📄 Same page",
    "temporal_proximity": "🕐 Same session",
    "co_retrieved":       "🔁 Frequently co-retrieved",
    "semantic":           "🔗 Semantically linked",
}

LINK_TYPE_COLORS = {
    "same_page":          "#2563EB",   # blue
    "temporal_proximity": "#7C3AED",   # purple
    "co_retrieved":       "#059669",   # green
    "semantic":           "#DC2626",   # red
}

MODALITY_LABELS = {
    "text":  "TEXT",
    "image": "IMAGE",
    "audio": "AUDIO",
}

MODALITY_COLORS = {
    "text":  "#1E40AF",
    "image": "#065F46",
    "audio": "#92400E",
}


# ── Data Structures ────────────────────────────────────────────────────────
@dataclass
class CitationLink:
    """
    Represents a cross-modal link involving a cited chunk.
    Displayed as a badge beneath the citation in the UI.
    """
    linked_chunk_id:   str
    linked_modality:   str
    link_type:         str
    strength:          float
    display_label:     str    # e.g. "📄 Same page as [3]"
    color:             str    # hex color for badge


@dataclass
class CitationObject:
    """
    Rich citation for a single [N] reference in the LLM response.
    Contains everything the UI needs to render a citation badge.
    """
    index:             int           # N in [N]
    chunk_id:          str
    modality:          str           # text | image | audio
    source_file:       str
    page_number:       int
    start_time:        float         # seconds (audio only)
    end_time:          float         # seconds (audio only)
    timestamp_display: str           # "HH:MM - HH:MM" (audio only)
    text_preview:      str           # first 200 chars of chunk text
    image_path:        str           # full path (image only)
    thumbnail_path:    str           # 256x256 thumbnail (image only)
    modality_label:    str           # "TEXT" | "IMAGE" | "AUDIO"
    modality_color:    str           # hex color for modality badge
    links:             List[CitationLink] = field(default_factory=list)
    html_span:         str = ""      # rendered <cite> HTML element

    def to_dict(self) -> Dict[str, Any]:
        return {
            "index":             self.index,
            "chunk_id":          self.chunk_id,
            "modality":          self.modality,
            "modality_label":    self.modality_label,
            "modality_color":    self.modality_color,
            "source_file":       self.source_file,
            "page_number":       self.page_number,
            "start_time":        self.start_time,
            "end_time":          self.end_time,
            "timestamp_display": self.timestamp_display,
            "text_preview":      self.text_preview,
            "image_path":        self.image_path,
            "thumbnail_path":    self.thumbnail_path,
            "html_span":         self.html_span,
            "links":             [
                {
                    "linked_chunk_id": lk.linked_chunk_id,
                    "linked_modality": lk.linked_modality,
                    "link_type":       lk.link_type,
                    "strength":        round(lk.strength, 4),
                    "display_label":   lk.display_label,
                    "color":           lk.color,
                }
                for lk in self.links
            ],
        }


@dataclass
class CrossModalCluster:
    """
    A group of cited chunks that are cross-modally linked to each other.
    Rendered as a connection panel in the UI.

    Example:
        chunks: [chunk_1 (text, page 7), chunk_2 (image, page 7)]
        link_type: "same_page"
        label: "📄 Text [1] and Image [2] are on the same page"
    """
    cluster_id:    str
    chunk_indices: List[int]         # citation indices e.g. [1, 3]
    chunk_ids:     List[str]
    link_type:     str
    strength:      float
    label:         str               # human-readable description
    color:         str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cluster_id":    self.cluster_id,
            "chunk_indices": self.chunk_indices,
            "chunk_ids":     self.chunk_ids,
            "link_type":     self.link_type,
            "strength":      round(self.strength, 4),
            "label":         self.label,
            "color":         self.color,
        }


@dataclass
class CitationResult:
    """
    Complete citation result for one LLM response.
    This is the final output of the CitationEngine.
    """
    original_response:   str
    annotated_response:  str                    # HTML-annotated version
    citations:           List[CitationObject]   = field(default_factory=list)
    clusters:            List[CrossModalCluster] = field(default_factory=list)
    source_summary:      List[Dict[str, Any]]   = field(default_factory=list)
    total_citations:     int = 0
    unique_sources:      int = 0
    has_cross_modal:     bool = False
    created_at:          str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_response":  self.original_response,
            "annotated_response": self.annotated_response,
            "citations":          [c.to_dict() for c in self.citations],
            "clusters":           [cl.to_dict() for cl in self.clusters],
            "source_summary":     self.source_summary,
            "total_citations":    self.total_citations,
            "unique_sources":     self.unique_sources,
            "has_cross_modal":    self.has_cross_modal,
            "created_at":         self.created_at,
        }


# ── HTML Annotation ────────────────────────────────────────────────────────
def _build_html_span(chunk: GoldChunk, index: int) -> str:
    """
    Build an HTML <cite> element for a citation reference.
    The data attributes let the frontend JavaScript attach
    hover cards, click handlers, and modal dialogs.

    Example output:
    <cite
      data-chunk-id="abc-123"
      data-modality="text"
      data-source="report.pdf"
      data-page="7"
      data-index="1"
      class="rag-citation rag-citation--text"
      title="report.pdf, page 7"
    >[1]</cite>
    """
    modality_class = f"rag-citation--{chunk.modality}"
    source_attr    = chunk.source_file.replace('"', '&quot;')
    title_text     = _build_citation_title(chunk)

    attrs = [
        f'data-chunk-id="{chunk.chunk_id}"',
        f'data-modality="{chunk.modality}"',
        f'data-source="{source_attr}"',
        f'data-page="{chunk.page_number}"',
        f'data-index="{index}"',
        f'class="rag-citation {modality_class}"',
        f'title="{title_text}"',
    ]

    if chunk.modality == "audio" and chunk.timestamp_display:
        attrs.append(f'data-timestamp="{chunk.timestamp_display}"')
    if chunk.modality == "image" and chunk.thumbnail_path:
        attrs.append(f'data-thumbnail="{chunk.thumbnail_path}"')

    return f'<cite {" ".join(attrs)}>[{index}]</cite>'


def _build_citation_title(chunk: GoldChunk) -> str:
    """
    Build the tooltip title string for a citation hover card.
    Shown on mouse-over in the UI.
    """
    if chunk.modality == "audio":
        ts = chunk.timestamp_display or f"{chunk.start_time:.0f}s"
        return f"{chunk.source_file}, {ts}"
    elif chunk.modality == "image":
        return f"{chunk.source_file}, page {chunk.page_number} (image)"
    else:
        return f"{chunk.source_file}, page {chunk.page_number}"


def annotate_response(
    response_text: str,
    citation_objects: List[CitationObject],
) -> str:
    """
    Replace all [N] markers in response text with HTML <cite> spans.

    Strategy:
    - Process citations in reverse order of index to preserve
      string positions when replacing (replacing [10] before [1]
      prevents accidental double-replacement)
    - Only replace exact [N] patterns, not partial matches

    Args:
        response_text:     Original LLM response with [N] markers
        citation_objects:  List of CitationObject, one per unique N

    Returns:
        HTML-annotated response string with <cite> elements.
    """
    if not citation_objects:
        return response_text

    annotated = response_text

    # Build index → html_span map
    span_map: Dict[int, str] = {
        c.index: c.html_span
        for c in citation_objects
        if c.html_span
    }

    # Replace in reverse index order to preserve positions
    for idx in sorted(span_map.keys(), reverse=True):
        span = span_map[idx]
        # Replace exact [N] — use word boundary to avoid [10] → [1]0
        pattern = rf"\[{idx}\]"
        annotated = re.sub(pattern, span, annotated)

    return annotated


# ── Cross-Modal Cluster Detection ──────────────────────────────────────────
def detect_clusters(
    citations: List[CitationObject],
    index_map: Dict[int, GoldChunk],
    sqlite_store: SQLiteStore,
    min_strength: float = 0.6,
) -> List[CrossModalCluster]:
    """
    Detect cross-modal clusters among the cited chunks.

    A cluster is created when two or more cited chunks have a
    cross-modal link between them with strength >= min_strength.

    For each pair of cited chunks (i, j):
    1. Query SQLiteStore.get_links_between(chunk_i, chunk_j)
    2. If links exist with strength >= min_strength, create a cluster
    3. Generate a human-readable label for the cluster

    Only creates clusters for DIFFERENT modalities (cross-modal).
    Same-modality links are less interesting for citation display.

    Args:
        citations:    List of CitationObject from current response
        index_map:    Map from citation index → GoldChunk
        sqlite_store: For querying cross-modal links
        min_strength: Minimum link strength to form a cluster

    Returns:
        List of CrossModalCluster objects, deduplicated.
    """
    clusters: List[CrossModalCluster] = []
    seen_pairs: Set[Tuple[str, str, str]] = set()   # (id_a, id_b, link_type)

    if len(citations) < 2:
        return clusters

    for i in range(len(citations)):
        for j in range(i + 1, len(citations)):
            cite_a = citations[i]
            cite_b = citations[j]

            # Only cross-modal clusters
            if cite_a.modality == cite_b.modality:
                continue

            # Query SQLite for links between these two chunks
            try:
                links = sqlite_store.get_links_between(
                    cite_a.chunk_id,
                    cite_b.chunk_id,
                )
            except Exception as e:
                logger.debug(f"Link query failed for cluster detection: {e}")
                continue

            for link in links:
                if link["strength"] < min_strength:
                    continue

                link_type  = link["link_type"]
                pair_key   = (cite_a.chunk_id, cite_b.chunk_id, link_type)
                if pair_key in seen_pairs:
                    continue
                seen_pairs.add(pair_key)

                label = _build_cluster_label(
                    cite_a, cite_b, link_type, link["strength"]
                )
                color = LINK_TYPE_COLORS.get(link_type, "#6B7280")

                import uuid as _uuid
                cluster = CrossModalCluster(
                    cluster_id=str(_uuid.uuid4())[:8],
                    chunk_indices=[cite_a.index, cite_b.index],
                    chunk_ids=[cite_a.chunk_id, cite_b.chunk_id],
                    link_type=link_type,
                    strength=link["strength"],
                    label=label,
                    color=color,
                )
                clusters.append(cluster)

    if clusters:
        logger.info(
            f"Cross-modal clusters detected: {len(clusters)} "
            f"({sum(1 for c in clusters if c.link_type == 'same_page')} same_page, "
            f"{sum(1 for c in clusters if c.link_type == 'semantic')} semantic)"
        )

    return clusters


def _build_cluster_label(
    cite_a: CitationObject,
    cite_b: CitationObject,
    link_type: str,
    strength: float,
) -> str:
    """
    Generate a human-readable cluster label for the UI connection panel.

    Examples:
        "📄 [1] (text) and [3] (image) are on the same page in report.pdf"
        "🔗 [2] (audio) and [4] (image) are semantically linked (87% match)"
        "🕐 [1] (text) and [5] (audio) were ingested in the same session"
    """
    base_label = LINK_TYPE_LABELS.get(link_type, link_type)
    mod_a = cite_a.modality.upper()
    mod_b = cite_b.modality.upper()
    idx_a = cite_a.index
    idx_b = cite_b.index

    if link_type == "same_page":
        return (
            f"{base_label}: [{idx_a}] ({mod_a}) and [{idx_b}] ({mod_b}) "
            f"are on the same page in {cite_a.source_file}"
        )
    elif link_type == "semantic":
        pct = int(strength * 100)
        return (
            f"{base_label}: [{idx_a}] ({mod_a}) and [{idx_b}] ({mod_b}) "
            f"share semantic content ({pct}% match)"
        )
    elif link_type == "temporal_proximity":
        return (
            f"{base_label}: [{idx_a}] ({mod_a}) and [{idx_b}] ({mod_b}) "
            f"were uploaded in the same session"
        )
    elif link_type == "co_retrieved":
        return (
            f"{base_label}: [{idx_a}] ({mod_a}) and [{idx_b}] ({mod_b}) "
            f"are frequently retrieved together"
        )
    else:
        return f"{base_label}: [{idx_a}] and [{idx_b}]"


# ── Source Summary ─────────────────────────────────────────────────────────
def build_source_summary(
    citations: List[CitationObject],
) -> List[Dict[str, Any]]:
    """
    Build a deduplicated summary of all source files cited.
    Used to render the "Sources" section at the bottom of the response.

    Returns:
        List of source dicts ordered by first citation index:
        [{
            "source_file":  "report_2024.pdf",
            "modalities":   ["text", "image"],
            "pages":        [7, 8, 12],
            "timestamps":   [],
            "citation_indices": [1, 3, 5],
            "first_index":  1,
        }]
    """
    source_map: Dict[str, Dict[str, Any]] = {}

    for citation in sorted(citations, key=lambda c: c.index):
        src = citation.source_file
        if src not in source_map:
            source_map[src] = {
                "source_file":      src,
                "modalities":       set(),
                "pages":            set(),
                "timestamps":       [],
                "citation_indices": [],
                "first_index":      citation.index,
            }

        source_map[src]["modalities"].add(citation.modality)
        source_map[src]["citation_indices"].append(citation.index)

        if citation.modality in ("text", "image") and citation.page_number > 0:
            source_map[src]["pages"].add(citation.page_number)

        if citation.modality == "audio" and citation.timestamp_display:
            if citation.timestamp_display not in source_map[src]["timestamps"]:
                source_map[src]["timestamps"].append(citation.timestamp_display)

    # Convert sets to sorted lists
    result = []
    for src_data in sorted(source_map.values(), key=lambda x: x["first_index"]):
        result.append({
            "source_file":      src_data["source_file"],
            "modalities":       sorted(src_data["modalities"]),
            "pages":            sorted(src_data["pages"]),
            "timestamps":       src_data["timestamps"],
            "citation_indices": src_data["citation_indices"],
            "first_index":      src_data["first_index"],
        })

    return result


# ── Public API ─────────────────────────────────────────────────────────────
class CitationEngine:
    """
    Produces richly annotated CitationResult objects from
    LLM response text and Phase 7 citation metadata.

    Usage:
        engine = CitationEngine(sqlite_store=store)

        # After LLMEngine.complete() returns:
        llm_result = llm_engine.complete(query, retrieval_result)
        index_map  = llm_engine.get_last_index_map()

        citation_result = engine.process(
            response_text=llm_result["response"],
            citation_metadata=llm_result["citations"],
            index_map=index_map,
        )

        # Use citation_result.annotated_response for the UI
        # Use citation_result.citations for the sidebar panel
        # Use citation_result.clusters for the connection panel
    """

    def __init__(self, sqlite_store: Optional[SQLiteStore] = None):
        self._sqlite = sqlite_store or SQLiteStore()

    def process(
        self,
        response_text: str,
        citation_metadata: List[Dict[str, Any]],
        index_map: Dict[int, GoldChunk],
        min_cluster_strength: float = 0.6,
    ) -> CitationResult:
        """
        Process an LLM response and build a complete CitationResult.

        Args:
            response_text:       Full LLM response text with [N] markers
            citation_metadata:   List of citation dicts from LLMEngine
                                 (output of build_citation_metadata())
            index_map:           Dict mapping citation index → GoldChunk
                                 (from LLMEngine.get_last_index_map())
            min_cluster_strength: Minimum link strength for cluster detection

        Returns:
            CitationResult with annotated response, citation objects,
            cross-modal clusters, and source summary.
        """
        now = datetime.now(timezone.utc).isoformat()

        # Step 1: Build CitationObject for each unique citation
        citation_objects: List[CitationObject] = []
        for meta in citation_metadata:
            idx   = meta.get("index", 0)
            chunk = index_map.get(idx)
            if chunk is None:
                logger.debug(f"Citation [{idx}] has no matching chunk in index_map")
                continue

            # Build HTML span
            html_span = _build_html_span(chunk, idx)

            # Gather cross-modal links for this chunk
            citation_links: List[CitationLink] = []
            raw_links = meta.get("links") or []

            # Also fetch from SQLite for completeness
            try:
                db_links = self._sqlite.get_linked_chunks(
                    chunk.chunk_id,
                    min_strength=0.5,
                    limit=5,
                )
                # Merge — SQLite links are more complete
                existing_linked_ids = {l.get("linked_chunk_id") for l in raw_links}
                for db_link in db_links:
                    if db_link["linked_chunk_id"] not in existing_linked_ids:
                        raw_links.append({
                            "linked_chunk_id": db_link["linked_chunk_id"],
                            "linked_modality": db_link["linked_modality"],
                            "link_type":       db_link["link_type"],
                            "strength":        db_link["strength"],
                        })
            except Exception as e:
                logger.debug(f"SQLite link fetch failed: {e}")

            for lk in raw_links:
                link_type = lk.get("link_type", "")
                strength  = float(lk.get("strength", 0.0))

                # Find citation index of linked chunk if it's cited
                linked_id  = lk.get("linked_chunk_id", "")
                linked_idx = next(
                    (k for k, gc in index_map.items() if gc.chunk_id == linked_id),
                    None,
                )
                display_label = LINK_TYPE_LABELS.get(link_type, link_type)
                if linked_idx is not None:
                    display_label += f" as [{linked_idx}]"

                citation_links.append(CitationLink(
                    linked_chunk_id=linked_id,
                    linked_modality=lk.get("linked_modality", "text"),
                    link_type=link_type,
                    strength=strength,
                    display_label=display_label,
                    color=LINK_TYPE_COLORS.get(link_type, "#6B7280"),
                ))

            text_preview = (chunk.text or "")[:200]

            citation_obj = CitationObject(
                index=idx,
                chunk_id=chunk.chunk_id,
                modality=chunk.modality,
                source_file=chunk.source_file,
                page_number=chunk.page_number,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                timestamp_display=chunk.timestamp_display,
                text_preview=text_preview,
                image_path=chunk.image_path,
                thumbnail_path=chunk.thumbnail_path,
                modality_label=MODALITY_LABELS.get(chunk.modality, chunk.modality.upper()),
                modality_color=MODALITY_COLORS.get(chunk.modality, "#374151"),
                links=citation_links,
                html_span=html_span,
            )
            citation_objects.append(citation_obj)

        # Step 2: Annotate response text
        annotated = annotate_response(response_text, citation_objects)

        # Step 3: Detect cross-modal clusters
        clusters = detect_clusters(
            citations=citation_objects,
            index_map=index_map,
            sqlite_store=self._sqlite,
            min_strength=min_cluster_strength,
        )

        # Step 4: Build source summary
        source_summary = build_source_summary(citation_objects)

        # Step 5: Build final CitationResult
        unique_sources = len({c.source_file for c in citation_objects})
        has_cross_modal = (
            len({c.modality for c in citation_objects}) > 1
            or len(clusters) > 0
        )

        result = CitationResult(
            original_response=response_text,
            annotated_response=annotated,
            citations=citation_objects,
            clusters=clusters,
            source_summary=source_summary,
            total_citations=len(citation_objects),
            unique_sources=unique_sources,
            has_cross_modal=has_cross_modal,
            created_at=now,
        )

        logger.info(
            f"CitationEngine: {len(citation_objects)} citations processed, "
            f"{len(clusters)} clusters, "
            f"{unique_sources} unique sources, "
            f"cross_modal={has_cross_modal}"
        )
        return result

    def process_from_llm_result(
        self,
        llm_result: Dict[str, Any],
        index_map: Dict[int, GoldChunk],
    ) -> CitationResult:
        """
        Convenience wrapper that accepts the dict returned
        by LLMEngine.complete() directly.

        Args:
            llm_result: Dict with keys 'response' and 'citations'
            index_map:  From LLMEngine.get_last_index_map()

        Returns:
            CitationResult
        """
        return self.process(
            response_text=llm_result.get("response", ""),
            citation_metadata=llm_result.get("citations", []),
            index_map=index_map,
        )
