"""
tests/test_retrieval_engine.py

Comprehensive tests for the Phase 6 RetrievalEngine module.

Tests are organized by pipeline step:
  1. Query routing (classify_query)
  2. Score normalization
  3. RRF fusion
  4. Modality diversification
  5. Reranker input construction
  6. Data structures (RawCandidate, GoldChunk, RetrievalResult)
  7. Full pipeline (with mock stores)

All tests use mocks — no Milvus, Tantivy, or SQLite required.
"""

import json
import math
import pytest
from unittest.mock import MagicMock, patch, PropertyMock
from collections import defaultdict

from modules.retrieval.retrieval_engine import (
    classify_query,
    normalize_bm25_scores,
    rrf_score,
    fuse_results_rrf,
    diversify_modalities,
    _build_reranker_input_text,
    _sigmoid,
    rerank_candidates,
    enrich_with_links,
    encode_query,
    run_parallel_retrieval,
    RawCandidate,
    GoldChunk,
    RetrievalResult,
    RetrievalEngine,
    RRF_K,
)

from config.settings import (
    BGE_M3_DIM,
    CLIP_DIM,
    FINAL_TOP_K,
    RERANKER_THRESHOLD,
    MILVUS_TOP_K,
    BM25_TOP_K,
)


# ── Helper factories ───────────────────────────────────────────────────────
def make_candidate(
    chunk_id: str = "chunk-001",
    score: float = 0.85,
    source: str = "milvus_text",
    modality: str = "text",
    text: str = "sample text",
    source_file: str = "report.pdf",
    page_number: int = 1,
    session_id: str = "sess-1",
    metadata_json: str = "{}",
    image_path: str = "",
    thumbnail_path: str = "",
    start_time: float = 0.0,
    end_time: float = 0.0,
) -> RawCandidate:
    return RawCandidate(
        chunk_id=chunk_id,
        score=score,
        source=source,
        modality=modality,
        text=text,
        source_file=source_file,
        page_number=page_number,
        session_id=session_id,
        metadata_json=metadata_json,
        image_path=image_path,
        thumbnail_path=thumbnail_path,
        start_time=start_time,
        end_time=end_time,
    )


def make_mock_milvus():
    """Create a MilvusStore mock with standard search methods."""
    mock = MagicMock()
    mock.initialize.return_value = None
    mock.release_models.return_value = None
    mock.compute_query_embedding.return_value = [0.1] * BGE_M3_DIM
    mock.search_text.return_value = [
        {
            "chunk_id": "text-001",
            "score": 0.90,
            "source_file": "doc.pdf",
            "page_number": 2,
            "session_id": "sess-1",
            "modality": "text",
            "metadata_json": '{"heading": "Introduction"}',
        },
        {
            "chunk_id": "text-002",
            "score": 0.75,
            "source_file": "doc.pdf",
            "page_number": 5,
            "session_id": "sess-1",
            "modality": "text",
            "metadata_json": '{}',
        },
    ]
    mock.search_images.return_value = [
        {
            "chunk_id": "img-001",
            "score": 0.82,
            "source_file": "slides.pdf",
            "page_number": 3,
            "session_id": "sess-1",
            "modality": "image",
            "metadata_json": '{}',
            "ocr_text": "Revenue Q3 2024",
            "image_path": "media/img001.png",
            "thumbnail_path": "thumbs/img001.webp",
        },
    ]
    mock.search_audio.return_value = [
        {
            "chunk_id": "aud-001",
            "score": 0.78,
            "source_file": "meeting.mp3",
            "page_number": 1,
            "session_id": "sess-1",
            "modality": "audio",
            "metadata_json": json.dumps({
                "start_time": 120.5, "end_time": 145.2
            }),
        },
    ]
    return mock


def make_mock_tantivy():
    """Create a TantivyIndex mock with standard search method."""
    mock = MagicMock()
    mock.initialize.return_value = None
    mock.search.return_value = [
        {
            "chunk_id": "text-001",  # overlaps with Milvus — tests dedup
            "score": 8.5,
            "text": "Q3 revenue analysis shows growth...",
            "source_file": "doc.pdf",
            "modality": "text",
            "session_id": "sess-1",
        },
        {
            "chunk_id": "bm25-001",
            "score": 6.2,
            "text": "Revenue breakdown by quarter...",
            "source_file": "financial.pdf",
            "modality": "text",
            "session_id": "sess-1",
        },
    ]
    return mock


def make_mock_sqlite():
    """Create a SQLiteStore mock."""
    mock = MagicMock()
    mock.initialize.return_value = None
    mock.get_linked_chunks.return_value = [
        {
            "linked_chunk_id": "linked-001",
            "linked_modality": "image",
            "link_type": "SAME_PAGE",
            "strength": 0.85,
        },
    ]
    mock.update_co_retrieved_links.return_value = 1
    return mock


# ═══════════════════════════════════════════════════════════════════════════
#  Step 1: Query Routing Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestClassifyQuery:
    """Tests for query classification and routing logic."""

    def test_basic_text_query(self):
        routing = classify_query("What is Q3 revenue?")
        assert routing["query_modality"] == "text"
        assert routing["search_text"] is True
        assert routing["search_images"] is True
        assert routing["search_audio"] is True
        assert routing["use_bm25"] is True
        assert routing["force_image"] is False
        assert routing["force_audio"] is False
        assert routing["force_bm25"] is False

    def test_image_attachment_forces_image(self):
        routing = classify_query("describe this", has_image_attachment=True)
        assert routing["force_image"] is True
        assert routing["query_modality"] == "image"

    def test_audio_attachment_forces_audio(self):
        routing = classify_query("what was said?", has_audio_attachment=True)
        assert routing["force_audio"] is True
        assert routing["query_modality"] == "audio"

    def test_visual_keyword_forces_image(self):
        routing = classify_query("show me the chart of revenue")
        assert routing["force_image"] is True
        assert routing["query_modality"] == "image"

    def test_visual_keyword_diagram(self):
        routing = classify_query("find the architecture diagram")
        assert routing["force_image"] is True

    def test_visual_keyword_screenshot(self):
        routing = classify_query("where is the screenshot?")
        assert routing["force_image"] is True

    def test_timestamp_forces_audio(self):
        routing = classify_query("what was said at 14:32?")
        assert routing["force_audio"] is True
        assert routing["query_modality"] == "audio"
        assert routing["has_timestamp"] is True
        assert routing["detected_timestamp"] is not None

    def test_timestamp_minutes_pattern(self):
        routing = classify_query("skip to the 45 minute mark")
        assert routing["force_audio"] is True
        assert routing["has_timestamp"] is True

    def test_filename_forces_bm25(self):
        routing = classify_query("find report.pdf")
        assert routing["force_bm25"] is True
        assert "report.pdf" in routing["detected_filename"]

    def test_filename_mp3(self):
        routing = classify_query("open meeting.mp3")
        assert "meeting.mp3" in routing["detected_filename"]
        assert routing["force_bm25"] is True

    def test_priority_image_over_timestamp(self):
        """Image attachment takes priority over timestamp detection."""
        routing = classify_query(
            "what is at 14:32?", has_image_attachment=True
        )
        assert routing["force_image"] is True
        assert routing["query_modality"] == "image"
        # Timestamp is still detected
        assert routing["has_timestamp"] is True

    def test_always_searches_all_modalities(self):
        """Even with force flags, all modalities are searched."""
        routing = classify_query("show me the chart")
        assert routing["search_text"] is True
        assert routing["search_images"] is True
        assert routing["search_audio"] is True

    def test_empty_query(self):
        routing = classify_query("")
        assert routing["query_modality"] == "text"
        assert routing["force_image"] is False

    def test_no_false_visual_keyword(self):
        """Words like 'charge' shouldn't trigger force_image."""
        routing = classify_query("how much was the charge?")
        assert routing["force_image"] is False


# ═══════════════════════════════════════════════════════════════════════════
#  Step 4: Score Normalization Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestScoreNormalization:
    """Tests for BM25 score normalization."""

    def test_normalize_bm25_min_max(self):
        candidates = [
            make_candidate(chunk_id="c1", score=10.0, source="bm25"),
            make_candidate(chunk_id="c2", score=5.0, source="bm25"),
            make_candidate(chunk_id="c3", score=0.0, source="bm25"),
        ]
        result = normalize_bm25_scores(candidates)
        # Highest score should be close to 1.0
        assert result[0].score == pytest.approx(1.0, abs=0.01)
        # Mid score should be ~0.5
        assert result[1].score == pytest.approx(0.5, abs=0.01)
        # Lowest should be close to 0.0
        assert result[2].score == pytest.approx(0.0, abs=0.01)

    def test_normalize_leaves_milvus_unchanged(self):
        candidates = [
            make_candidate(chunk_id="c1", score=0.9, source="milvus_text"),
            make_candidate(chunk_id="c2", score=8.0, source="bm25"),
            make_candidate(chunk_id="c3", score=2.0, source="bm25"),
        ]
        result = normalize_bm25_scores(candidates)
        # Milvus score should be unchanged
        assert result[0].score == 0.9
        # BM25 scores should be normalized
        assert result[1].score != 8.0
        assert 0 <= result[1].score <= 1.0

    def test_normalize_single_bm25(self):
        """Single BM25 candidate should not be normalized."""
        candidates = [
            make_candidate(chunk_id="c1", score=5.0, source="bm25"),
        ]
        result = normalize_bm25_scores(candidates)
        assert result[0].score == 5.0  # unchanged — need ≥2 for normalization

    def test_normalize_no_bm25(self):
        """No BM25 candidates should leave all unchanged."""
        candidates = [
            make_candidate(chunk_id="c1", score=0.9, source="milvus_text"),
            make_candidate(chunk_id="c2", score=0.8, source="milvus_image"),
        ]
        result = normalize_bm25_scores(candidates)
        assert result[0].score == 0.9
        assert result[1].score == 0.8


# ═══════════════════════════════════════════════════════════════════════════
#  Step 5: RRF Fusion Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestRRFFusion:
    """Tests for Reciprocal Rank Fusion."""

    def test_rrf_score_formula(self):
        """rrf_score(rank=0, k=60) = 1/60."""
        assert rrf_score(0, k=60) == pytest.approx(1 / 60)
        assert rrf_score(1, k=60) == pytest.approx(1 / 61)
        assert rrf_score(10, k=60) == pytest.approx(1 / 70)

    def test_rrf_custom_k(self):
        assert rrf_score(0, k=10) == pytest.approx(1 / 10)

    def test_fuse_single_source(self):
        candidates = [
            make_candidate(chunk_id="c1", score=0.9, source="milvus_text"),
            make_candidate(chunk_id="c2", score=0.7, source="milvus_text"),
        ]
        result = fuse_results_rrf(candidates, top_n=10)
        assert len(result) == 2
        # c1 should rank higher (higher score → rank 0 → higher RRF)
        assert result[0][0] == "c1"
        assert result[1][0] == "c2"

    def test_fuse_multi_source_dedup(self):
        """Chunk appearing in both Milvus and BM25 should get higher RRF score."""
        candidates = [
            make_candidate(chunk_id="dup-1", score=0.9, source="milvus_text"),
            make_candidate(chunk_id="uniq-1", score=0.88, source="milvus_text"),
            make_candidate(chunk_id="dup-1", score=0.8, source="bm25"),
            make_candidate(chunk_id="uniq-2", score=0.95, source="bm25"),
        ]
        result = fuse_results_rrf(candidates, top_n=10)
        # dup-1 appears in 2 sources — should get cumulative RRF score
        ids = [r[0] for r in result]
        assert "dup-1" in ids
        # Check unique count: dup-1 + uniq-1 + uniq-2 = 3
        assert len(result) == 3

        # dup-1 should have highest RRF score (rank 0 in both sources)
        rrf_dup = next(s for cid, s, _ in result if cid == "dup-1")
        rrf_uniq = next(s for cid, s, _ in result if cid == "uniq-2")
        assert rrf_dup > rrf_uniq  # multi-source boost

    def test_fuse_empty(self):
        result = fuse_results_rrf([], top_n=10)
        assert result == []

    def test_fuse_top_n_limit(self):
        candidates = [
            make_candidate(chunk_id=f"c{i}", score=0.9 - i * 0.01, source="milvus_text")
            for i in range(20)
        ]
        result = fuse_results_rrf(candidates, top_n=5)
        assert len(result) == 5

    def test_fuse_preserves_best_candidate(self):
        """The best_candidate for a deduped chunk should be the one with highest score."""
        candidates = [
            make_candidate(
                chunk_id="dup-1", score=0.5, source="bm25",
                text="bm25 text"
            ),
            make_candidate(
                chunk_id="dup-1", score=0.9, source="milvus_text",
                text="milvus text"
            ),
        ]
        result = fuse_results_rrf(candidates, top_n=10)
        assert len(result) == 1
        # Best candidate should have the higher score text
        assert result[0][2].text == "milvus text"
        assert result[0][2].score == 0.9


# ═══════════════════════════════════════════════════════════════════════════
#  Step 6: Modality Diversification Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestDiversification:
    """Tests for modality diversification."""

    def test_no_change_when_diverse_enough(self):
        """If all minimums are met, no padding needed."""
        fused = [
            ("t1", 0.5, make_candidate(chunk_id="t1", modality="text")),
            ("i1", 0.4, make_candidate(chunk_id="i1", modality="image")),
            ("a1", 0.3, make_candidate(chunk_id="a1", modality="audio")),
        ]
        result = diversify_modalities(
            fused, min_text=1, min_image=1, min_audio=1,
            all_candidates=[],
        )
        assert len(result) == 3

    def test_padding_under_represented_modality(self):
        """If images are under-represented, should add extras."""
        fused = [
            ("t1", 0.5, make_candidate(chunk_id="t1", modality="text")),
            ("t2", 0.4, make_candidate(chunk_id="t2", modality="text")),
        ]
        extra_images = [
            make_candidate(chunk_id="img-extra-1", modality="image", score=0.7),
            make_candidate(chunk_id="img-extra-2", modality="image", score=0.6),
        ]
        result = diversify_modalities(
            fused, min_text=1, min_image=2, min_audio=0,
            all_candidates=extra_images,
        )
        # Should add 2 image chunks
        assert len(result) == 4
        ids = [r[0] for r in result]
        assert "img-extra-1" in ids
        assert "img-extra-2" in ids

    def test_no_duplicates_in_padding(self):
        """Padding should not add chunks already in fused."""
        fused = [
            ("img-1", 0.5, make_candidate(chunk_id="img-1", modality="image")),
        ]
        all_cands = [
            make_candidate(chunk_id="img-1", modality="image", score=0.9),
            make_candidate(chunk_id="img-2", modality="image", score=0.8),
        ]
        result = diversify_modalities(
            fused, min_image=2, min_text=0, min_audio=0,
            all_candidates=all_cands,
        )
        assert len(result) == 2
        ids = [r[0] for r in result]
        assert ids.count("img-1") == 1  # no duplicate

    def test_no_all_candidates_returns_unchanged(self):
        fused = [("t1", 0.5, make_candidate(chunk_id="t1"))]
        result = diversify_modalities(fused)  # no all_candidates
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  Reranker Input Construction Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestRerankerInput:
    """Tests for _build_reranker_input_text."""

    def test_text_chunk(self):
        cand = make_candidate(modality="text", text="Revenue grew 15%")
        result = _build_reranker_input_text("What is revenue?", cand)
        assert result == "Revenue grew 15%"

    def test_empty_text_chunk(self):
        cand = make_candidate(modality="text", text="")
        result = _build_reranker_input_text("query", cand)
        assert result == "[empty]"

    def test_image_chunk_with_ocr(self):
        cand = make_candidate(modality="image", text="Q3 Revenue Chart")
        result = _build_reranker_input_text("query", cand)
        assert "[OCR]" in result
        assert "Q3 Revenue Chart" in result

    def test_image_chunk_with_description(self):
        meta = json.dumps({"llm_description": "Bar chart showing revenue growth"})
        cand = make_candidate(
            modality="image", text="Q3 Chart", metadata_json=meta
        )
        result = _build_reranker_input_text("query", cand)
        assert "[OCR]" in result
        assert "[Description]" in result
        assert "Bar chart" in result

    def test_image_chunk_no_text(self):
        cand = make_candidate(modality="image", text="")
        result = _build_reranker_input_text("query", cand)
        assert result == "[image]"

    def test_audio_chunk_with_timestamps(self):
        cand = make_candidate(
            modality="audio",
            text="We discussed the quarterly results",
            start_time=120.0,
            end_time=145.0,
        )
        with patch(
            "modules.retrieval.retrieval_engine._format_timestamp_range",
            create=True,
        ):
            # Even if import fails, should still return text
            result = _build_reranker_input_text("query", cand)
            assert "quarterly results" in result

    def test_audio_chunk_no_text(self):
        cand = make_candidate(modality="audio", text="")
        result = _build_reranker_input_text("query", cand)
        assert "[audio segment]" in result


# ═══════════════════════════════════════════════════════════════════════════
#  Sigmoid & RRF Score Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestSigmoid:
    """Tests for the sigmoid function."""

    def test_sigmoid_zero(self):
        assert _sigmoid(0.0) == pytest.approx(0.5)

    def test_sigmoid_large_positive(self):
        assert _sigmoid(10.0) > 0.99

    def test_sigmoid_large_negative(self):
        assert _sigmoid(-10.0) < 0.01

    def test_sigmoid_range(self):
        """Sigmoid output should always be in [0, 1]."""
        for x in [-100, -10, -1, 0, 1, 10, 100]:
            val = _sigmoid(float(x))
            assert 0 <= val <= 1


# ═══════════════════════════════════════════════════════════════════════════
#  Data Structure Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestDataStructures:
    """Tests for RawCandidate, GoldChunk, RetrievalResult."""

    def test_raw_candidate_creation(self):
        c = make_candidate(chunk_id="test-1", score=0.95)
        assert c.chunk_id == "test-1"
        assert c.score == 0.95
        assert c.modality == "text"

    def test_gold_chunk_to_dict(self):
        gc = GoldChunk(
            chunk_id="gc-1",
            modality="text",
            text="some text",
            source_file="doc.pdf",
            page_number=3,
            reranker_score=0.92,
            rrf_score=0.015,
        )
        d = gc.to_dict()
        assert d["chunk_id"] == "gc-1"
        assert d["reranker_score"] == 0.92
        assert d["rrf_score"] == 0.015
        assert isinstance(d["linked_chunk_ids"], list)

    def test_gold_chunk_default_lists(self):
        gc = GoldChunk(
            chunk_id="gc-2",
            modality="image",
            text="",
            source_file="",
            page_number=1,
            reranker_score=0.5,
            rrf_score=0.01,
        )
        assert gc.linked_chunk_ids == []
        assert gc.link_types == []

    def test_retrieval_result_all_chunks(self):
        gold = GoldChunk(
            chunk_id="g1", modality="text", text="gold",
            source_file="", page_number=1,
            reranker_score=0.9, rrf_score=0.01,
        )
        linked = GoldChunk(
            chunk_id="l1", modality="image", text="linked",
            source_file="", page_number=1,
            reranker_score=0.0, rrf_score=0.0,
        )
        rr = RetrievalResult(
            query="test",
            gold_chunks=[gold],
            linked_chunks=[linked],
        )
        assert len(rr.all_chunks) == 2
        assert rr.all_chunk_ids == ["g1", "l1"]

    def test_retrieval_result_empty(self):
        rr = RetrievalResult(query="empty test")
        assert rr.gold_chunks == []
        assert rr.linked_chunks == []
        assert rr.all_chunks == []
        assert rr.all_chunk_ids == []


# ═══════════════════════════════════════════════════════════════════════════
#  Step 7: Reranking Tests (with mocked model)
# ═══════════════════════════════════════════════════════════════════════════
class TestReranking:
    """Tests for cross-encoder reranking."""

    @patch("modules.retrieval.retrieval_engine._load_reranker")
    def test_rerank_basic(self, mock_load):
        """Basic reranking with mocked scores."""
        mock_model = MagicMock()
        # Return raw scores (before sigmoid)
        mock_model.predict.return_value = [5.0, -2.0, 0.5]
        mock_load.return_value = mock_model

        candidates = [
            ("c1", 0.5, make_candidate(chunk_id="c1", text="first")),
            ("c2", 0.4, make_candidate(chunk_id="c2", text="second")),
            ("c3", 0.3, make_candidate(chunk_id="c3", text="third")),
        ]

        result = rerank_candidates("query", candidates, top_k=3, score_threshold=0.0)
        assert len(result) >= 1
        # c1 has highest raw score (5.0) so highest after sigmoid
        assert result[0][0].chunk_id == "c1"
        assert result[0][1] > 0.99  # sigmoid(5.0) ≈ 0.993

    @patch("modules.retrieval.retrieval_engine._load_reranker")
    def test_rerank_threshold_filtering(self, mock_load):
        """Chunks below threshold should be filtered out."""
        mock_model = MagicMock()
        # sigmoid(-5) ≈ 0.007, sigmoid(5) ≈ 0.993
        mock_model.predict.return_value = [5.0, -5.0]
        mock_load.return_value = mock_model

        candidates = [
            ("c1", 0.5, make_candidate(chunk_id="c1", text="relevant")),
            ("c2", 0.4, make_candidate(chunk_id="c2", text="irrelevant")),
        ]
        result = rerank_candidates(
            "query", candidates, top_k=10, score_threshold=0.5
        )
        # Only c1 should pass (sigmoid(5.0) > 0.5)
        assert len(result) == 1
        assert result[0][0].chunk_id == "c1"

    @patch("modules.retrieval.retrieval_engine._load_reranker")
    def test_rerank_top_k_limit(self, mock_load):
        """Should return at most top_k results."""
        mock_model = MagicMock()
        mock_model.predict.return_value = [5.0, 4.0, 3.0, 2.0, 1.0]
        mock_load.return_value = mock_model

        candidates = [
            (f"c{i}", 0.5, make_candidate(chunk_id=f"c{i}", text=f"text {i}"))
            for i in range(5)
        ]
        result = rerank_candidates("query", candidates, top_k=2, score_threshold=0.0)
        assert len(result) == 2

    def test_rerank_empty_candidates(self):
        result = rerank_candidates("query", [], top_k=5)
        assert result == []

    @patch("modules.retrieval.retrieval_engine._load_reranker")
    def test_rerank_fallback_on_predict_failure(self, mock_load):
        """If predict() fails, should fall back to RRF order."""
        mock_model = MagicMock()
        mock_model.predict.side_effect = RuntimeError("OOM")
        mock_load.return_value = mock_model

        candidates = [
            ("c1", 0.5, make_candidate(chunk_id="c1", text="text1")),
            ("c2", 0.4, make_candidate(chunk_id="c2", text="text2")),
        ]
        result = rerank_candidates("query", candidates, top_k=2)
        # Fallback should return candidates in RRF order
        assert len(result) == 2

    @patch("modules.retrieval.retrieval_engine._load_reranker")
    def test_rerank_fallback_on_load_failure(self, mock_load):
        """If reranker fails to load, should fall back to RRF order."""
        mock_load.side_effect = RuntimeError("Model not found")

        candidates = [
            ("c1", 0.5, make_candidate(chunk_id="c1", text="text1")),
        ]
        result = rerank_candidates("query", candidates, top_k=5)
        assert len(result) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  Step 8: Link Enrichment Tests
# ═══════════════════════════════════════════════════════════════════════════
class TestLinkEnrichment:
    """Tests for cross-modal link enrichment."""

    def test_enrich_basic(self):
        sqlite = make_mock_sqlite()
        gold_results = [
            (make_candidate(chunk_id="c1", text="main text"), 0.95),
        ]
        gold, linked = enrich_with_links(gold_results, sqlite, max_linked=2)
        assert len(gold) == 1
        assert gold[0].chunk_id == "c1"
        # Should have linked chunks from mock
        assert len(linked) <= 2

    def test_enrich_linked_chunk_not_duplicate_gold(self):
        """Linked chunk should not duplicate a gold chunk."""
        sqlite = MagicMock()
        sqlite.get_linked_chunks.return_value = [
            {
                "linked_chunk_id": "c1",   # same as gold chunk!
                "linked_modality": "text",
                "link_type": "SAME_PAGE",
                "strength": 0.9,
            },
        ]
        gold_results = [
            (make_candidate(chunk_id="c1", text="main text"), 0.95),
        ]
        gold, linked = enrich_with_links(gold_results, sqlite, max_linked=2)
        # c1 is already in gold — should not appear in linked
        linked_ids = [lc.chunk_id for lc in linked]
        assert "c1" not in linked_ids

    def test_enrich_max_linked_limit(self):
        """Should not exceed max_linked count."""
        sqlite = MagicMock()
        sqlite.get_linked_chunks.return_value = [
            {
                "linked_chunk_id": f"link-{i}",
                "linked_modality": "image",
                "link_type": "SAME_PAGE",
                "strength": 0.9,
            }
            for i in range(10)
        ]
        gold_results = [
            (make_candidate(chunk_id="c1", text="text"), 0.95),
        ]
        gold, linked = enrich_with_links(gold_results, sqlite, max_linked=2)
        assert len(linked) <= 2

    def test_enrich_no_links(self):
        sqlite = MagicMock()
        sqlite.get_linked_chunks.return_value = []
        gold_results = [
            (make_candidate(chunk_id="c1", text="text"), 0.95),
        ]
        gold, linked = enrich_with_links(gold_results, sqlite)
        assert len(linked) == 0
        assert len(gold) == 1


# ═══════════════════════════════════════════════════════════════════════════
#  Parallel Retrieval Tests (mocked)
# ═══════════════════════════════════════════════════════════════════════════
class TestParallelRetrieval:
    """Tests for run_parallel_retrieval with mocked stores."""

    def test_parallel_retrieval_basic(self):
        milvus = make_mock_milvus()
        tantivy = make_mock_tantivy()
        routing = classify_query("What is Q3 revenue?")
        embeddings = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }

        candidates, latencies = run_parallel_retrieval(
            query="What is Q3 revenue?",
            embeddings=embeddings,
            routing=routing,
            milvus_store=milvus,
            tantivy_index=tantivy,
        )
        # Should get candidates from all sources
        assert len(candidates) > 0
        # Should have latency tracking
        assert "total_retrieval" in latencies

    def test_parallel_retrieval_no_clip_embedding(self):
        """If CLIP embedding is None, image search should be skipped."""
        milvus = make_mock_milvus()
        tantivy = make_mock_tantivy()
        routing = classify_query("some query")
        embeddings = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": None,
        }

        candidates, _ = run_parallel_retrieval(
            query="some query",
            embeddings=embeddings,
            routing=routing,
            milvus_store=milvus,
            tantivy_index=tantivy,
        )
        # Should still get text and audio results
        sources = {c.source for c in candidates}
        assert "milvus_image" not in sources

    def test_parallel_retrieval_no_bge_embedding(self):
        """If BGE-M3 embedding is None, text+audio search should be skipped."""
        milvus = make_mock_milvus()
        tantivy = make_mock_tantivy()
        routing = classify_query("some query")
        embeddings = {
            "bge_m3": None,
            "clip": [0.1] * CLIP_DIM,
        }

        candidates, _ = run_parallel_retrieval(
            query="some query",
            embeddings=embeddings,
            routing=routing,
            milvus_store=milvus,
            tantivy_index=tantivy,
        )
        sources = {c.source for c in candidates}
        assert "milvus_text" not in sources
        assert "milvus_audio" not in sources

    def test_parallel_retrieval_modality_mapping(self):
        """Check that modalities are correctly assigned to candidates."""
        milvus = make_mock_milvus()
        tantivy = make_mock_tantivy()
        routing = classify_query("test query")
        embeddings = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }

        candidates, _ = run_parallel_retrieval(
            query="test query",
            embeddings=embeddings,
            routing=routing,
            milvus_store=milvus,
            tantivy_index=tantivy,
        )
        modalities = {c.modality for c in candidates}
        # Should have text, image, and audio (+ BM25 text)
        assert "text" in modalities
        assert "image" in modalities
        assert "audio" in modalities

    def test_parallel_retrieval_with_session_filter(self):
        """Session ID should be passed to search methods."""
        milvus = make_mock_milvus()
        tantivy = make_mock_tantivy()
        routing = classify_query("query")
        embeddings = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }

        run_parallel_retrieval(
            query="query",
            embeddings=embeddings,
            routing=routing,
            milvus_store=milvus,
            tantivy_index=tantivy,
            session_id="test-session",
        )
        # Verify session_id was passed
        milvus.search_text.assert_called_once()
        call_args = milvus.search_text.call_args
        assert call_args[1]["session_id"] == "test-session"


# ═══════════════════════════════════════════════════════════════════════════
#  Full Pipeline Tests (mocked)
# ═══════════════════════════════════════════════════════════════════════════
class TestRetrievalEnginePipeline:
    """Integration tests for the full RetrievalEngine pipeline."""

    def _make_engine(self):
        """Create a RetrievalEngine with mocked stores."""
        engine = RetrievalEngine(
            milvus_store=make_mock_milvus(),
            tantivy_index=make_mock_tantivy(),
            sqlite_store=make_mock_sqlite(),
        )
        engine._initialized = True
        return engine

    @patch("modules.retrieval.retrieval_engine.encode_query")
    @patch("modules.retrieval.retrieval_engine.rerank_candidates")
    def test_full_pipeline_basic(self, mock_rerank, mock_encode):
        """Test the full pipeline with mocked encoding and reranking."""
        mock_encode.return_value = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }
        mock_rerank.return_value = [
            (make_candidate(chunk_id="c1", text="result 1"), 0.95),
            (make_candidate(chunk_id="c2", text="result 2"), 0.85),
        ]

        engine = self._make_engine()
        result = engine.retrieve("What is Q3 revenue?")

        assert isinstance(result, RetrievalResult)
        assert result.query == "What is Q3 revenue?"
        assert len(result.gold_chunks) > 0
        assert "total" in result.latency_ms

    @patch("modules.retrieval.retrieval_engine.encode_query")
    def test_empty_query_returns_empty(self, mock_encode):
        engine = self._make_engine()
        result = engine.retrieve("")
        assert result.gold_chunks == []
        mock_encode.assert_not_called()

    @patch("modules.retrieval.retrieval_engine.encode_query")
    def test_whitespace_query_returns_empty(self, mock_encode):
        engine = self._make_engine()
        result = engine.retrieve("   ")
        assert result.gold_chunks == []
        mock_encode.assert_not_called()

    @patch("modules.retrieval.retrieval_engine.encode_query")
    def test_all_embeddings_fail(self, mock_encode):
        """If all embeddings fail, should return empty result."""
        mock_encode.return_value = {"bge_m3": None, "clip": None}
        engine = self._make_engine()
        result = engine.retrieve("test query")
        assert result.gold_chunks == []
        assert result.total_candidates == 0

    @patch("modules.retrieval.retrieval_engine.encode_query")
    @patch("modules.retrieval.retrieval_engine.rerank_candidates")
    def test_pipeline_without_reranking(self, mock_rerank, mock_encode):
        """Test pipeline with reranking disabled."""
        mock_encode.return_value = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }

        engine = self._make_engine()
        result = engine.retrieve(
            "test query", enable_reranking=False
        )
        # Reranker should not be called
        mock_rerank.assert_not_called()
        # Should still have results via RRF
        assert isinstance(result, RetrievalResult)

    @patch("modules.retrieval.retrieval_engine.encode_query")
    @patch("modules.retrieval.retrieval_engine.rerank_candidates")
    def test_pipeline_without_link_enrichment(self, mock_rerank, mock_encode):
        """Test pipeline with link enrichment disabled."""
        mock_encode.return_value = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }
        mock_rerank.return_value = [
            (make_candidate(chunk_id="c1", text="text"), 0.9),
        ]

        engine = self._make_engine()
        result = engine.retrieve(
            "test", enable_link_enrichment=False
        )
        assert isinstance(result, RetrievalResult)
        # Linked chunks should be empty when enrichment disabled
        assert len(result.linked_chunks) == 0

    @patch("modules.retrieval.retrieval_engine.encode_query")
    @patch("modules.retrieval.retrieval_engine.rerank_candidates")
    def test_routing_preserved_in_result(self, mock_rerank, mock_encode):
        mock_encode.return_value = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }
        mock_rerank.return_value = [
            (make_candidate(chunk_id="c1", text="text"), 0.9),
        ]

        engine = self._make_engine()
        result = engine.retrieve("show me the chart")
        assert result.routing["force_image"] is True
        assert result.query_modality == "image"

    @patch("modules.retrieval.retrieval_engine.encode_query")
    @patch("modules.retrieval.retrieval_engine.rerank_candidates")
    def test_co_retrieved_links_updated(self, mock_rerank, mock_encode):
        """CO_RETRIEVED links should be updated after retrieval."""
        mock_encode.return_value = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }
        mock_rerank.return_value = [
            (make_candidate(chunk_id="c1", text="t1"), 0.9),
            (make_candidate(chunk_id="c2", text="t2"), 0.8),
        ]

        engine = self._make_engine()
        result = engine.retrieve("test query")

        # Verify update_co_retrieved_links was called
        engine._sqlite.update_co_retrieved_links.assert_called_once()

    @patch("modules.retrieval.retrieval_engine.encode_query")
    @patch("modules.retrieval.retrieval_engine.rerank_candidates")
    def test_latency_tracking(self, mock_rerank, mock_encode):
        mock_encode.return_value = {
            "bge_m3": [0.1] * BGE_M3_DIM,
            "clip": [0.1] * CLIP_DIM,
        }
        mock_rerank.return_value = [
            (make_candidate(chunk_id="c1", text="text"), 0.9),
        ]

        engine = self._make_engine()
        result = engine.retrieve("test")
        assert "total" in result.latency_ms
        assert "encoding" in result.latency_ms
        assert result.latency_ms["total"] >= 0

    def test_initialize_calls_all_stores(self):
        milvus = make_mock_milvus()
        tantivy = make_mock_tantivy()
        sqlite = make_mock_sqlite()

        engine = RetrievalEngine(milvus, tantivy, sqlite)
        engine.initialize()

        milvus.initialize.assert_called_once()
        tantivy.initialize.assert_called_once()
        sqlite.initialize.assert_called_once()

    @patch("modules.retrieval.retrieval_engine._unload_reranker")
    def test_release_models(self, mock_unload):
        engine = self._make_engine()
        engine.release_models()
        mock_unload.assert_called_once()
        engine._milvus.release_models.assert_called_once()


# ═══════════════════════════════════════════════════════════════════════════
#  Encode Query Tests (mocked)
# ═══════════════════════════════════════════════════════════════════════════
class TestEncodeQuery:
    """Tests for query encoding."""

    def test_encode_returns_bge_embedding(self):
        """BGE-M3 embedding should always be computed via MilvusStore."""
        milvus = make_mock_milvus()
        result = encode_query("test query", milvus)
        assert result["bge_m3"] is not None
        assert len(result["bge_m3"]) == BGE_M3_DIM

    def test_encode_bge_failure_returns_none(self):
        milvus = make_mock_milvus()
        milvus.compute_query_embedding.side_effect = RuntimeError("fail")
        result = encode_query("test", milvus)
        assert result["bge_m3"] is None
