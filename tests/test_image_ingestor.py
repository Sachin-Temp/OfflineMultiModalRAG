"""
tests/test_image_ingestor.py
Full test suite for the ImageIngestor module.

Tests cover:
- Standalone image ingestion (valid image)
- PDF image extraction
- Image validation (too small, corrupted)
- OCR output
- CLIP embedding dimensions and normalization
- Thumbnail generation
- Metadata completeness
- VRAMManager interaction
- Unsupported file type rejection
- Missing file handling

Run with: pytest tests/test_image_ingestor.py -v
Note: CLIP tests require the CLIP model to be downloaded (run download_models.py first).
      Tests that require CLIP are marked with @pytest.mark.clip
"""

import io
import pytest
import numpy as np
from pathlib import Path
from PIL import Image


# ── Helpers ────────────────────────────────────────────────────────────────
def _make_test_image_bytes(
    width: int = 200,
    height: int = 200,
    color: tuple = (100, 150, 200),
    text: str = None,
) -> bytes:
    """
    Create a minimal test image as bytes.
    Optionally draws text on it for OCR testing.
    """
    img = Image.new("RGB", (width, height), color=color)

    if text:
        try:
            from PIL import ImageDraw
            draw = ImageDraw.Draw(img)
            # Use default font (no external font file needed)
            draw.text((10, 10), text, fill=(0, 0, 0))
        except Exception:
            pass   # text drawing is best-effort for tests

    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _make_test_pdf_with_image(tmp_path: Path) -> Path:
    """Create a minimal PDF with one embedded image using PyMuPDF."""
    import fitz
    pdf_path = tmp_path / "test_with_image.pdf"
    doc = fitz.open()
    page = doc.new_page(width=595, height=842)

    # Add some text
    page.insert_text((72, 72), "This is a test PDF with an embedded image.", fontsize=12)

    # Create and insert a small image
    img_bytes = _make_test_image_bytes(150, 150, color=(200, 100, 50))
    img_rect = fitz.Rect(72, 100, 250, 280)
    page.insert_image(img_rect, stream=img_bytes)

    doc.save(str(pdf_path))
    doc.close()
    return pdf_path


# ── Image Loading & Validation Tests ──────────────────────────────────────
class TestImageValidation:

    def test_valid_image_loads(self):
        from modules.ingestion.image_ingestor import _load_and_validate_image
        img_bytes = _make_test_image_bytes(200, 200)
        img = _load_and_validate_image(img_bytes)
        assert img is not None
        assert img.mode == "RGB"

    def test_too_small_image_rejected(self):
        from modules.ingestion.image_ingestor import _load_and_validate_image
        img_bytes = _make_test_image_bytes(10, 10)   # below MIN_IMAGE_DIMENSION
        img = _load_and_validate_image(img_bytes)
        assert img is None

    def test_corrupted_bytes_rejected(self):
        from modules.ingestion.image_ingestor import _load_and_validate_image
        result = _load_and_validate_image(b"not an image at all 1234567890")
        assert result is None

    def test_rgba_converted_to_rgb(self):
        from modules.ingestion.image_ingestor import _load_and_validate_image
        # Create RGBA image
        img = Image.new("RGBA", (200, 200), color=(100, 150, 200, 128))
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        result = _load_and_validate_image(buf.getvalue())
        assert result is not None
        assert result.mode == "RGB"

    def test_resize_for_ocr_small_image(self):
        from modules.ingestion.image_ingestor import _resize_for_ocr
        small_img = Image.new("RGB", (50, 50))
        resized = _resize_for_ocr(small_img)
        # Should be upscaled
        assert resized.width > 50
        assert resized.height > 50

    def test_resize_for_ocr_large_image(self):
        from modules.ingestion.image_ingestor import _resize_for_ocr, MAX_IMAGE_DIMENSION
        large_img = Image.new("RGB", (5000, 5000))
        resized = _resize_for_ocr(large_img)
        assert max(resized.width, resized.height) <= MAX_IMAGE_DIMENSION


# ── Thumbnail Tests ────────────────────────────────────────────────────────
class TestThumbnails:

    def test_thumbnail_created(self):
        from modules.ingestion.image_ingestor import _generate_thumbnail
        from config.settings import THUMB_DIR
        img = Image.new("RGB", (400, 300), color=(100, 200, 100))
        chunk_id = "test-chunk-id-0001"
        result = _generate_thumbnail(img, chunk_id)
        assert result is not None
        assert "thumb" in result
        thumb_file = THUMB_DIR / f"{chunk_id}_thumb.jpg"
        assert thumb_file.exists()

    def test_thumbnail_correct_size(self):
        from modules.ingestion.image_ingestor import _generate_thumbnail, THUMBNAIL_SIZE
        from config.settings import THUMB_DIR
        img = Image.new("RGB", (800, 600))
        chunk_id = "test-chunk-id-0002"
        _generate_thumbnail(img, chunk_id)
        thumb = Image.open(THUMB_DIR / f"{chunk_id}_thumb.jpg")
        assert thumb.size == THUMBNAIL_SIZE

    def test_thumbnail_non_square_image(self):
        """Non-square images should be padded to 256x256, not stretched."""
        from modules.ingestion.image_ingestor import _generate_thumbnail, THUMBNAIL_SIZE
        from config.settings import THUMB_DIR
        img = Image.new("RGB", (800, 200))   # very wide
        chunk_id = "test-chunk-id-0003"
        _generate_thumbnail(img, chunk_id)
        thumb = Image.open(THUMB_DIR / f"{chunk_id}_thumb.jpg")
        assert thumb.size == THUMBNAIL_SIZE


# ── OCR Tests ──────────────────────────────────────────────────────────────
class TestOCR:

    def test_ocr_returns_string(self):
        from modules.ingestion.image_ingestor import _run_ocr
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        result = _run_ocr(img)
        assert isinstance(result, str)

    def test_ocr_blank_image_returns_empty_or_string(self):
        """Blank white image should return empty string or whitespace."""
        from modules.ingestion.image_ingestor import _run_ocr
        img = Image.new("RGB", (200, 200), color=(255, 255, 255))
        result = _run_ocr(img)
        assert isinstance(result, str)
        # blank image may return "" or some noise — both are acceptable
        assert len(result) < 50


# ── CLIP Embedding Tests ───────────────────────────────────────────────────
class TestCLIPEmbedding:

    @pytest.mark.clip
    def test_clip_embedding_dimension(self):
        from modules.ingestion.image_ingestor import _compute_clip_embedding
        from config.settings import CLIP_DIM
        img = Image.new("RGB", (224, 224), color=(100, 150, 200))
        embedding = _compute_clip_embedding(img)
        assert embedding is not None
        assert len(embedding) == CLIP_DIM

    @pytest.mark.clip
    def test_clip_embedding_is_normalized(self):
        """CLIP embeddings should be L2-normalized (norm approx 1.0)."""
        from modules.ingestion.image_ingestor import _compute_clip_embedding
        img = Image.new("RGB", (224, 224), color=(50, 100, 200))
        embedding = _compute_clip_embedding(img)
        assert embedding is not None
        norm = np.linalg.norm(np.array(embedding))
        assert abs(norm - 1.0) < 0.01, f"Embedding not normalized: norm={norm}"

    @pytest.mark.clip
    def test_clip_text_embedding_dimension(self):
        from modules.ingestion.image_ingestor import _compute_clip_text_embedding
        from config.settings import CLIP_DIM
        embedding = _compute_clip_text_embedding("revenue chart quarterly results")
        assert embedding is not None
        assert len(embedding) == CLIP_DIM

    @pytest.mark.clip
    def test_clip_text_image_similarity(self):
        """
        A colored square labeled 'red square' should have higher CLIP similarity
        to the text 'red square' than to 'blue ocean'.
        This validates that CLIP's text-image alignment is working.
        """
        from modules.ingestion.image_ingestor import (
            _compute_clip_embedding,
            _compute_clip_text_embedding,
        )
        red_img = Image.new("RGB", (224, 224), color=(220, 30, 30))
        img_emb = np.array(_compute_clip_embedding(red_img))

        text_red  = np.array(_compute_clip_text_embedding("red square"))
        text_blue = np.array(_compute_clip_text_embedding("blue ocean wave"))

        sim_red  = float(np.dot(img_emb, text_red))
        sim_blue = float(np.dot(img_emb, text_blue))

        # Red image should be more similar to "red square" than "blue ocean"
        assert sim_red > sim_blue, (
            f"CLIP alignment broken: sim_red={sim_red:.3f}, sim_blue={sim_blue:.3f}"
        )


# ── VRAMManager Interaction ────────────────────────────────────────────────
class TestVRAMManager:

    @pytest.mark.clip
    def test_vram_manager_tracks_clip(self):
        from core.vram_manager import vram_manager
        from modules.ingestion.image_ingestor import _load_clip, _unload_clip
        initial_used = vram_manager.used_gb
        try:
            _load_clip()
            assert vram_manager.used_gb >= initial_used
        finally:
            _unload_clip()

    def test_release_models_clears_clip(self):
        from core.vram_manager import vram_manager
        from modules.ingestion.image_ingestor import ImageIngestor
        ingestor = ImageIngestor()
        ingestor.release_models()
        # After release, CLIP should not be in loaded models
        assert "clip" not in vram_manager.status()["loaded_models"]


# ── ImageIngestor Integration Tests ───────────────────────────────────────
class TestImageIngestor:

    @pytest.fixture
    def ingestor(self):
        from modules.ingestion.image_ingestor import ImageIngestor
        i = ImageIngestor()
        yield i
        i.release_models()   # always clean up after each test

    @pytest.fixture
    def sample_image(self, tmp_path):
        img_path = tmp_path / "test_image.png"
        img = Image.new("RGB", (300, 300), color=(80, 120, 200))
        img.save(str(img_path))
        return img_path

    @pytest.fixture
    def sample_pdf_with_image(self, tmp_path):
        return _make_test_pdf_with_image(tmp_path)

    @pytest.mark.clip
    def test_standalone_image_ingestion(self, ingestor, sample_image):
        result = ingestor.ingest(sample_image, session_id="test-sess")
        assert result.success
        assert len(result.image_chunks) == 1

    @pytest.mark.clip
    def test_image_chunk_metadata(self, ingestor, sample_image):
        result = ingestor.ingest(sample_image, session_id="test-sess")
        chunk = result.image_chunks[0]
        assert chunk.chunk_id is not None
        assert chunk.source_file == sample_image.name
        assert chunk.page_number == 1
        assert chunk.session_id == "test-sess"
        assert chunk.modality.value == "image"
        assert chunk.ingest_timestamp is not None

    @pytest.mark.clip
    def test_image_chunk_has_thumbnail(self, ingestor, sample_image):
        result = ingestor.ingest(sample_image)
        chunk = result.image_chunks[0]
        assert chunk.thumbnail_path is not None
        assert Path(chunk.thumbnail_path).name.endswith("_thumb.jpg")

    @pytest.mark.clip
    def test_image_chunk_has_image_path(self, ingestor, sample_image):
        result = ingestor.ingest(sample_image)
        chunk = result.image_chunks[0]
        assert chunk.image_path is not None
        assert chunk.image_path != ""

    @pytest.mark.clip
    def test_image_chunk_has_embedding(self, ingestor, sample_image):
        from config.settings import CLIP_DIM
        result = ingestor.ingest(sample_image)
        chunk = result.image_chunks[0]
        assert chunk.embedding is not None
        assert len(chunk.embedding) == CLIP_DIM

    @pytest.mark.clip
    def test_pdf_image_extraction(self, ingestor, sample_pdf_with_image):
        result = ingestor.ingest_from_pdf(sample_pdf_with_image, session_id="test-sess")
        assert len(result.image_chunks) >= 1

    @pytest.mark.clip
    def test_pdf_image_page_number_recorded(self, ingestor, sample_pdf_with_image):
        result = ingestor.ingest_from_pdf(sample_pdf_with_image)
        for chunk in result.image_chunks:
            assert chunk.page_number >= 1

    def test_missing_file_returns_error(self, ingestor, tmp_path):
        fake = tmp_path / "nonexistent.png"
        result = ingestor.ingest(fake)
        assert not result.success
        assert len(result.errors) > 0

    def test_unsupported_file_type_rejected(self, ingestor, tmp_path):
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("not an image")
        result = ingestor.ingest(txt_file)
        assert not result.success
        assert len(result.errors) > 0

    @pytest.mark.clip
    def test_batch_ingest(self, ingestor, tmp_path):
        images = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGB", (200, 200), color=(i*80, 100, 150)).save(str(p))
            images.append(p)

        results = ingestor.ingest_batch(images, session_id="batch-test")
        assert len(results) == 3
        assert all(r.success for r in results)

    @pytest.mark.clip
    def test_no_duplicate_chunk_ids(self, ingestor, tmp_path):
        images = []
        for i in range(3):
            p = tmp_path / f"img_{i}.png"
            Image.new("RGB", (200, 200), color=(i*50, 100, 200)).save(str(p))
            images.append(p)

        results = ingestor.ingest_batch(images)
        all_ids = [c.chunk_id for r in results for c in r.image_chunks]
        assert len(all_ids) == len(set(all_ids)), "Duplicate chunk IDs found"

    @pytest.mark.clip
    def test_get_clip_text_embedding(self, ingestor):
        embedding = ingestor.get_clip_text_embedding("revenue chart")
        assert embedding is not None
        assert len(embedding) == 512
