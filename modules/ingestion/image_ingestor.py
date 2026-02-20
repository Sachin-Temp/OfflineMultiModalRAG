"""
modules/ingestion/image_ingestor.py

Handles ingestion of images into ImageChunk objects.

Supports two input modes:
  A) Standalone image files (PNG, JPG, JPEG, WEBP, BMP, TIFF)
  B) Images embedded inside PDF files (extracted via PyMuPDF)

Pipeline per image:
  1. Load and validate image (Pillow)
  2. Resize/normalize for processing
  3. OCR Layer — EasyOCR on CPU → produces searchable text for BM25
  4. Visual Embedding Layer — CLIP ViT-B/32 → produces 512-dim vector
     for semantic image retrieval (text queries can find images with
     no OCR match via CLIP's native text-image alignment)
  5. Thumbnail generation — 256x256 saved to THUMB_DIR
  6. Return ImageChunk with all metadata populated

CRITICAL: CLIP embeddings are 512-dim and stored in a SEPARATE Milvus
collection from text (1024-dim BGE-M3). Do NOT project or merge spaces.
The embedding field on ImageChunk is populated HERE (unlike TextChunk
where embedding is filled by the indexing phase) because CLIP must be
loaded for image processing anyway — doing it here avoids a second
model load during indexing.

No Milvus/Tantivy/SQLite writes happen here — that is Phase 4's job.
"""

import io
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Any

import fitz                          # PyMuPDF — for PDF image extraction
import numpy as np
from PIL import Image, UnidentifiedImageError
import magic
from loguru import logger

from config.settings import (
    MEDIA_DIR,
    THUMB_DIR,
    CLIP_DIM,
    UPLOAD_DIR,
)
from core.vram_manager import vram_manager
from models.schemas import ImageChunk, BBox, Modality, IngestionResult


# ── Constants ──────────────────────────────────────────────────────────────
THUMBNAIL_SIZE       = (256, 256)
MIN_IMAGE_DIMENSION  = 32      # px — images smaller than this are skipped
MAX_IMAGE_DIMENSION  = 4096    # px — images larger are downscaled before OCR
OCR_DPI_SCALE        = 2       # upscale factor for OCR quality on small images

SUPPORTED_IMAGE_MIMES = {
    "image/png":  ".png",
    "image/jpeg": ".jpg",
    "image/jpg":  ".jpg",
    "image/webp": ".webp",
    "image/bmp":  ".bmp",
    "image/tiff": ".tiff",
}

# Global model references — loaded lazily, released via VRAMManager
_clip_model      = None
_clip_preprocess = None
_clip_device     = None
_easyocr_reader  = None


# ── Device Detection ───────────────────────────────────────────────────────
def _get_clip_device() -> str:
    """Return 'cuda' if available, else 'cpu'."""
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


# ── Model Loading ──────────────────────────────────────────────────────────
def _load_clip() -> Tuple[Any, Any, str]:
    """
    Load CLIP ViT-B/32 model with VRAM management.
    Returns (model, preprocess, device).
    Acquires VRAM slot before loading — raises RuntimeError if VRAM unavailable.
    """
    global _clip_model, _clip_preprocess, _clip_device

    if _clip_model is not None:
        logger.debug("CLIP already loaded — reusing cached instance")
        return _clip_model, _clip_preprocess, _clip_device

    # Register eviction callback so VRAMManager can unload CLIP if needed
    def _evict_clip():
        global _clip_model, _clip_preprocess, _clip_device
        import torch
        if _clip_model is not None:
            del _clip_model
            del _clip_preprocess
            _clip_model      = None
            _clip_preprocess = None
            _clip_device     = None
            torch.cuda.empty_cache()
            logger.info("CLIP evicted from VRAM by VRAMManager")

    vram_manager.register_evict_callback("clip", _evict_clip)

    acquired = vram_manager.acquire("clip")
    if not acquired:
        raise RuntimeError(
            "VRAMManager could not allocate VRAM for CLIP. "
            "Ensure LLM is not loaded before running image ingestion."
        )

    try:
        import clip
        device = _get_clip_device()
        logger.info(f"Loading CLIP ViT-B/32 on {device}...")
        model, preprocess = clip.load("ViT-B/32", device=device)
        model.eval()
        _clip_model      = model
        _clip_preprocess = preprocess
        _clip_device     = device
        logger.info(f"CLIP ViT-B/32 loaded on {device}")
        return model, preprocess, device
    except Exception as e:
        vram_manager.release("clip")
        raise RuntimeError(f"Failed to load CLIP: {e}") from e


def _unload_clip():
    """
    Explicitly unload CLIP and release VRAM.
    Call this after finishing a batch of image ingestion.
    """
    global _clip_model, _clip_preprocess, _clip_device
    import torch
    if _clip_model is not None:
        del _clip_model
        del _clip_preprocess
        _clip_model      = None
        _clip_preprocess = None
        _clip_device     = None
        torch.cuda.empty_cache()
        vram_manager.release("clip")
        logger.info("CLIP explicitly unloaded and VRAM released")


def _load_easyocr() -> Any:
    """
    Load EasyOCR reader. Always runs on CPU — never acquires VRAM.
    Cached after first load.
    """
    global _easyocr_reader
    if _easyocr_reader is not None:
        return _easyocr_reader

    import easyocr
    logger.info("Loading EasyOCR (CPU mode)...")
    # gpu=False is mandatory — VRAM is reserved for LLM and CLIP
    _easyocr_reader = easyocr.Reader(["en"], gpu=False, verbose=False)
    logger.info("EasyOCR loaded on CPU")
    return _easyocr_reader


# ── Image Validation & Loading ─────────────────────────────────────────────
def _load_and_validate_image(image_bytes: bytes) -> Optional[Image.Image]:
    """
    Load image from bytes using Pillow. Validate minimum dimensions.
    Convert to RGB (handles RGBA, grayscale, palette modes).
    Returns PIL Image or None if invalid.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes))
        img.verify()                       # Check file integrity
        img = Image.open(io.BytesIO(image_bytes))  # Re-open after verify
        img = img.convert("RGB")
    except (UnidentifiedImageError, Exception) as e:
        logger.warning(f"Image load/validation failed: {e}")
        return None

    w, h = img.size
    if w < MIN_IMAGE_DIMENSION or h < MIN_IMAGE_DIMENSION:
        logger.debug(f"Image too small ({w}x{h}) — skipping")
        return None

    return img


def _resize_for_ocr(img: Image.Image) -> Image.Image:
    """
    If image is very small, upscale for better OCR accuracy.
    If image is very large, downscale to avoid memory issues.
    Maintains aspect ratio.
    """
    w, h = img.size

    # Upscale small images for better OCR
    if max(w, h) < 300:
        scale = OCR_DPI_SCALE
        img = img.resize((w * scale, h * scale), Image.LANCZOS)

    # Downscale very large images
    if max(w, h) > MAX_IMAGE_DIMENSION:
        ratio = MAX_IMAGE_DIMENSION / max(w, h)
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        img = img.resize((new_w, new_h), Image.LANCZOS)

    return img


# ── OCR ────────────────────────────────────────────────────────────────────
def _run_ocr(img: Image.Image) -> str:
    """
    Run EasyOCR on a PIL Image. Returns extracted text as a single string.
    Empty string if no text detected.
    Always runs on CPU.
    """
    try:
        reader = _load_easyocr()
        ocr_img = _resize_for_ocr(img)
        img_array = np.array(ocr_img)

        # detail=0 returns text strings only (no bounding boxes)
        # paragraph=True merges nearby text blocks into paragraphs
        results = reader.readtext(img_array, detail=0, paragraph=True)
        text = " ".join(str(r).strip() for r in results if str(r).strip())
        logger.debug(f"OCR extracted {len(text)} chars")
        return text
    except Exception as e:
        logger.warning(f"OCR failed: {e}")
        return ""


# ── CLIP Embedding ─────────────────────────────────────────────────────────
def _compute_clip_embedding(img: Image.Image) -> Optional[List[float]]:
    """
    Compute 512-dim CLIP visual embedding for a PIL Image.
    Returns list of 512 floats (L2-normalized, ready for COSINE similarity).
    Returns None if CLIP fails.

    IMPORTANT: This embedding is in CLIP's native 512-dim space.
    Do NOT compare with BGE-M3 1024-dim embeddings directly.
    Cross-modal text-image search uses CLIP's text encoder on the query,
    not BGE-M3, to stay in the same 512-dim CLIP space.
    """
    try:
        import torch

        model, preprocess, device = _load_clip()

        # Preprocess image to CLIP's expected format
        image_tensor = preprocess(img).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.encode_image(image_tensor)
            # L2-normalize so cosine similarity = dot product
            embedding = embedding / embedding.norm(dim=-1, keepdim=True)
            embedding_list = embedding.squeeze(0).cpu().numpy().tolist()

        assert len(embedding_list) == CLIP_DIM, \
            f"Expected {CLIP_DIM}-dim CLIP embedding, got {len(embedding_list)}"

        logger.debug(f"CLIP embedding computed: {CLIP_DIM}-dim vector")
        return embedding_list

    except Exception as e:
        logger.error(f"CLIP embedding failed: {e}")
        return None


def _compute_clip_text_embedding(text: str) -> Optional[List[float]]:
    """
    Compute 512-dim CLIP TEXT embedding for a query string.
    Used during retrieval to find images semantically matching a text query.
    Both this and _compute_clip_embedding output are in the same CLIP space,
    so cosine similarity between them is directly meaningful.

    This function is used by the retrieval engine (Phase 6), not during ingest.
    Defined here to keep all CLIP operations in one place.
    """
    try:
        import torch
        import clip as clip_module

        model, preprocess, device = _load_clip()
        text_tokens = clip_module.tokenize([text], truncate=True).to(device)

        with torch.no_grad():
            text_embedding = model.encode_text(text_tokens)
            text_embedding = text_embedding / text_embedding.norm(dim=-1, keepdim=True)
            embedding_list = text_embedding.squeeze(0).cpu().numpy().tolist()

        assert len(embedding_list) == CLIP_DIM
        return embedding_list

    except Exception as e:
        logger.error(f"CLIP text embedding failed: {e}")
        return None


# ── Thumbnail Generation ───────────────────────────────────────────────────
def _generate_thumbnail(img: Image.Image, chunk_id: str) -> Optional[str]:
    """
    Generate a 256x256 thumbnail and save to THUMB_DIR.
    Returns relative path string (relative to project root) or None on failure.
    Thumbnail filename is based on chunk_id for uniqueness.
    """
    try:
        thumb = img.copy()
        thumb.thumbnail(THUMBNAIL_SIZE, Image.LANCZOS)

        # Pad to exact 256x256 with white background (for non-square images)
        padded = Image.new("RGB", THUMBNAIL_SIZE, (255, 255, 255))
        offset = (
            (THUMBNAIL_SIZE[0] - thumb.width)  // 2,
            (THUMBNAIL_SIZE[1] - thumb.height) // 2,
        )
        padded.paste(thumb, offset)

        thumb_filename = f"{chunk_id}_thumb.jpg"
        thumb_path = THUMB_DIR / thumb_filename
        padded.save(str(thumb_path), "JPEG", quality=85, optimize=True)

        # Return path relative to project root for portability
        relative_path = f"data/media/thumbs/{thumb_filename}"
        logger.debug(f"Thumbnail saved: {relative_path}")
        return relative_path

    except Exception as e:
        logger.warning(f"Thumbnail generation failed: {e}")
        return None


# ── Image Saving ───────────────────────────────────────────────────────────
def _save_image(img: Image.Image, filename: str) -> Optional[str]:
    """
    Save a PIL Image to MEDIA_DIR.
    Returns relative path string or None on failure.
    """
    try:
        save_path = MEDIA_DIR / filename
        img.save(str(save_path), "PNG", optimize=True)
        relative_path = f"data/media/{filename}"
        logger.debug(f"Image saved: {relative_path}")
        return relative_path
    except Exception as e:
        logger.warning(f"Image save failed for {filename}: {e}")
        return None


# ── Single Image Processing ────────────────────────────────────────────────
def _process_single_image(
    image_bytes: bytes,
    source_file: str,
    page_number: int,
    session_id: Optional[str],
    bbox: Optional[BBox] = None,
    filename_hint: Optional[str] = None,
) -> Optional[ImageChunk]:
    """
    Core processing pipeline for a single image (given as raw bytes).
    Steps: validate -> OCR -> CLIP embed -> thumbnail -> save -> build ImageChunk.

    Args:
        image_bytes:   Raw image bytes (PNG, JPG, etc.)
        source_file:   Name of the source file (PDF name or image filename)
        page_number:   Page number in PDF (1-indexed), or 1 for standalone images
        session_id:    Session identifier
        bbox:          Bounding box within the source PDF page (if from PDF)
        filename_hint: Preferred filename for saving (without extension)

    Returns:
        ImageChunk with all fields populated, or None if image is invalid.
    """
    img = _load_and_validate_image(image_bytes)
    if img is None:
        return None

    chunk_id = str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()

    # Determine save filename
    safe_source = Path(source_file).stem.replace(" ", "_")
    if filename_hint:
        img_filename = f"{filename_hint}.png"
    else:
        short_id = chunk_id[:8]
        img_filename = f"{safe_source}_p{page_number}_{short_id}.png"

    # 1. Save original image
    image_path = _save_image(img, img_filename)

    # 2. Generate thumbnail
    thumbnail_path = _generate_thumbnail(img, chunk_id)

    # 3. OCR — always on CPU
    ocr_text = _run_ocr(img)

    # 4. CLIP visual embedding — 512-dim, uses GPU if available
    embedding = _compute_clip_embedding(img)

    # Build ImageChunk
    chunk = ImageChunk(
        chunk_id=chunk_id,
        modality=Modality.IMAGE,
        image_path=image_path or "",
        thumbnail_path=thumbnail_path,
        ocr_text=ocr_text if ocr_text else None,
        llm_description=None,   # filled later by LLM during ingest (Phase 7)
        bbox=bbox,
        source_file=source_file,
        page_number=page_number,
        ingest_timestamp=now,
        session_id=session_id,
        embedding=embedding,    # 512-dim CLIP vector (or None if CLIP failed)
    )

    logger.debug(
        f"ImageChunk created: {chunk_id[:8]}... | "
        f"page={page_number} | "
        f"ocr_chars={len(ocr_text)} | "
        f"has_embedding={embedding is not None}"
    )
    return chunk


# ── PDF Image Extraction ───────────────────────────────────────────────────
def _extract_images_from_pdf(
    pdf_path: Path,
    session_id: Optional[str],
) -> Tuple[List[ImageChunk], List[str]]:
    """
    Extract all embedded images from a PDF file.
    Uses PyMuPDF to iterate pages and extract image xrefs.

    Strategy:
    - For each page, get list of image references via page.get_images(full=True)
    - Extract raw image bytes via doc.extract_image(xref)
    - Skip images smaller than MIN_IMAGE_DIMENSION
    - Skip duplicate images (same xref already processed)
    - Process each valid image through _process_single_image()

    Returns:
        (list of ImageChunk objects, list of warning strings)
    """
    chunks: List[ImageChunk] = []
    warnings: List[str] = []
    processed_xrefs = set()   # avoid processing duplicate embedded images

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as e:
        return [], [f"PyMuPDF failed to open PDF for image extraction: {e}"]

    logger.info(f"Extracting images from PDF: {pdf_path.name} ({len(doc)} pages)")

    for page_idx in range(len(doc)):
        page = doc[page_idx]
        page_num = page_idx + 1   # 1-indexed

        image_list = page.get_images(full=True)
        if not image_list:
            continue

        logger.debug(f"Page {page_num}: found {len(image_list)} image reference(s)")

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]   # xref is the unique image reference in the PDF

            if xref in processed_xrefs:
                logger.debug(f"Skipping duplicate xref {xref}")
                continue
            processed_xrefs.add(xref)

            try:
                extracted = doc.extract_image(xref)
                if not extracted:
                    continue

                image_bytes = extracted["image"]
                img_width   = extracted.get("width", 0)
                img_height  = extracted.get("height", 0)

                # Skip tiny images (likely icons, bullets, decorations)
                if img_width < MIN_IMAGE_DIMENSION or img_height < MIN_IMAGE_DIMENSION:
                    logger.debug(
                        f"Skipping tiny image on page {page_num}: "
                        f"{img_width}x{img_height}px"
                    )
                    continue

                # Get image bounding box on the page
                # get_image_rects returns list of Rect objects for this xref
                rects = page.get_image_rects(xref)
                bbox = None
                if rects:
                    r = rects[0]
                    bbox = BBox(x0=r.x0, y0=r.y0, x1=r.x1, y1=r.y1)

                # Build filename hint
                safe_stem = Path(pdf_path).stem.replace(" ", "_")
                filename_hint = f"{safe_stem}_p{page_num}_img{img_idx+1}_{xref}"

                chunk = _process_single_image(
                    image_bytes=image_bytes,
                    source_file=pdf_path.name,
                    page_number=page_num,
                    session_id=session_id,
                    bbox=bbox,
                    filename_hint=filename_hint,
                )

                if chunk is not None:
                    chunks.append(chunk)
                else:
                    warnings.append(
                        f"Page {page_num}, image {img_idx+1}: "
                        f"skipped (invalid or too small after loading)"
                    )

            except Exception as e:
                warnings.append(f"Page {page_num}, image {img_idx+1}: extraction error — {e}")
                logger.warning(f"Image extraction error on page {page_num}: {e}")

    doc.close()
    logger.info(
        f"PDF image extraction complete: {pdf_path.name} -> "
        f"{len(chunks)} image chunks, {len(warnings)} warnings"
    )
    return chunks, warnings


# ── Standalone Image Ingestion ─────────────────────────────────────────────
def _ingest_standalone_image(
    file_path: Path,
    session_id: Optional[str],
) -> Tuple[List[ImageChunk], List[str]]:
    """
    Ingest a single standalone image file (not from a PDF).
    Returns (list with one ImageChunk, list of warnings).
    """
    warnings: List[str] = []

    try:
        image_bytes = file_path.read_bytes()
    except Exception as e:
        return [], [f"Failed to read image file {file_path.name}: {e}"]

    chunk = _process_single_image(
        image_bytes=image_bytes,
        source_file=file_path.name,
        page_number=1,        # standalone images have no page concept
        session_id=session_id,
        bbox=None,
        filename_hint=None,   # will use auto-generated name
    )

    if chunk is None:
        return [], [f"Image {file_path.name} could not be processed (invalid or too small)"]

    logger.info(f"Standalone image ingested: {file_path.name} -> 1 image chunk")
    return [chunk], warnings


# ── File Type Detection ────────────────────────────────────────────────────
def _detect_mime_type(file_path: Path) -> str:
    """Detect MIME type using libmagic. Fallback to extension."""
    try:
        return magic.from_file(str(file_path), mime=True)
    except Exception as e:
        logger.warning(f"python-magic failed ({e}), using extension fallback")
        suffix = file_path.suffix.lower()
        fallback = {
            ".png":  "image/png",
            ".jpg":  "image/jpeg",
            ".jpeg": "image/jpeg",
            ".webp": "image/webp",
            ".bmp":  "image/bmp",
            ".tiff": "image/tiff",
            ".tif":  "image/tiff",
            ".pdf":  "application/pdf",
        }
        return fallback.get(suffix, "application/octet-stream")


# ── Public API ─────────────────────────────────────────────────────────────
class ImageIngestor:
    """
    Public interface for image ingestion.

    Handles two modes:
      1. Standalone image files -> ingest(file_path)
      2. Images embedded in PDFs -> ingest_from_pdf(pdf_path)

    Usage:
        ingestor = ImageIngestor()

        # Standalone image
        result = ingestor.ingest(Path("photo.png"), session_id="abc")
        # result.image_chunks contains list of ImageChunk objects

        # Images from PDF (call AFTER text_ingestor.ingest for same PDF)
        result = ingestor.ingest_from_pdf(Path("report.pdf"), session_id="abc")
        # result.image_chunks contains one ImageChunk per embedded image

    VRAM Note:
        CLIP is loaded on first call and kept loaded for batch efficiency.
        Call ingestor.release_models() after finishing a batch to free VRAM
        before the LLM loads.
    """

    SUPPORTED_IMAGE_MIMES = set(SUPPORTED_IMAGE_MIMES.keys())
    PDF_MIME = "application/pdf"

    def ingest(
        self,
        file_path: Path,
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        Ingest a standalone image file.

        Args:
            file_path:  Path to image file (PNG, JPG, JPEG, WEBP, BMP, TIFF)
            session_id: Optional session identifier

        Returns:
            IngestionResult with image_chunks populated.
        """
        file_path = Path(file_path).resolve()
        result = IngestionResult(
            source_file=file_path.name,
            modality=Modality.IMAGE,
        )

        if not file_path.exists():
            result.errors.append(f"File not found: {file_path}")
            logger.error(f"ImageIngestor: File not found: {file_path}")
            return result

        mime_type = _detect_mime_type(file_path)

        if mime_type not in self.SUPPORTED_IMAGE_MIMES:
            # Fallback: check extension for common image types
            ext = file_path.suffix.lower()
            if ext in (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tiff", ".tif"):
                logger.debug(f"MIME type '{mime_type}' not recognized, but extension '{ext}' is valid — proceeding")
            else:
                result.errors.append(
                    f"Unsupported file type '{mime_type}' for {file_path.name}. "
                    f"Supported: PNG, JPG, JPEG, WEBP, BMP, TIFF."
                )
                return result

        logger.info(
            f"ImageIngestor: Ingesting standalone image "
            f"{file_path.name} (session={session_id})"
        )

        try:
            chunks, warnings = _ingest_standalone_image(file_path, session_id)
            result.image_chunks = chunks
            result.warnings     = warnings
            if not chunks:
                result.errors.append(f"No image chunks produced from {file_path.name}")
        except Exception as e:
            result.errors.append(f"Unexpected error: {e}")
            logger.exception(f"ImageIngestor: Error on {file_path.name}")

        logger.info(
            f"ImageIngestor: {file_path.name} complete — "
            f"{len(result.image_chunks)} chunks, "
            f"{len(result.warnings)} warnings, "
            f"{len(result.errors)} errors"
        )
        return result

    def ingest_from_pdf(
        self,
        pdf_path: Path,
        session_id: Optional[str] = None,
    ) -> IngestionResult:
        """
        Extract and ingest all images embedded in a PDF file.
        This is called in addition to TextIngestor.ingest() for the same PDF —
        the two ingestors handle different aspects of the same file.

        Args:
            pdf_path:   Path to PDF file
            session_id: Optional session identifier

        Returns:
            IngestionResult with image_chunks populated.
        """
        pdf_path = Path(pdf_path).resolve()
        result = IngestionResult(
            source_file=pdf_path.name,
            modality=Modality.IMAGE,
        )

        if not pdf_path.exists():
            result.errors.append(f"File not found: {pdf_path}")
            return result

        mime_type = _detect_mime_type(pdf_path)
        if mime_type != self.PDF_MIME:
            # Extension fallback for PDF
            if pdf_path.suffix.lower() == ".pdf":
                logger.debug(f"MIME type '{mime_type}' but extension is .pdf — proceeding")
            else:
                result.errors.append(
                    f"ingest_from_pdf() requires a PDF file. "
                    f"Got MIME type: {mime_type}"
                )
                return result

        logger.info(
            f"ImageIngestor: Extracting images from PDF "
            f"{pdf_path.name} (session={session_id})"
        )

        try:
            chunks, warnings = _extract_images_from_pdf(pdf_path, session_id)
            result.image_chunks = chunks
            result.warnings     = warnings
        except Exception as e:
            result.errors.append(f"Unexpected error during PDF image extraction: {e}")
            logger.exception(f"ImageIngestor: Error extracting from {pdf_path.name}")

        logger.info(
            f"ImageIngestor: PDF image extraction {pdf_path.name} complete — "
            f"{len(result.image_chunks)} image chunks"
        )
        return result

    def ingest_batch(
        self,
        file_paths: List[Path],
        session_id: Optional[str] = None,
    ) -> List[IngestionResult]:
        """
        Ingest multiple image files sequentially.
        CLIP is loaded once and reused across all files in the batch.
        Call release_models() when done to free VRAM.

        Returns list of IngestionResult, one per file.
        """
        results = []
        for fp in file_paths:
            results.append(self.ingest(Path(fp), session_id))
        return results

    def release_models(self):
        """
        Explicitly unload CLIP from GPU and release VRAM.
        MUST be called after finishing image ingestion and before LLM loads.
        The VRAMManager will also trigger this automatically via LRU eviction
        if needed, but explicit release is preferred for predictable behavior.
        """
        _unload_clip()
        logger.info("ImageIngestor: Models released")

    def get_clip_text_embedding(self, text: str) -> Optional[List[float]]:
        """
        Public accessor for CLIP text embeddings.
        Used by the retrieval engine (Phase 6) to encode text queries
        into CLIP's 512-dim space for image search.

        Args:
            text: Query string to encode

        Returns:
            512-dim list of floats (L2-normalized) or None on failure
        """
        return _compute_clip_text_embedding(text)
