"""
Global configuration for Multimodal RAG System.
All constants, paths, and model settings live here.
Import this in every module instead of hardcoding values.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Paths ──────────────────────────────────────────────────────────────────
PROJECT_ROOT    = Path(os.getenv("PROJECT_ROOT", ".")).resolve()
UPLOAD_DIR      = Path(os.getenv("UPLOAD_DIR",      "./data/uploads"))
MEDIA_DIR       = Path(os.getenv("MEDIA_DIR",       "./data/media"))
THUMB_DIR       = Path(os.getenv("THUMB_DIR",        "./data/media/thumbs"))
AUDIO_DIR       = Path(os.getenv("AUDIO_DIR",        "./data/audio"))
TANTIVY_DIR     = Path(os.getenv("TANTIVY_INDEX_DIR","./data/tantivy_index"))
SQLITE_DB_PATH  = Path(os.getenv("SQLITE_DB_PATH",  "./data/sqlite/rag_store.db"))
GGUF_DIR        = Path(os.getenv("GGUF_DIR",         "./models/gguf"))
LOG_DIR         = Path(os.getenv("LOG_DIR",          "./logs"))
OUTPUT_DIR      = Path(os.getenv("OUTPUT_DIR",       "./data/exports"))

# Ensure all directories exist at import time
for _dir in [UPLOAD_DIR, MEDIA_DIR, THUMB_DIR, AUDIO_DIR,
             TANTIVY_DIR, SQLITE_DB_PATH.parent, GGUF_DIR, LOG_DIR, OUTPUT_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)

# ── Model Names ────────────────────────────────────────────────────────────
LLM_MAIN_MODEL      = os.getenv("LLM_MAIN_MODEL",  "Llama-3.2-3B-Instruct-Q4_K_M.gguf")
LLM_DRAFT_MODEL     = os.getenv("LLM_DRAFT_MODEL", "Llama-3.2-1B-Instruct-Q4_K_M.gguf")
BGE_M3_MODEL        = "BAAI/bge-m3"
BGE_RERANKER_MODEL  = "BAAI/bge-reranker-v2-m3"
CLIP_MODEL          = "openai/clip-vit-base-patch32"
WHISPER_MODEL_SIZE  = "small"

# ── Embedding Dimensions ───────────────────────────────────────────────────
BGE_M3_DIM  = 1024   # text_chunks and audio_chunks collections
CLIP_DIM    = 512    # image_chunks collection

# ── VRAM Budget (GB) ───────────────────────────────────────────────────────
VRAM_CEILING_GB = float(os.getenv("VRAM_CEILING_GB", "3.5"))
VRAM_MODEL_SIZES = {
    "llm_3b":    2.1,
    "llm_1b":    0.7,
    "clip":      0.35,
    "bge_m3":    0.55,
    "reranker":  0.45,
    "whisper":   0.46,
    "easyocr":   0.30,
}

# ── Chunking ───────────────────────────────────────────────────────────────
CHUNK_SIZE_TOKENS   = 500
CHUNK_OVERLAP_TOKENS = 50

# ── Retrieval ──────────────────────────────────────────────────────────────
MILVUS_TOP_K        = 50
BM25_TOP_K          = 50
RRF_K               = 60
RERANKER_THRESHOLD  = 0.3
FINAL_TOP_K         = 5
CROSS_MODAL_LINK_STRENGTH_THRESHOLD = 0.6

# ── Milvus Collections ─────────────────────────────────────────────────────
MILVUS_DB_PATH              = "./data/milvus_lite.db"
COLLECTION_TEXT             = "text_chunks"
COLLECTION_IMAGE            = "image_chunks"
COLLECTION_AUDIO            = "audio_chunks"
MILVUS_NLIST_TEXT           = 128
MILVUS_NLIST_IMAGE          = 64
MILVUS_NLIST_AUDIO          = 64
MILVUS_NPROBE               = 10

# ── Audio ──────────────────────────────────────────────────────────────────
WHISPER_LANGUAGE            = "en"
AUDIO_CHUNK_MIN_SEC         = 30
AUDIO_CHUNK_MAX_SEC         = 60
AUDIO_CHUNK_OVERLAP_SEC     = 5
SILENCE_THRESHOLD_SEC       = 2.0

# ── LLM ───────────────────────────────────────────────────────────────────
LLM_N_CTX          = 4096
LLM_N_GPU_LAYERS   = -1       # offload all layers to GPU
LLM_SPECULATIVE_DRAFT_TOKENS = 5
CHAT_HISTORY_TURNS  = 5       # how many past turns to include in prompt

# ── Semantic Link Threshold ────────────────────────────────────────────────
CLIP_SEMANTIC_LINK_THRESHOLD = 0.75

# ── Export Triggers ────────────────────────────────────────────────────────
EXPORT_TRIGGER_WORDS = [
    "export", "save as", "create excel", "make word",
    "download", "generate report", "create document"
]

# ── Logging ────────────────────────────────────────────────────────────────
LOG_LEVEL = "DEBUG" if os.getenv("DEBUG", "False") == "True" else "INFO"

# ── API ────────────────────────────────────────────────────────────────────
API_HOST    = os.getenv("API_HOST",    "0.0.0.0")
API_PORT    = int(os.getenv("API_PORT", "8000"))
API_RELOAD  = os.getenv("API_RELOAD",  "true").lower() == "true"
PROJECT_NAME = "Multimodal RAG System"
VERSION      = "1.0.0"
