"""
download_models.py
Run this ONCE before first use to pre-download all required models.
After running, the system works fully offline.

Usage:
    python scripts/download_models.py
"""

import sys
from pathlib import Path
from loguru import logger

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from config.settings import GGUF_DIR, BGE_M3_MODEL, BGE_RERANKER_MODEL, CLIP_MODEL, WHISPER_MODEL_SIZE


def download_sentence_transformers():
    logger.info("Downloading BGE-M3 (text embeddings + audio transcript embeddings)...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(BGE_M3_MODEL)
    logger.success(f"BGE-M3 downloaded and cached.")

    logger.info("Downloading BGE-Reranker-v2-m3 (cross-encoder reranker)...")
    from sentence_transformers import CrossEncoder
    reranker = CrossEncoder(BGE_RERANKER_MODEL)
    logger.success("BGE-Reranker downloaded and cached.")


def download_clip():
    logger.info("Downloading CLIP ViT-B/32 (visual embeddings)...")
    import clip
    model, preprocess = clip.load("ViT-B/32", device="cpu")
    logger.success("CLIP ViT-B/32 downloaded and cached.")


def download_whisper():
    logger.info(f"Downloading Whisper-{WHISPER_MODEL_SIZE} (audio transcription)...")
    import whisper
    model = whisper.load_model(WHISPER_MODEL_SIZE, device="cpu")
    logger.success(f"Whisper-{WHISPER_MODEL_SIZE} downloaded and cached.")


def download_easyocr():
    logger.info("Pre-downloading EasyOCR English model...")
    import easyocr
    reader = easyocr.Reader(["en"], gpu=False)
    logger.success("EasyOCR English model downloaded and cached.")


def print_gguf_instructions():
    print("\n" + "="*60)
    print("MANUAL DOWNLOAD REQUIRED — GGUF LLM Models")
    print("="*60)
    print(f"Download these two files and place them in: {GGUF_DIR.resolve()}\n")
    print("1. Llama-3.2-3B-Instruct-Q4_K_M.gguf (~2.0GB)")
    print("   URL: https://huggingface.co/bartowski/Llama-3.2-3B-Instruct-GGUF")
    print()
    print("2. Llama-3.2-1B-Instruct-Q4_K_M.gguf (~0.7GB)")
    print("   URL: https://huggingface.co/bartowski/Llama-3.2-1B-Instruct-GGUF")
    print("="*60 + "\n")


if __name__ == "__main__":
    logger.add("logs/download_models.log", rotation="10MB")
    logger.info("Starting model downloads for Multimodal RAG System...")

    try:
        download_sentence_transformers()
    except Exception as e:
        logger.error(f"BGE models failed: {e}")

    try:
        download_clip()
    except Exception as e:
        logger.error(f"CLIP failed: {e}")

    try:
        download_whisper()
    except Exception as e:
        logger.error(f"Whisper failed: {e}")

    try:
        download_easyocr()
    except Exception as e:
        logger.error(f"EasyOCR failed: {e}")

    print_gguf_instructions()
    logger.success("All auto-downloadable models complete. See instructions above for GGUF files.")
