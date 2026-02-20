"""
verify_setup.py
Run after download_models.py to confirm everything is ready.
Checks: imports, directory structure, GGUF files, CUDA availability.

Usage:
    python scripts/verify_setup.py
"""

import sys
import os
from pathlib import Path

# Fix Windows console encoding for Unicode characters
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config.settings import GGUF_DIR, LLM_MAIN_MODEL, LLM_DRAFT_MODEL

CHECKS = []

def check(name, fn):
    try:
        fn()
        print(f"  [PASS] {name}")
        CHECKS.append((name, True))
    except Exception as e:
        print(f"  [FAIL] {name} -- {e}")
        CHECKS.append((name, False))

print("\n=== Multimodal RAG -- Environment Verification ===\n")

print("[1] Python Imports")
check("torch",                   lambda: __import__("torch"))
check("transformers",            lambda: __import__("transformers"))
check("sentence_transformers",   lambda: __import__("sentence_transformers"))
check("clip",                    lambda: __import__("clip"))
check("whisper",                 lambda: __import__("whisper"))
check("easyocr",                 lambda: __import__("easyocr"))
check("pymilvus",                lambda: __import__("pymilvus"))
check("tantivy",                 lambda: __import__("tantivy"))
check("fitz (PyMuPDF)",          lambda: __import__("fitz"))
check("docx",                    lambda: __import__("docx"))
check("pdfplumber",              lambda: __import__("pdfplumber"))
check("fastapi",                 lambda: __import__("fastapi"))
check("gradio",                  lambda: __import__("gradio"))
check("llama_cpp",               lambda: __import__("llama_cpp"))
check("pandas",                  lambda: __import__("pandas"))
check("openpyxl",                lambda: __import__("openpyxl"))
check("pptx",                    lambda: __import__("pptx"))
check("loguru",                  lambda: __import__("loguru"))
check("dotenv",                  lambda: __import__("dotenv"))
check("magic",                   lambda: __import__("magic"))
check("tiktoken",                lambda: __import__("tiktoken"))

print("\n[2] CUDA")
import torch
check("CUDA available", lambda: (_ for _ in ()).throw(RuntimeError("No CUDA")) if not torch.cuda.is_available() else None)
if torch.cuda.is_available():
    print(f"       GPU: {torch.cuda.get_device_name(0)}")
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"       VRAM: {total_vram:.1f} GB")

def check_gguf(path):
    if not path.exists():
        print(f"      Checked: {path.resolve()}")
        raise FileNotFoundError(f"File not found: {path}")

print("\n[3] GGUF Model Files")
check(LLM_MAIN_MODEL,  lambda: check_gguf(GGUF_DIR / LLM_MAIN_MODEL))
check(LLM_DRAFT_MODEL, lambda: check_gguf(GGUF_DIR / LLM_DRAFT_MODEL))

print("\n[4] Directory Structure")
from config.settings import UPLOAD_DIR, MEDIA_DIR, THUMB_DIR, AUDIO_DIR, TANTIVY_DIR, LOG_DIR
for d in [UPLOAD_DIR, MEDIA_DIR, THUMB_DIR, AUDIO_DIR, TANTIVY_DIR, LOG_DIR]:
    check(str(d), lambda d=d: (_ for _ in ()).throw(NotADirectoryError()) if not d.exists() else None)

print("\n[5] VRAMManager")
check("VRAMManager singleton", lambda: __import__("core.vram_manager", fromlist=["vram_manager"]))

passed = sum(1 for _, ok in CHECKS if ok)
total  = len(CHECKS)
print(f"\n=== Result: {passed}/{total} checks passed ===")
if passed == total:
    print("[PASS] Environment is fully ready. Proceed to Phase 1.\n")
else:
    print("[FAIL] Fix the failing checks above before proceeding.\n")
