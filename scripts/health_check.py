"""
scripts/health_check.py

Pre-demo system health checker.
Validates environment, dependencies, configuration, and connectivity
before running the Multimodal RAG system.

Usage:
    python scripts/health_check.py
"""

import sys
import os
import importlib
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


# ── Color Helpers ──────────────────────────────────────────────────────────

def green(text: str) -> str:
    return f"\033[92m{text}\033[0m"


def red(text: str) -> str:
    return f"\033[91m{text}\033[0m"


def yellow(text: str) -> str:
    return f"\033[93m{text}\033[0m"


def bold(text: str) -> str:
    return f"\033[1m{text}\033[0m"


PASS = green("✓ PASS")
FAIL = red("✗ FAIL")
WARN = yellow("⚠ WARN")


# ── Check Functions ────────────────────────────────────────────────────────

def check_python_version():
    """Check Python version ≥ 3.10."""
    v = sys.version_info
    if v.major >= 3 and v.minor >= 10:
        return True, f"Python {v.major}.{v.minor}.{v.micro}"
    return False, f"Python {v.major}.{v.minor}.{v.micro} (requires ≥ 3.10)"


def check_required_packages():
    """Check all required packages are importable."""
    packages = [
        ("fastapi", "FastAPI"),
        ("uvicorn", "Uvicorn"),
        ("loguru", "Loguru"),
        ("pydantic", "Pydantic"),
        ("dotenv", "python-dotenv"),
        ("openpyxl", "openpyxl"),
        ("docx", "python-docx"),
        ("pptx", "python-pptx"),
        ("pytest", "pytest"),
    ]

    results = []
    for import_name, display_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            results.append((True, f"{display_name} ({version})"))
        except ImportError:
            results.append((False, f"{display_name} — NOT FOUND"))

    return results


def check_optional_packages():
    """Check optional GPU/ML packages."""
    packages = [
        ("torch", "PyTorch"),
        ("sentence_transformers", "SentenceTransformers"),
        ("transformers", "Transformers"),
        ("pymilvus", "PyMilvus"),
        ("whisper", "Whisper"),
        ("easyocr", "EasyOCR"),
        ("sseclient", "SSEClient"),
        ("gradio", "Gradio"),
    ]

    results = []
    for import_name, display_name in packages:
        try:
            mod = importlib.import_module(import_name)
            version = getattr(mod, "__version__", "?")
            results.append((True, f"{display_name} ({version})"))
        except ImportError:
            results.append((None, f"{display_name} — not installed (optional)"))

    return results


def check_config_paths():
    """Check that configured directories exist."""
    try:
        from config.settings import (
            UPLOAD_DIR, OUTPUT_DIR, MEDIA_DIR, THUMB_DIR,
            AUDIO_DIR, TANTIVY_DIR, GGUF_DIR, LOG_DIR,
        )

        paths = {
            "UPLOAD_DIR": UPLOAD_DIR,
            "OUTPUT_DIR": OUTPUT_DIR,
            "MEDIA_DIR": MEDIA_DIR,
            "THUMB_DIR": THUMB_DIR,
            "AUDIO_DIR": AUDIO_DIR,
            "TANTIVY_DIR": TANTIVY_DIR,
            "GGUF_DIR": GGUF_DIR,
            "LOG_DIR": LOG_DIR,
        }

        results = []
        for name, path in paths.items():
            p = Path(path)
            if p.exists():
                results.append((True, f"{name} → {p}"))
            else:
                results.append((False, f"{name} → {p} — MISSING"))

        return results

    except Exception as e:
        return [(False, f"Config import failed: {e}")]


def check_sqlite_store():
    """Check SQLiteStore can initialize and perform basic operations."""
    try:
        from modules.indexing.sqlite_store import SQLiteStore

        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "health_check.db"
            store = SQLiteStore(db_path=db_path)
            store.initialize()

            # Write
            row_id = store.add_message("health-check", "user", "test message")
            if not row_id:
                return False, "INSERT returned no row ID"

            # Read
            history = store.get_history("health-check")
            if len(history) != 1:
                return False, f"Expected 1 message, got {len(history)}"

            # Delete
            deleted = store.delete_session_history("health-check")
            if deleted != 1:
                return False, f"Expected 1 deletion, got {deleted}"

            return True, "Read/Write/Delete OK"

    except Exception as e:
        return False, f"SQLiteStore failed: {e}"


def check_vram_config():
    """Check VRAM ceiling and model size configuration."""
    try:
        from config.settings import VRAM_CEILING_GB, VRAM_MODEL_SIZES

        if VRAM_CEILING_GB <= 0:
            return False, f"VRAM_CEILING_GB = {VRAM_CEILING_GB} (must be > 0)"

        total_model_size = sum(VRAM_MODEL_SIZES.values())
        if total_model_size > VRAM_CEILING_GB * 2:
            return None, f"Total model sizes ({total_model_size:.1f}GB) >> ceiling ({VRAM_CEILING_GB}GB)"

        models = ", ".join(f"{k}={v}GB" for k, v in VRAM_MODEL_SIZES.items())
        return True, f"Ceiling: {VRAM_CEILING_GB}GB | Models: {models}"

    except Exception as e:
        return False, f"VRAM config check failed: {e}"


def check_api_server():
    """Check if the API server is reachable (optional)."""
    try:
        import urllib.request

        from config.settings import API_HOST, API_PORT
        host = "127.0.0.1" if API_HOST == "0.0.0.0" else API_HOST
        url = f"http://{host}:{API_PORT}/health/"

        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            if resp.status == 200:
                return True, f"API server at {url} is alive"
            return False, f"API returned status {resp.status}"

    except Exception:
        return None, "API server not running (start with: python -m uvicorn api.main:app)"


# ── Main ───────────────────────────────────────────────────────────────────

def main():
    import sys
    if sys.platform == "win32":
        try:
            sys.stdout.reconfigure(encoding='utf-8')
        except AttributeError:
            pass
            
    print()
    print(bold("=" * 60))
    print(bold("  Multimodal RAG — System Health Check"))
    print(bold("=" * 60))
    print()

    total_checks = 0
    passed = 0
    failed = 0
    warnings = 0

    def report(label: str, success, detail: str):
        nonlocal total_checks, passed, failed, warnings
        total_checks += 1
        if success is True:
            print(f"  {PASS}  {label}: {detail}")
            passed += 1
        elif success is False:
            print(f"  {FAIL}  {label}: {detail}")
            failed += 1
        else:
            print(f"  {WARN}  {label}: {detail}")
            warnings += 1

    # 1. Python version
    print(bold("  ── Python Environment ──"))
    ok, detail = check_python_version()
    report("Python Version", ok, detail)
    print()

    # 2. Required packages
    print(bold("  ── Required Packages ──"))
    for ok, detail in check_required_packages():
        report("Package", ok, detail)
    print()

    # 3. Optional packages
    print(bold("  ── Optional Packages ──"))
    for ok, detail in check_optional_packages():
        report("Package", ok, detail)
    print()

    # 4. Config paths
    print(bold("  ── Configuration Paths ──"))
    for ok, detail in check_config_paths():
        report("Path", ok, detail)
    print()

    # 5. SQLite
    print(bold("  ── SQLite Store ──"))
    ok, detail = check_sqlite_store()
    report("SQLiteStore", ok, detail)
    print()

    # 6. VRAM
    print(bold("  ── VRAM Configuration ──"))
    ok, detail = check_vram_config()
    report("VRAM Config", ok, detail)
    print()

    # 7. API Server
    print(bold("  ── API Server ──"))
    ok, detail = check_api_server()
    report("API Server", ok, detail)
    print()

    # Summary
    print(bold("  ── Summary ──"))
    print(f"  Total: {total_checks} | {green(f'Passed: {passed}')} | {red(f'Failed: {failed}')} | {yellow(f'Warnings: {warnings}')}")
    print()

    if failed > 0:
        print(red("  ✗ System is NOT ready for demo."))
        print(f"  Fix the {failed} failing check(s) above before proceeding.")
        sys.exit(1)
    elif warnings > 0:
        print(yellow("  ⚠ System has warnings but may be functional."))
        print("  Review the warnings above.")
        sys.exit(0)
    else:
        print(green("  ✓ System is ready for demo!"))
        sys.exit(0)


if __name__ == "__main__":
    main()
