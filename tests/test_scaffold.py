"""
test_scaffold.py — Smoke tests for Phase 0 scaffold.
Run with: pytest tests/test_scaffold.py -v
"""

import pytest
from pathlib import Path


def test_settings_import():
    from config.settings import (
        UPLOAD_DIR, MEDIA_DIR, THUMB_DIR, AUDIO_DIR,
        TANTIVY_DIR, SQLITE_DB_PATH, GGUF_DIR, LOG_DIR,
        BGE_M3_DIM, CLIP_DIM, VRAM_CEILING_GB
    )
    assert BGE_M3_DIM == 1024
    assert CLIP_DIM == 512
    assert VRAM_CEILING_GB == 3.5


def test_directories_exist():
    from config.settings import (
        UPLOAD_DIR, MEDIA_DIR, THUMB_DIR, AUDIO_DIR,
        TANTIVY_DIR, LOG_DIR
    )
    for d in [UPLOAD_DIR, MEDIA_DIR, THUMB_DIR, AUDIO_DIR, TANTIVY_DIR, LOG_DIR]:
        assert Path(d).exists(), f"Directory missing: {d}"


def test_vram_manager_singleton():
    from core.vram_manager import VRAMManager
    a = VRAMManager()
    b = VRAMManager()
    assert a is b, "VRAMManager must be a singleton"


def test_vram_manager_acquire_release():
    from core.vram_manager import VRAMManager
    mgr = VRAMManager()
    initial = mgr.used_gb
    mgr.acquire("clip")
    assert mgr.used_gb > initial
    mgr.release("clip")
    assert abs(mgr.used_gb - initial) < 0.01


def test_vram_manager_status():
    from core.vram_manager import VRAMManager
    mgr = VRAMManager()
    status = mgr.status()
    assert "used_gb" in status
    assert "ceiling_gb" in status
    assert "loaded_models" in status


def test_placeholder_modules_importable():
    # All placeholder modules must exist and be importable
    import modules.ingestion.text_ingestor
    import modules.ingestion.image_ingestor
    import modules.ingestion.audio_ingestor
    import modules.indexing.milvus_store
    import modules.indexing.tantivy_index
    import modules.indexing.sqlite_store
    import modules.retrieval.retrieval_engine
    import modules.llm.llm_engine
    import modules.citation.citation_engine
    import modules.export.file_exporter
    import api.endpoints
    import frontend.gradio_app
