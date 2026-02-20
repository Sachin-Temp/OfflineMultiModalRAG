"""
tests/test_llm_engine.py

Tests for the LLM Intelligence Engine (Phase 7).
Mocks external dependencies (llama_cpp, vram_manager, sqlite_store).
"""

import json
import pytest
from unittest.mock import MagicMock, patch, ANY

from modules.llm.llm_engine import (
    LLMEngine,
    _format_text_context,
    _format_image_context,
    _format_audio_context,
    build_rag_prompt,
    is_export_request,
    _detect_export_type,
    build_export_prompt,
    _load_llm_models,
    _unload_llm_models,
)
from modules.retrieval.retrieval_engine import RetrievalResult, GoldChunk


# ── Fixtures ───────────────────────────────────────────────────────────────

@pytest.fixture
def mock_chunk_text():
    return GoldChunk(
        chunk_id="text_1",
        modality="text",
        text="This is a test document.",
        source_file="test.pdf",
        page_number=1,
        reranker_score=0.9,
        rrf_score=0.8,
    )

@pytest.fixture
def mock_chunk_image():
    return GoldChunk(
        chunk_id="image_1",
        modality="image",
        text="OCR content",
        source_file="test.pdf",
        page_number=2,
        reranker_score=0.85,
        rrf_score=0.75,
        image_path="section1_figure1.jpg",
        metadata_json=json.dumps({"llm_description": "A chart showing growth"})
    )

@pytest.fixture
def mock_chunk_audio():
    return GoldChunk(
        chunk_id="audio_1",
        modality="audio",
        text="Speaker: Hello world.",
        source_file="meeting.mp3",
        page_number=0,
        reranker_score=0.88,
        rrf_score=0.78,
        start_time=10.0,
        end_time=20.0,
        timestamp_display="00:10-00:20"
    )

@pytest.fixture
def mock_retrieval_result(mock_chunk_text, mock_chunk_image):
    return RetrievalResult(
        query="test query",
        gold_chunks=[mock_chunk_text, mock_chunk_image],
        linked_chunks=[],
        total_candidates=10
    )


# ── Context Formatting Tests ───────────────────────────────────────────────

def test_format_text_context(mock_chunk_text):
    output = _format_text_context(mock_chunk_text, 1)
    assert "[1] SOURCE: test.pdf | PAGE: 1 | TYPE: text" in output
    assert "TEXT: This is a test document." in output

def test_format_image_context(mock_chunk_image):
    output = _format_image_context(mock_chunk_image, 2)
    assert "[2] SOURCE: test.pdf | PAGE: 2 | TYPE: image" in output
    assert "OCR: OCR content" in output
    assert "DESCRIPTION: A chart showing growth" in output
    assert "FILE: section1_figure1.jpg" in output

def test_format_audio_context(mock_chunk_audio):
    output = _format_audio_context(mock_chunk_audio, 3)
    assert "[3] SOURCE: meeting.mp3 | TIMESTAMP: 00:10-00:20 | TYPE: audio" in output
    assert "TRANSCRIPT: Speaker: Hello world." in output

def test_build_rag_prompt(mock_retrieval_result):
    history = [{"role": "user", "message": "Previous question"}, {"role": "assistant", "message": "Previous answer"}]
    prompt, index_map = build_rag_prompt("Current question", mock_retrieval_result, history)
    
    assert "<|system|>" in prompt
    assert "<|context|>" in prompt
    assert "<|history|>" in prompt
    assert "User: Previous question" in prompt
    assert "<|user|>\nCurrent question" in prompt
    # Verify index mapping
    assert 1 in index_map and index_map[1].chunk_id == "text_1"
    assert 2 in index_map and index_map[2].chunk_id == "image_1"


# ── Export Logic Tests ─────────────────────────────────────────────────────

def test_is_export_request():
    assert is_export_request("Export this to excel")
    assert is_export_request("generate report about sales")
    assert is_export_request("create document")
    assert not is_export_request("What is the revenue?")

def test_detect_export_type():
    assert _detect_export_type("export to excel") == "xlsx"
    assert _detect_export_type("make a word doc") == "docx"
    assert _detect_export_type("create a presentation") == "pptx"
    assert _detect_export_type("give me a csv") == "csv"
    assert _detect_export_type("export data") == "xlsx"  # default

def test_build_export_prompt(mock_retrieval_result):
    prompt = build_export_prompt("export to excel", mock_retrieval_result)
    assert '"export_type": "xlsx"' in prompt
    assert '"headers": [str]' in prompt
    assert "Extract all relevant structured data" in prompt


# ── Model Loading & Verification Tests ─────────────────────────────────────

@patch("modules.llm.llm_engine.vram_manager")
def test_load_models_vram_calls(mock_vram):
    # Mocking existence of model files
    with patch("pathlib.Path.exists", return_value=True):
        # Mocking the local import of llama_cpp by injecting into sys.modules
        mock_llama_cpp = MagicMock()
        with patch.dict("sys.modules", {"llama_cpp": mock_llama_cpp}):
            mock_vram.acquire.return_value = True
            
            # Reset global variables
            with patch("modules.llm.llm_engine._llm_main", None), \
                 patch("modules.llm.llm_engine._llm_draft", None):
                
                main, draft = _load_llm_models()
                
                # Check 2 acquisitions
                assert mock_vram.acquire.call_count == 2
                mock_vram.acquire.assert_any_call("llm_3b")
                mock_vram.acquire.assert_any_call("llm_1b")
                
                # Check register calls
                assert mock_vram.register_evict_callback.call_count == 2
                
                # Check Llama instantiation
                assert mock_llama_cpp.Llama.call_count == 2


# ── End-to-End Engine Tests ────────────────────────────────────────────────

@patch("modules.llm.llm_engine.LLMEngine._inject_citations", side_effect=lambda x, y: x)  # pass through
@patch("modules.llm.llm_engine._load_llm_models")
def test_llm_engine_generate_stream(mock_load, mock_inject, mock_retrieval_result):
    # Mock models
    mock_main = MagicMock()
    mock_draft = MagicMock()
    mock_load.return_value = (mock_main, mock_draft)
    
    # Mock generation output
    mock_main.create_completion.return_value = [{"choices": [{"text": "Hello"}]}, {"choices": [{"text": " World"}]}]
    
    engine = LLMEngine()
    # Mock sqlite
    engine.sqlite = MagicMock()
    
    # Run generation
    generator = engine.generate_stream("Hello", mock_retrieval_result, [])
    tokens = list(generator)
    
    assert tokens == ["Hello", " World"]
    # Check that create_completion was called with draft model
    mock_main.create_completion.assert_called_once()
    _, kwargs = mock_main.create_completion.call_args
    assert kwargs["draft_model"] == mock_draft
    
    # Check creation of history
    assert engine.sqlite.add_message.call_count == 2  # user + assistant

@patch("modules.llm.llm_engine.is_export_request", return_value=True)
@patch("modules.llm.llm_engine._load_llm_models")
def test_llm_engine_export_mode(mock_load, mock_is_export, mock_retrieval_result):
    mock_main = MagicMock()
    # Mock return for export: a JSON string
    json_resp = '{"export_type": "xlsx", "data": []}'
    mock_main.return_value = {"choices": [{"text": json_resp}]}
    mock_load.return_value = (mock_main, None)
    
    engine = LLMEngine()
    engine.sqlite = MagicMock()
    
    generator = engine.generate_stream("export stuff", mock_retrieval_result, [])
    tokens = list(generator)
    
    assert len(tokens) == 1
    assert tokens[0] == json_resp
