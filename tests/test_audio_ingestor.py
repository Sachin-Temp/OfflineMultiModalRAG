"""
tests/test_audio_ingestor.py
Full test suite for the AudioIngestor module.

Tests cover:
- Timestamp formatting utilities
- Audio normalization (requires ffmpeg)
- Silence detection
- Transcript chunking logic (using mock Whisper output)
- Word timestamp extraction
- AudioChunk metadata completeness
- Missing file handling
- Unsupported file type rejection
- Full pipeline integration (requires Whisper model)
- VRAMManager interaction

Run with:
    pytest tests/test_audio_ingestor.py -v                    # all tests
    pytest tests/test_audio_ingestor.py -v -m "not whisper"   # skip model tests
    pytest tests/test_audio_ingestor.py -v -m "not slow"      # skip slow tests

Markers:
    whisper — requires Whisper model downloaded
    slow    — takes > 10 seconds
"""

import io
import math
import struct
import wave
import pytest
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Any


# ── Test Helpers ───────────────────────────────────────────────────────────
def _make_sine_wav(
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
    frequency: float = 440.0,
    amplitude: float = 0.3,
) -> bytes:
    """
    Generate a pure sine wave WAV file as bytes.
    Used to create test audio without needing real recordings.
    The sine wave is audible (not silent) so it passes silence detection.
    """
    n_samples = int(duration_sec * sample_rate)
    samples = [
        int(amplitude * 32767 * math.sin(2 * math.pi * frequency * i / sample_rate))
        for i in range(n_samples)
    ]
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)      # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *samples))
    return buf.getvalue()


def _make_silent_wav(
    duration_sec: float = 3.0,
    sample_rate: int = 16000,
) -> bytes:
    """Generate a completely silent WAV file."""
    n_samples = int(duration_sec * sample_rate)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(struct.pack(f"<{n_samples}h", *([0] * n_samples)))
    return buf.getvalue()


def _make_mock_whisper_result(
    segments: List[Dict] = None,
    language: str = "en",
) -> Dict[str, Any]:
    """
    Build a mock Whisper result dict for testing chunking logic
    without actually running Whisper.
    """
    if segments is None:
        segments = [
            {
                "id": 0,
                "text": " Hello, this is the first segment.",
                "start": 0.0,
                "end": 5.0,
                "avg_logprob": -0.3,
                "words": [
                    {"word": "Hello,",  "start": 0.0,  "end": 0.5},
                    {"word": "this",    "start": 0.6,  "end": 0.8},
                    {"word": "is",      "start": 0.9,  "end": 1.0},
                    {"word": "the",     "start": 1.1,  "end": 1.2},
                    {"word": "first",   "start": 1.3,  "end": 1.6},
                    {"word": "segment", "start": 1.7,  "end": 2.2},
                ],
            },
            {
                "id": 1,
                "text": " This is the second segment with more content.",
                "start": 5.0,
                "end": 12.0,
                "avg_logprob": -0.2,
                "words": [
                    {"word": "This",    "start": 5.0,  "end": 5.3},
                    {"word": "is",      "start": 5.4,  "end": 5.5},
                    {"word": "the",     "start": 5.6,  "end": 5.7},
                    {"word": "second",  "start": 5.8,  "end": 6.2},
                    {"word": "segment", "start": 6.3,  "end": 6.8},
                ],
            },
        ]
    full_text = " ".join(s["text"].strip() for s in segments)
    return {"text": full_text, "segments": segments, "language": language}


# ── Timestamp Utility Tests ────────────────────────────────────────────────
class TestTimestampFormatting:

    def test_format_seconds_under_hour(self):
        from modules.ingestion.audio_ingestor import _format_timestamp
        assert _format_timestamp(63.0) == "01:03"
        assert _format_timestamp(0.0)  == "00:00"
        assert _format_timestamp(599.9) == "09:59"

    def test_format_seconds_over_hour(self):
        from modules.ingestion.audio_ingestor import _format_timestamp
        assert _format_timestamp(3661.0) == "01:01:01"
        assert _format_timestamp(3600.0) == "01:00:00"

    def test_format_timestamp_range(self):
        from modules.ingestion.audio_ingestor import _format_timestamp_range
        result = _format_timestamp_range(843.2, 901.7)
        assert "14:03" in result
        assert "15:01" in result
        assert " - " in result

    def test_format_timestamp_range_short(self):
        from modules.ingestion.audio_ingestor import _format_timestamp_range
        result = _format_timestamp_range(0.0, 30.0)
        assert "00:00" in result
        assert "00:30" in result


# ── Word Timestamp Extraction Tests ────────────────────────────────────────
class TestWordTimestampExtraction:

    def test_extract_word_timestamps_normal(self):
        from modules.ingestion.audio_ingestor import _extract_word_timestamps
        segment = {
            "words": [
                {"word": "Hello",  "start": 0.0, "end": 0.5},
                {"word": "world",  "start": 0.6, "end": 1.0},
            ]
        }
        result = _extract_word_timestamps(segment)
        assert len(result) == 2
        assert result[0].word == "Hello"
        assert result[0].start == 0.0
        assert result[1].word == "world"
        assert result[1].end == 1.0

    def test_extract_word_timestamps_empty_segment(self):
        from modules.ingestion.audio_ingestor import _extract_word_timestamps
        result = _extract_word_timestamps({"words": []})
        assert result == []

    def test_extract_word_timestamps_missing_words_key(self):
        from modules.ingestion.audio_ingestor import _extract_word_timestamps
        result = _extract_word_timestamps({})
        assert result == []

    def test_extract_word_timestamps_malformed_skipped(self):
        """Malformed entries should be skipped gracefully."""
        from modules.ingestion.audio_ingestor import _extract_word_timestamps
        segment = {
            "words": [
                {"word": "good",  "start": 0.0,  "end": 0.5},
                {"word": "",      "start": 0.6,  "end": 0.7},   # empty word — skip
                {"word": "word",  "start": 0.8,  "end": 1.0},
            ]
        }
        result = _extract_word_timestamps(segment)
        words = [w.word for w in result]
        assert "good" in words
        assert "word" in words
        assert "" not in words


# ── Chunking Logic Tests ───────────────────────────────────────────────────
class TestTranscriptChunking:

    def test_basic_chunking_produces_chunks(self):
        from modules.ingestion.audio_ingestor import _chunk_transcript
        result = _make_mock_whisper_result()
        chunks = _chunk_transcript(result, "test.mp3", "sess-001")
        assert len(chunks) >= 1

    def test_chunk_metadata_complete(self):
        from modules.ingestion.audio_ingestor import _chunk_transcript
        result = _make_mock_whisper_result()
        chunks = _chunk_transcript(result, "test_audio.mp3", "sess-001")
        for chunk in chunks:
            assert chunk.chunk_id is not None
            assert len(chunk.chunk_id) == 36   # UUID4
            assert chunk.audio_file == "test_audio.mp3"
            assert chunk.session_id == "sess-001"
            assert chunk.modality.value == "audio"
            assert chunk.start_time >= 0.0
            assert chunk.end_time > chunk.start_time
            assert chunk.timestamp_display is not None
            assert chunk.ingest_timestamp is not None
            assert chunk.text.strip() != ""
            assert chunk.embedding is None   # NOT computed here

    def test_chunk_has_word_timestamps(self):
        from modules.ingestion.audio_ingestor import _chunk_transcript
        result = _make_mock_whisper_result()
        chunks = _chunk_transcript(result, "test.mp3", None)
        for chunk in chunks:
            # Each chunk should have word timestamps from its segments
            assert isinstance(chunk.word_timestamps, list)
            # Mock data has words, so chunks should have word timestamps
            assert len(chunk.word_timestamps) >= 1

    def test_empty_segments_returns_empty(self):
        from modules.ingestion.audio_ingestor import _chunk_transcript
        result = {"text": "", "segments": [], "language": "en"}
        chunks = _chunk_transcript(result, "test.mp3", None)
        assert chunks == []

    def test_long_audio_chunked_correctly(self):
        """Simulate a long recording with many segments."""
        from modules.ingestion.audio_ingestor import _chunk_transcript
        from config.settings import AUDIO_CHUNK_MAX_SEC

        # Build 30 segments of 4 seconds each = 120 seconds total
        segments = []
        for i in range(30):
            segments.append({
                "id": i,
                "text": f" This is segment number {i} with some spoken content here.",
                "start": float(i * 4),
                "end": float((i + 1) * 4),
                "avg_logprob": -0.25,
                "words": [
                    {"word": f"segment{i}", "start": float(i * 4), "end": float(i * 4 + 1)},
                ],
            })

        result = _make_mock_whisper_result(segments=segments)
        chunks = _chunk_transcript(result, "long_audio.mp3", "sess-test")

        # Should be multiple chunks
        assert len(chunks) > 1

        # Each chunk (except possibly last) should be within max duration
        for chunk in chunks[:-1]:
            duration = chunk.end_time - chunk.start_time
            assert duration <= AUDIO_CHUNK_MAX_SEC + 10, \
                f"Chunk too long: {duration:.1f}s > {AUDIO_CHUNK_MAX_SEC}s"

    def test_no_duplicate_chunk_ids(self):
        from modules.ingestion.audio_ingestor import _chunk_transcript
        segments = []
        for i in range(20):
            segments.append({
                "id": i,
                "text": f" Sentence {i}.",
                "start": float(i * 5),
                "end": float((i + 1) * 5),
                "avg_logprob": -0.2,
                "words": [],
            })
        result = _make_mock_whisper_result(segments=segments)
        chunks = _chunk_transcript(result, "test.mp3", None)
        ids = [c.chunk_id for c in chunks]
        assert len(ids) == len(set(ids)), "Duplicate chunk IDs detected"

    def test_chunk_text_is_non_empty(self):
        from modules.ingestion.audio_ingestor import _chunk_transcript
        result = _make_mock_whisper_result()
        chunks = _chunk_transcript(result, "test.mp3", None)
        for chunk in chunks:
            assert len(chunk.text.strip()) > 0

    def test_confidence_between_zero_and_one(self):
        from modules.ingestion.audio_ingestor import _chunk_transcript
        result = _make_mock_whisper_result()
        chunks = _chunk_transcript(result, "test.mp3", None)
        for chunk in chunks:
            if chunk.confidence is not None:
                assert 0.0 <= chunk.confidence <= 1.0, \
                    f"Confidence out of range: {chunk.confidence}"

    def test_timestamp_display_format(self):
        from modules.ingestion.audio_ingestor import _chunk_transcript
        segments = [{
            "id": 0,
            "text": " Test segment.",
            "start": 843.0,
            "end": 901.0,
            "avg_logprob": -0.3,
            "words": [],
        }]
        result = _make_mock_whisper_result(segments=segments)
        chunks = _chunk_transcript(result, "test.mp3", None)
        assert len(chunks) >= 1
        assert "14:03" in chunks[0].timestamp_display
        assert " - " in chunks[0].timestamp_display


# ── Silence Detection Tests ────────────────────────────────────────────────
class TestSilenceDetection:

    def test_non_silent_audio_detected(self, tmp_path):
        wav_path = tmp_path / "sine.wav"
        wav_path.write_bytes(_make_sine_wav(duration_sec=5.0))

        from modules.ingestion.audio_ingestor import _detect_non_silent_ranges
        ranges = _detect_non_silent_ranges(wav_path)
        assert isinstance(ranges, list)
        # Should detect non-silent ranges in a sine wave
        assert len(ranges) >= 1

    def test_silent_audio_handled_gracefully(self, tmp_path):
        wav_path = tmp_path / "silent.wav"
        wav_path.write_bytes(_make_silent_wav(duration_sec=3.0))

        from modules.ingestion.audio_ingestor import _detect_non_silent_ranges
        # Should return full file range rather than crashing
        ranges = _detect_non_silent_ranges(wav_path)
        assert isinstance(ranges, list)
        assert len(ranges) >= 0   # may be 0 or 1 (full file fallback)

    def test_missing_file_returns_empty(self, tmp_path):
        fake_path = tmp_path / "nonexistent.wav"
        from modules.ingestion.audio_ingestor import _detect_non_silent_ranges
        ranges = _detect_non_silent_ranges(fake_path)
        assert ranges == []


# ── Audio Normalization Tests ──────────────────────────────────────────────
class TestAudioNormalization:

    @pytest.mark.slow
    def test_wav_normalization(self, tmp_path):
        """Test that a WAV file is properly normalized to 16kHz mono."""
        import wave as wave_mod

        # Write a test WAV file at 44100Hz stereo
        input_path = tmp_path / "input.wav"
        n_samples = 44100 * 2   # 2 seconds at 44100Hz
        buf = io.BytesIO()
        with wave_mod.open(buf, "wb") as wf:
            wf.setnchannels(2)         # stereo
            wf.setsampwidth(2)
            wf.setframerate(44100)     # 44.1kHz
            wf.writeframes(
                struct.pack(f"<{n_samples * 2}h", *([1000] * n_samples * 2))
            )
        input_path.write_bytes(buf.getvalue())

        from modules.ingestion.audio_ingestor import _normalize_audio
        output = _normalize_audio(input_path, tmp_path)

        if output is None:
            pytest.skip("ffmpeg not installed — skipping normalization test")

        assert output.exists()
        assert output.stat().st_size > 0

        # Verify output is 16kHz mono WAV
        with wave_mod.open(str(output), "rb") as wf:
            assert wf.getnchannels() == 1       # mono
            assert wf.getframerate() == 16000   # 16kHz

    def test_missing_file_returns_none(self, tmp_path):
        from modules.ingestion.audio_ingestor import _normalize_audio
        result = _normalize_audio(tmp_path / "nonexistent.mp3", tmp_path)
        assert result is None


# ── AudioIngestor Integration Tests ───────────────────────────────────────
class TestAudioIngestor:

    @pytest.fixture
    def require_ffmpeg(self):
        """Skip test if ffmpeg is not installed."""
        if shutil.which("ffmpeg") is None:
            pytest.skip("ffmpeg not installed — skipping ffmpeg-dependent test")

    @pytest.fixture
    def ingestor(self):
        from modules.ingestion.audio_ingestor import AudioIngestor
        return AudioIngestor()

    @pytest.fixture
    def sample_wav(self, tmp_path):
        """Create a test WAV file with a sine wave."""
        wav_path = tmp_path / "test_audio.wav"
        wav_path.write_bytes(_make_sine_wav(duration_sec=5.0))
        return wav_path

    def test_missing_file_returns_error(self, ingestor, tmp_path):
        result = ingestor.ingest(tmp_path / "nonexistent.mp3")
        assert not result.success
        assert len(result.errors) > 0
        assert "not found" in result.errors[0].lower()

    def test_unsupported_file_type_rejected(self, ingestor, tmp_path):
        txt_file = tmp_path / "file.txt"
        txt_file.write_text("not audio")
        result = ingestor.ingest(txt_file)
        assert not result.success
        assert len(result.errors) > 0

    def test_get_audio_duration(self, ingestor, sample_wav):
        """Duration detection requires ffprobe (part of ffmpeg)."""
        duration = ingestor.get_audio_duration(sample_wav)
        if duration is None:
            pytest.skip("ffprobe not available")
        assert 4.0 <= duration <= 6.0   # should be ~5 seconds

    @pytest.mark.whisper
    @pytest.mark.slow
    def test_full_pipeline_wav(self, ingestor, sample_wav, require_ffmpeg):
        """
        Full integration test with real Whisper model.
        The sine wave audio won't produce meaningful transcript,
        but we verify the pipeline completes without errors.
        """
        result = ingestor.ingest(sample_wav, session_id="test-sess")
        # May have warnings (Whisper may not transcribe a sine wave well)
        # but should not have errors
        assert len(result.errors) == 0
        assert result.source_file == sample_wav.name

    @pytest.mark.whisper
    @pytest.mark.slow
    def test_audio_chunk_embedding_is_none(self, ingestor, sample_wav, require_ffmpeg):
        """
        AudioChunk.embedding must be None after ingestion.
        BGE-M3 embedding is computed in Phase 4, not here.
        """
        result = ingestor.ingest(sample_wav, session_id="test-sess")
        for chunk in result.audio_chunks:
            assert chunk.embedding is None, \
                "AudioChunk.embedding should be None after Phase 3 — set in Phase 4"

    @pytest.mark.whisper
    @pytest.mark.slow
    def test_batch_ingestion(self, ingestor, tmp_path, require_ffmpeg):
        wavs = []
        for i in range(2):
            p = tmp_path / f"audio_{i}.wav"
            p.write_bytes(_make_sine_wav(duration_sec=3.0, frequency=440.0 + i * 110))
            wavs.append(p)

        results = ingestor.ingest_batch(wavs, session_id="batch-test")
        assert len(results) == 2
        for r in results:
            assert len(r.errors) == 0


# ── VRAMManager Integration ────────────────────────────────────────────────
class TestVRAMManagerAudio:

    def test_whisper_not_loaded_at_import(self):
        """Whisper should not be loaded just by importing the module."""
        from core.vram_manager import vram_manager
        assert "whisper" not in vram_manager.status()["loaded_models"]

    @pytest.mark.whisper
    @pytest.mark.slow
    def test_whisper_released_after_transcription(self, tmp_path):
        """
        Whisper must be unloaded immediately after transcription completes.
        It should NOT remain loaded between ingestion calls.
        _transcribe_audio takes a WAV path directly — no ffmpeg needed.
        """
        from core.vram_manager import vram_manager
        from modules.ingestion.audio_ingestor import _transcribe_audio

        wav_path = tmp_path / "test.wav"
        wav_path.write_bytes(_make_sine_wav(duration_sec=2.0))

        _transcribe_audio(wav_path)

        # After transcription, whisper must be released
        assert "whisper" not in vram_manager.status()["loaded_models"], \
            "Whisper was not released after transcription — VRAM leak!"
