"""
modules/ingestion/audio_ingestor.py

Handles ingestion of audio files into AudioChunk objects.

Supports: MP3, M4A, WAV, OGG, FLAC (any format ffmpeg can decode)

Pipeline:
  1. Format normalization: ffmpeg -> 16kHz mono WAV
  2. Silence detection: pydub -> skip silent segments > 2s
  3. Transcription: Whisper-small with word_timestamps=True
  4. Chunking: on Whisper segment boundaries, 30-60s per chunk, 5s overlap
  5. Diarization (optional): pyannote.audio CPU mode -> speaker_id per chunk
  6. Return AudioChunk objects (embedding=None, filled in Phase 4)

CRITICAL: word_timestamps=True is NON-NEGOTIABLE.
Without word-level timestamps, citation linking to exact audio moments
is impossible. Every AudioChunk must carry start_time and end_time
in seconds from the audio start, plus a human-readable timestamp_display.

No Milvus/Tantivy/SQLite writes happen here. No BGE-M3 embedding here.
"""

import math
import os
import re
import uuid
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import magic
from loguru import logger

from config.settings import (
    AUDIO_DIR,
    WHISPER_MODEL_SIZE,
    WHISPER_LANGUAGE,
    AUDIO_CHUNK_MIN_SEC,
    AUDIO_CHUNK_MAX_SEC,
    AUDIO_CHUNK_OVERLAP_SEC,
    SILENCE_THRESHOLD_SEC,
)
from core.vram_manager import vram_manager
from models.schemas import AudioChunk, WordTimestamp, Modality, IngestionResult


# ── Constants ──────────────────────────────────────────────────────────────
SUPPORTED_AUDIO_MIMES = {
    "audio/mpeg":       ".mp3",
    "audio/mp4":        ".m4a",
    "audio/x-m4a":      ".m4a",
    "audio/wav":        ".wav",
    "audio/x-wav":      ".wav",
    "audio/ogg":        ".ogg",
    "audio/flac":       ".flac",
    "audio/x-flac":     ".flac",
    "video/mp4":        ".mp4",    # some recorders save as mp4
}

# pydub silence detection parameters
SILENCE_MIN_SILENCE_LEN_MS = int(SILENCE_THRESHOLD_SEC * 1000)  # ms
SILENCE_THRESH_DBfs        = -40    # dBFS below which audio is considered silent

# Global model references
_whisper_model = None


# ── Utility: Seconds -> HH:MM:SS ──────────────────────────────────────────
def _format_timestamp(seconds: float) -> str:
    """
    Convert float seconds to human-readable HH:MM:SS format.
    Used for AudioChunk.timestamp_display field.

    Example: 843.2 -> "14:03"
             3723.5 -> "01:02:03"
    """
    seconds = int(seconds)
    hours   = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs    = seconds % 60

    if hours > 0:
        return f"{hours:02d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_timestamp_range(start: float, end: float) -> str:
    """Return 'HH:MM - HH:MM' display string for a chunk's time range."""
    return f"{_format_timestamp(start)} - {_format_timestamp(end)}"


# ── Step 1: Format Normalization ───────────────────────────────────────────
def _normalize_audio(input_path: Path, output_dir: Path) -> Optional[Path]:
    """
    Convert any audio file to 16kHz mono WAV using ffmpeg.
    This is Whisper's required input format.

    Args:
        input_path: Path to input audio file (any format)
        output_dir: Directory to write the normalized WAV file

    Returns:
        Path to normalized WAV file, or None if conversion failed.

    Why 16kHz mono WAV:
        - Whisper was trained on 16kHz audio
        - Mono reduces file size without quality loss for speech
        - WAV avoids any codec decoding overhead during transcription
    """
    safe_stem = re.sub(r"[^\w\-]", "_", input_path.stem)
    output_filename = f"{safe_stem}_normalized.wav"
    output_path = output_dir / output_filename

    if output_path.exists():
        logger.debug(f"Normalized audio already exists: {output_path.name}")
        return output_path

    cmd = [
        "ffmpeg",
        "-i", str(input_path),      # input file
        "-ar", "16000",             # sample rate: 16kHz
        "-ac", "1",                 # channels: mono
        "-acodec", "pcm_s16le",     # codec: 16-bit PCM WAV
        "-y",                       # overwrite output if exists
        "-loglevel", "error",       # suppress ffmpeg output (errors only)
        str(output_path),
    ]

    logger.info(f"Normalizing audio: {input_path.name} -> 16kHz mono WAV")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=300,    # 5 minute timeout for very large files
        )
        if result.returncode != 0:
            logger.error(f"ffmpeg failed for {input_path.name}: {result.stderr}")
            return None

        if not output_path.exists() or output_path.stat().st_size == 0:
            logger.error(f"ffmpeg produced empty output for {input_path.name}")
            return None

        logger.info(f"Audio normalized: {output_path.name} "
                    f"({output_path.stat().st_size / 1024:.1f} KB)")
        return output_path

    except subprocess.TimeoutExpired:
        logger.error(f"ffmpeg timeout for {input_path.name}")
        return None
    except FileNotFoundError:
        logger.error(
            "ffmpeg not found. Install ffmpeg and add to PATH. "
            "Windows: choco install ffmpeg  |  Linux: sudo apt install ffmpeg"
        )
        return None
    except Exception as e:
        logger.error(f"Audio normalization failed: {e}")
        return None


# ── Step 2: Silence Detection ──────────────────────────────────────────────
def _detect_non_silent_ranges(wav_path: Path) -> List[Tuple[int, int]]:
    """
    Detect non-silent time ranges in a WAV file using pydub.
    Returns list of (start_ms, end_ms) tuples for non-silent segments.

    Silent segments longer than SILENCE_THRESHOLD_SEC are excluded.
    This can reduce transcription time by up to 40% for meeting recordings
    with long pauses.

    Returns empty list on error (caller should treat full file as non-silent).
    """
    try:
        from pydub import AudioSegment
        from pydub.silence import detect_nonsilent

        audio = AudioSegment.from_wav(str(wav_path))
        total_duration_ms = len(audio)

        nonsilent_ranges = detect_nonsilent(
            audio,
            min_silence_len=SILENCE_MIN_SILENCE_LEN_MS,
            silence_thresh=SILENCE_THRESH_DBfs,
            seek_step=100,    # check every 100ms (balance speed vs accuracy)
        )

        if not nonsilent_ranges:
            logger.warning(f"No non-silent segments found in {wav_path.name}. "
                           "File may be silent or silence threshold too aggressive.")
            # Return full file as non-silent to avoid losing content
            return [(0, total_duration_ms)]

        total_non_silent_ms = sum(end - start for start, end in nonsilent_ranges)
        skipped_pct = 100 * (1 - total_non_silent_ms / total_duration_ms)

        logger.info(
            f"Silence detection: {len(nonsilent_ranges)} non-silent ranges, "
            f"{skipped_pct:.1f}% silence skipped"
        )
        return nonsilent_ranges

    except ImportError:
        logger.warning("pydub not available — skipping silence detection")
        return []
    except Exception as e:
        logger.warning(f"Silence detection failed: {e} — processing full file")
        return []


# ── Step 3: Whisper Transcription ──────────────────────────────────────────
def _load_whisper():
    """
    Load Whisper model with VRAM management.
    Acquires VRAM slot before loading.
    Returns loaded whisper model.
    """
    global _whisper_model

    if _whisper_model is not None:
        logger.debug("Whisper already loaded — reusing cached instance")
        return _whisper_model

    def _evict_whisper():
        global _whisper_model
        import torch
        if _whisper_model is not None:
            del _whisper_model
            _whisper_model = None
            torch.cuda.empty_cache()
            logger.info("Whisper evicted from VRAM by VRAMManager")

    vram_manager.register_evict_callback("whisper", _evict_whisper)

    acquired = vram_manager.acquire("whisper")
    if not acquired:
        raise RuntimeError(
            "VRAMManager could not allocate VRAM for Whisper. "
            "Ensure CLIP and LLM are not both loaded simultaneously."
        )

    try:
        import whisper
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Loading Whisper-{WHISPER_MODEL_SIZE} on {device}...")
        model = whisper.load_model(WHISPER_MODEL_SIZE, device=device)
        _whisper_model = model
        logger.info(f"Whisper-{WHISPER_MODEL_SIZE} loaded on {device}")
        return model
    except Exception as e:
        vram_manager.release("whisper")
        raise RuntimeError(f"Failed to load Whisper: {e}") from e


def _unload_whisper():
    """
    Explicitly unload Whisper and release VRAM.
    Call immediately after transcription is complete.
    Whisper should never stay loaded while other models are needed.
    """
    global _whisper_model
    import torch
    if _whisper_model is not None:
        del _whisper_model
        _whisper_model = None
        torch.cuda.empty_cache()
        vram_manager.release("whisper")
        logger.info("Whisper explicitly unloaded and VRAM released")


def _transcribe_audio(wav_path: Path) -> Optional[Dict[str, Any]]:
    """
    Transcribe a WAV file using Whisper with word-level timestamps.

    Returns Whisper's full result dict containing:
        - result["text"]: full transcript string
        - result["segments"]: list of segment dicts, each with:
            - segment["text"]: text of this segment
            - segment["start"]: start time in seconds
            - segment["end"]: end time in seconds
            - segment["words"]: list of word dicts (if word_timestamps=True):
                - word["word"]: the word string
                - word["start"]: word start time in seconds
                - word["end"]: word end time in seconds

    Returns None if transcription fails.

    CRITICAL: word_timestamps=True is NON-NEGOTIABLE.
    Without it, AudioChunk.word_timestamps will be empty and
    citation linking to exact moments in audio will be impossible.
    """
    try:
        model = _load_whisper()
        logger.info(f"Transcribing: {wav_path.name}")

        result = model.transcribe(
            str(wav_path),
            language=WHISPER_LANGUAGE,
            word_timestamps=True,       # NON-NEGOTIABLE
            verbose=False,
            # fp16=False forces CPU precision — avoids NaN issues on some GPUs
            fp16=False,
            # temperature=0 for deterministic output
            temperature=0,
            # condition_on_previous_text helps with long recordings
            condition_on_previous_text=True,
            # Compression ratio filter — removes hallucinated repetitions
            compression_ratio_threshold=2.4,
            # Log probability threshold — removes low-confidence outputs
            logprob_threshold=-1.0,
            # No speech probability threshold
            no_speech_threshold=0.6,
        )

        n_segments = len(result.get("segments", []))
        total_words = sum(
            len(seg.get("words", []))
            for seg in result.get("segments", [])
        )
        logger.info(
            f"Transcription complete: {n_segments} segments, "
            f"{total_words} words with timestamps"
        )
        return result

    except Exception as e:
        logger.error(f"Whisper transcription failed for {wav_path.name}: {e}")
        return None
    finally:
        # Always unload Whisper immediately after transcription
        _unload_whisper()


# ── Step 4: Transcript Chunking ────────────────────────────────────────────
def _extract_word_timestamps(segment: Dict) -> List[WordTimestamp]:
    """
    Extract WordTimestamp objects from a Whisper segment dict.
    Handles cases where word timestamps may be missing or malformed.
    """
    words = []
    for w in segment.get("words", []):
        try:
            word_text  = str(w.get("word", "")).strip()
            word_start = float(w.get("start", 0.0))
            word_end   = float(w.get("end", 0.0))
            if word_text:
                words.append(WordTimestamp(
                    word=word_text,
                    start=word_start,
                    end=word_end,
                ))
        except (TypeError, ValueError) as e:
            logger.debug(f"Skipping malformed word timestamp: {w} — {e}")
            continue
    return words


def _chunk_transcript(
    whisper_result: Dict[str, Any],
    audio_file: str,
    session_id: Optional[str],
) -> List[AudioChunk]:
    """
    Convert Whisper's segment output into AudioChunk objects.

    Chunking strategy:
    - Use Whisper's natural segment boundaries (pauses, sentence ends)
    - Greedily merge segments until AUDIO_CHUNK_MAX_SEC is reached
    - If merged chunk exceeds max, save it and start a new one
    - Add AUDIO_CHUNK_OVERLAP_SEC overlap: the last segment(s) of a chunk
      are repeated at the start of the next chunk
    - Target chunk duration: AUDIO_CHUNK_MIN_SEC to AUDIO_CHUNK_MAX_SEC

    Why segment boundaries (not word boundaries):
    - Whisper's segments respect natural pauses and sentence endings
    - Cutting mid-segment would split sentences unnaturally
    - Segment boundaries are the best proxy for semantic boundaries in audio

    Args:
        whisper_result: Full Whisper transcription result dict
        audio_file:     Source audio filename (for metadata)
        session_id:     Session identifier

    Returns:
        List of AudioChunk objects with all metadata populated.
        embedding field is None — filled in Phase 4.
    """
    segments = whisper_result.get("segments", [])
    if not segments:
        logger.warning("Whisper returned no segments — empty transcript")
        return []

    chunks: List[AudioChunk] = []
    now = datetime.now(timezone.utc).isoformat()

    # We'll accumulate segments into a chunk
    current_segments: List[Dict] = []
    current_start: float = 0.0
    current_end: float   = 0.0
    current_duration: float = 0.0

    def _flush_chunk(segs: List[Dict]) -> Optional[AudioChunk]:
        """Build an AudioChunk from a list of Whisper segments."""
        if not segs:
            return None

        texts  = [s.get("text", "").strip() for s in segs]
        text   = " ".join(t for t in texts if t)

        if not text.strip():
            return None

        start  = float(segs[0].get("start", 0.0))
        end    = float(segs[-1].get("end", start))

        # Collect all word timestamps across all segments in this chunk
        all_words: List[WordTimestamp] = []
        for seg in segs:
            all_words.extend(_extract_word_timestamps(seg))

        # Compute average confidence across segments
        confidences = [
            float(seg.get("avg_logprob", -1.0))
            for seg in segs
            if "avg_logprob" in seg
        ]
        # Convert log-prob to approximate confidence (0-1)
        avg_confidence = None
        if confidences:
            avg_logprob = sum(confidences) / len(confidences)
            # Clamp to reasonable range: logprob of 0 = confidence 1.0
            avg_confidence = round(min(1.0, math.exp(avg_logprob)), 4)

        return AudioChunk(
            chunk_id=str(uuid.uuid4()),
            modality=Modality.AUDIO,
            text=text,
            audio_file=audio_file,
            start_time=start,
            end_time=end,
            timestamp_display=_format_timestamp_range(start, end),
            speaker_id=None,            # filled by diarization step if enabled
            word_timestamps=all_words,
            confidence=avg_confidence,
            ingest_timestamp=now,
            session_id=session_id,
            embedding=None,             # filled in Phase 4 by BGE-M3
        )

    for i, segment in enumerate(segments):
        seg_start    = float(segment.get("start", 0.0))
        seg_end      = float(segment.get("end", seg_start))
        seg_duration = seg_end - seg_start

        if not current_segments:
            # Starting a new chunk
            current_segments = [segment]
            current_start    = seg_start
            current_end      = seg_end
            current_duration = seg_duration
            continue

        potential_duration = seg_end - current_start

        if potential_duration <= AUDIO_CHUNK_MAX_SEC:
            # Still within max chunk duration — keep merging
            current_segments.append(segment)
            current_end      = seg_end
            current_duration = potential_duration
        else:
            # Exceeds max — flush current chunk
            # Check if we meet minimum duration
            if current_duration >= AUDIO_CHUNK_MIN_SEC or len(chunks) == 0:
                chunk = _flush_chunk(current_segments)
                if chunk:
                    chunks.append(chunk)

            # Build overlap: take last N seconds worth of segments
            overlap_segments: List[Dict] = []
            overlap_duration: float = 0.0
            for seg in reversed(current_segments):
                seg_dur = float(seg.get("end", 0)) - float(seg.get("start", 0))
                if overlap_duration + seg_dur > AUDIO_CHUNK_OVERLAP_SEC:
                    break
                overlap_segments.insert(0, seg)
                overlap_duration += seg_dur

            # Start new chunk with overlap + current segment
            current_segments = overlap_segments + [segment]
            current_start    = float(current_segments[0].get("start", 0.0))
            current_end      = seg_end
            current_duration = current_end - current_start

    # Flush final chunk
    if current_segments:
        chunk = _flush_chunk(current_segments)
        if chunk:
            chunks.append(chunk)

    logger.info(
        f"Transcript chunked: {len(segments)} Whisper segments -> "
        f"{len(chunks)} AudioChunks"
    )
    return chunks


# ── Step 5: Speaker Diarization (Optional) ────────────────────────────────
def _apply_diarization(
    chunks: List[AudioChunk],
    wav_path: Path,
) -> List[AudioChunk]:
    """
    Apply speaker diarization to AudioChunks using pyannote.audio.
    Runs entirely on CPU — never acquires VRAM.

    For each AudioChunk, determines which speaker is dominant in that
    time range and sets chunk.speaker_id accordingly.

    This is OPTIONAL — if pyannote is not installed or fails, the chunks
    are returned unchanged with speaker_id=None.

    Args:
        chunks:   List of AudioChunk objects to annotate
        wav_path: Path to the normalized WAV file

    Returns:
        The same list of AudioChunks with speaker_id fields populated
        (or unchanged if diarization fails/unavailable).
    """
    try:
        from pyannote.audio import Pipeline
        import torch
    except ImportError:
        logger.info("pyannote.audio not available — skipping speaker diarization")
        return chunks

    try:
        logger.info("Running speaker diarization (CPU mode)...")

        # Load pipeline on CPU
        # Note: pyannote may require HuggingFace auth token for some models
        # We use a basic segmentation approach that works without auth
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=False,
        )
        pipeline = pipeline.to(torch.device("cpu"))

        diarization = pipeline(str(wav_path))

        # Build a timeline of (start, end, speaker) from diarization output
        speaker_timeline: List[Tuple[float, float, str]] = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_timeline.append((turn.start, turn.end, speaker))

        # For each chunk, find the dominant speaker in its time range
        for chunk in chunks:
            speaker_durations: Dict[str, float] = {}
            for seg_start, seg_end, speaker in speaker_timeline:
                # Compute overlap of diarization segment with chunk
                overlap_start = max(chunk.start_time, seg_start)
                overlap_end   = min(chunk.end_time, seg_end)
                overlap = overlap_end - overlap_start
                if overlap > 0:
                    speaker_durations[speaker] = (
                        speaker_durations.get(speaker, 0.0) + overlap
                    )

            if speaker_durations:
                dominant_speaker = max(speaker_durations, key=speaker_durations.get)
                chunk.speaker_id = dominant_speaker

        speakers_found = set(c.speaker_id for c in chunks if c.speaker_id)
        logger.info(
            f"Diarization complete: {len(speakers_found)} unique speaker(s) identified"
        )

    except Exception as e:
        logger.warning(
            f"Speaker diarization failed (non-fatal): {e}. "
            "Chunks will have speaker_id=None."
        )

    return chunks


# ── File Type Detection ────────────────────────────────────────────────────
def _detect_mime_type(file_path: Path) -> str:
    """Detect MIME type using libmagic. Fallback to extension."""
    try:
        return magic.from_file(str(file_path), mime=True)
    except Exception as e:
        logger.warning(f"python-magic failed ({e}), using extension fallback")
        suffix = file_path.suffix.lower()
        fallback = {
            ".mp3":  "audio/mpeg",
            ".m4a":  "audio/x-m4a",
            ".wav":  "audio/wav",
            ".ogg":  "audio/ogg",
            ".flac": "audio/flac",
            ".mp4":  "video/mp4",
        }
        return fallback.get(suffix, "application/octet-stream")


def _is_supported_audio(mime_type: str, file_path: Path) -> bool:
    """Return True if MIME type is a supported audio format (with extension fallback)."""
    if mime_type in SUPPORTED_AUDIO_MIMES:
        return True
    # Extension fallback for Windows MIME misclassification
    ext = file_path.suffix.lower()
    return ext in (".mp3", ".m4a", ".wav", ".ogg", ".flac", ".mp4")


# ── Normalized Audio Saving ────────────────────────────────────────────────
def _get_normalized_audio_path(source_path: Path) -> Path:
    """
    Return the path where the normalized WAV for a source file will be saved.
    Normalized files are persisted in AUDIO_DIR for potential reuse.
    """
    safe_stem = re.sub(r"[^\w\-]", "_", source_path.stem)
    return AUDIO_DIR / f"{safe_stem}_normalized.wav"


# ── Public API ─────────────────────────────────────────────────────────────
class AudioIngestor:
    """
    Public interface for audio ingestion.

    Usage:
        ingestor = AudioIngestor()

        # Basic ingestion
        result = ingestor.ingest(Path("call_recording.mp3"), session_id="abc")
        # result.audio_chunks contains list of AudioChunk objects

        # With speaker diarization
        result = ingestor.ingest(
            Path("meeting.mp3"),
            session_id="abc",
            enable_diarization=True
        )
        # Each chunk will have speaker_id set (e.g., "SPEAKER_00", "SPEAKER_01")

    Notes:
        - Whisper is loaded and immediately unloaded after transcription
        - Normalized WAV files are saved to AUDIO_DIR for debugging/reuse
        - AudioChunk.embedding is None after this phase — BGE-M3 embedding
          is computed in Phase 4 alongside text chunk embeddings
    """

    def ingest(
        self,
        file_path: Path,
        session_id: Optional[str] = None,
        enable_diarization: bool = False,
    ) -> IngestionResult:
        """
        Ingest a single audio file.

        Args:
            file_path:           Path to audio file (MP3, M4A, WAV, OGG, FLAC)
            session_id:          Optional session identifier
            enable_diarization:  If True, run pyannote speaker diarization.
                                 Adds ~30-60s processing time on CPU.
                                 Requires pyannote.audio to be installed.

        Returns:
            IngestionResult with audio_chunks populated and any errors/warnings.
        """
        file_path = Path(file_path).resolve()
        result = IngestionResult(
            source_file=file_path.name,
            modality=Modality.AUDIO,
        )

        # ── Validation ─────────────────────────────────────────────────────
        if not file_path.exists():
            result.errors.append(f"File not found: {file_path}")
            logger.error(f"AudioIngestor: File not found: {file_path}")
            return result

        mime_type = _detect_mime_type(file_path)
        if not _is_supported_audio(mime_type, file_path):
            result.errors.append(
                f"Unsupported file type '{mime_type}' for {file_path.name}. "
                f"Supported: MP3, M4A, WAV, OGG, FLAC."
            )
            return result

        logger.info(
            f"AudioIngestor: Ingesting {file_path.name} "
            f"(session={session_id}, diarization={enable_diarization})"
        )

        # ── Step 1: Normalize Audio ────────────────────────────────────────
        wav_path = _normalize_audio(file_path, AUDIO_DIR)
        if wav_path is None:
            result.errors.append(
                f"Audio normalization failed for {file_path.name}. "
                "Check that ffmpeg is installed and accessible."
            )
            return result

        # ── Step 2: Silence Detection ──────────────────────────────────────
        nonsilent_ranges = _detect_non_silent_ranges(wav_path)
        if nonsilent_ranges:
            total_audio_ms = sum(end - start for start, end in nonsilent_ranges)
            logger.debug(
                f"Non-silent audio: {total_audio_ms / 1000:.1f}s "
                f"across {len(nonsilent_ranges)} segments"
            )

        # ── Step 3: Transcription ──────────────────────────────────────────
        # We transcribe the full normalized WAV (Whisper handles silence well
        # internally). The silence detection info is used for logging/reporting
        # but we don't slice the audio before transcription — doing so risks
        # losing cross-sentence context at boundaries.
        try:
            whisper_result = _transcribe_audio(wav_path)
        except RuntimeError as e:
            result.errors.append(str(e))
            return result

        if whisper_result is None:
            result.errors.append(
                f"Transcription failed for {file_path.name}. "
                "Check Whisper model is downloaded."
            )
            return result

        full_transcript = whisper_result.get("text", "").strip()
        if not full_transcript:
            result.warnings.append(
                f"Whisper returned empty transcript for {file_path.name}. "
                "File may be silent, contain only music, or be in an unsupported language."
            )
            return result

        logger.info(
            f"Transcript: {len(full_transcript)} chars, "
            f"language detected: {whisper_result.get('language', 'unknown')}"
        )

        # ── Step 4: Chunking ───────────────────────────────────────────────
        try:
            chunks = _chunk_transcript(
                whisper_result=whisper_result,
                audio_file=file_path.name,
                session_id=session_id,
            )
        except Exception as e:
            result.errors.append(f"Transcript chunking failed: {e}")
            logger.exception("AudioIngestor: Chunking error")
            return result

        if not chunks:
            result.warnings.append(
                f"No audio chunks produced from {file_path.name} "
                "despite successful transcription."
            )
            return result

        # ── Step 5: Speaker Diarization (Optional) ─────────────────────────
        if enable_diarization:
            try:
                chunks = _apply_diarization(chunks, wav_path)
            except Exception as e:
                result.warnings.append(
                    f"Diarization failed (non-fatal): {e}. "
                    "Chunks have speaker_id=None."
                )
                logger.warning(f"AudioIngestor: Diarization error: {e}")

        # ── Done ───────────────────────────────────────────────────────────
        result.audio_chunks = chunks

        logger.info(
            f"AudioIngestor: {file_path.name} complete — "
            f"{len(chunks)} audio chunks, "
            f"{len(result.warnings)} warnings, "
            f"{len(result.errors)} errors"
        )
        return result

    def ingest_batch(
        self,
        file_paths: List[Path],
        session_id: Optional[str] = None,
        enable_diarization: bool = False,
    ) -> List[IngestionResult]:
        """
        Ingest multiple audio files sequentially.
        Whisper is loaded and unloaded for EACH file to keep VRAM clean.

        Returns list of IngestionResult, one per file.
        """
        results = []
        for fp in file_paths:
            result = self.ingest(
                Path(fp),
                session_id=session_id,
                enable_diarization=enable_diarization,
            )
            results.append(result)
        return results

    def get_audio_duration(self, file_path: Path) -> Optional[float]:
        """
        Get duration of an audio file in seconds using ffmpeg.
        Useful for progress reporting before ingestion.

        Returns duration in seconds, or None if cannot be determined.
        """
        try:
            cmd = [
                "ffprobe",
                "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(file_path),
            ]
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30,
            )
            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
        except Exception as e:
            logger.debug(f"Could not get duration for {file_path.name}: {e}")
        return None
